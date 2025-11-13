import argparse, sys, os, json, subprocess, time, threading, csv, tempfile
import pulp
from contextlib import contextmanager
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage

# energia en GPU (J)
@contextmanager
def gpu_monitor(interval=0.05): # intervalo de muestreo en segundos (0.05s)
    nvmlInit()
    # 
    handle = nvmlDeviceGetHandleByIndex(0)
    samples = []
    stop_flag = [False]

    def sampler():
        while not stop_flag[0]:
            try:
                power_w = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Salida en W
                samples.append((time.time(), power_w))
            except:
                pass
            time.sleep(interval)

    t = threading.Thread(target=sampler)
    t.start()
    start_time = time.time()

    try:
        yield samples
    finally:
        stop_flag[0] = True
        t.join()
        nvmlShutdown()

        # calcular energia total usando integracion trapezoidal
        energy = 0.0
        for i in range(1, len(samples)):
            dt = samples[i][0] - samples[i-1][0]
            avg_p = (samples[i][1] + samples[i-1][1]) / 2
            energy += avg_p * dt

        print(f"[GPU Monitor] Tiempo total: {time.time() - start_time:.2f}s, Energía total: {energy:.2f}J")

# energia en CPU (J)
# falta...

def compile_kernel(op, backend):
    src_dir = "./kernels"
    ext = "cpp" if backend == "CPU" else "cu"
    src = os.path.join(src_dir, f"{op.lower()}_{backend.lower()}.{ext}")
    exe = os.path.join(src_dir, f"{op.lower()}_{backend.lower()}.exe")

    if os.path.exists(exe):
        return exe

    print(f"[Compilando] {src} → {exe}")
    if backend == "CPU":
        cmd = ["g++", "-O3", src, "-o", exe]
    else:
        cmd = ["nvcc", "-O3", src, "-o", exe]

    subprocess.run(cmd, check=True)
    return exe

def run_kernel_real(op, params, backend):
    exe_path = compile_kernel(op, backend)
    cmd = [exe_path]

    if op == "GEMM":
        cmd += [str(params["M"]), str(params["N"]), str(params["K"])]
    elif op == "SpMV":
        cmd += [str(params["rows"]), str(params["nnz"])]
    elif op == "FFT":
        cmd += [str(params["N"])]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        elapsed = float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Kernel {op} ({backend}) no devolvió un tiempo válido.")
    return elapsed

def get_cplex_solver_instance(msg=False):
    try:
        solver = pulp.CPLEX_PY(msg=msg)
        prob = pulp.LpProblem("test_cplex", pulp.LpMinimize)
        x = pulp.LpVariable("x", lowBound=0)
        prob += x
        res = prob.solve(solver)
        if pulp.LpStatus[prob.status] in ("Optimal", "Optimal Solution Found"):
            return solver
    except Exception:
        pass
    raise RuntimeError("CPLEX Error")


def build_and_solve_ilp_cplex(op, params, alpha, beta):
    print(f"\n=== Midiendo CPU ===")
    t_cpu, e_cpu = measure_time_energy(op, params, "CPU")

    print(f"\n=== Midiendo GPU ===")
    t_gpu, e_gpu = measure_time_energy(op, params, "GPU")

    print("\n=== ILP con CPLEX ===")
    prob = pulp.LpProblem("placement", pulp.LpMinimize)
    x_cpu = pulp.LpVariable("x_cpu", cat="Binary")
    x_gpu = pulp.LpVariable("x_gpu", cat="Binary")

    prob += x_cpu + x_gpu == 1
    prob += alpha * (t_cpu*x_cpu + t_gpu*x_gpu) + beta * (e_cpu*x_cpu + e_gpu*x_gpu)

    solver = get_cplex_solver_instance(msg=False)
    res = prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    decision = {"x_cpu": pulp.value(x_cpu), "x_gpu": pulp.value(x_gpu)}
    objective = pulp.value(prob.objective)

    return {
        "status": status,
        "decision": decision,
        "objective": objective,
        "measured": {
            "cpu": {"time_s": t_cpu, "energy_j": e_cpu},
            "gpu": {"time_s": t_gpu, "energy_j": e_gpu}
        },
        "alpha": alpha,
        "beta": beta,
        "op": op,
        "params": params
    }

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["GEMM", "SpMV", "FFT"], required=True)
    p.add_argument("--M", type=int)
    p.add_argument("--N", type=int)
    p.add_argument("--K", type=int)
    p.add_argument("--rows", type=int)
    p.add_argument("--nnz", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--beta", type=float)
    p.add_argument("--repeats", type=int)
    p.add_argument("--print-json", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    params = {}

    if args.op == "GEMM":
        if args.M is None or args.N is None or args.K is None:
            print("GEMM requiere --M --N --K"); sys.exit(1)
        params = {"M": args.M, "N": args.N, "K": args.K}
    elif args.op == "SpMV":
        if args.rows is None or args.nnz is None:
            print("SpMV requiere --rows --nnz"); sys.exit(1)
        params = {"rows": args.rows, "nnz": args.nnz}
    elif args.op == "FFT":
        if args.N is None:
            print("FFT requiere --N"); sys.exit(1)
        params = {"N": args.N}

    try:
        res = build_and_solve_ilp_cplex(args.op, params, alpha=args.alpha, beta=args.beta)
    except RuntimeError as e:
        print("ERROR:", e)
        sys.exit(2)

    print("\n=== RESULTADOS ===")
    print("Status:", res["status"])
    print("Operación:", res["op"], "Parámetros:", res["params"])
    print("CPU:  tiempo={:.6f}s  energía={:.6f}J".format(
        res["measured"]["cpu"]["time_s"], res["measured"]["cpu"]["energy_j"]))
    print("GPU:  tiempo={:.6f}s  energía={:.6f}J".format(
        res["measured"]["gpu"]["time_s"], res["measured"]["gpu"]["energy_j"]))
    print("Decisión ILP: CPU={}  GPU={}".format(
        res["decision"]["x_cpu"], res["decision"]["x_gpu"]))
    print("Objetivo:", res["objective"])

    if args.print_json:
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()

# Ejemplos:
# python ilp_cplex.py --op GEMM --M 1024 --N 1024 --K 1024 --alpha 0.5 --beta 0.5
# python ilp_cplex.py --op SpMV --rows 1000000 --nnz 5000000 --alpha 0.5 --beta 0.5
# python ilp_cplex.py --op FFT --N 1048576 --alpha 0.5 --beta 0.5