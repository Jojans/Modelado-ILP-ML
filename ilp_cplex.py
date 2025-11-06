import argparse, math, sys, os, json
import subprocess
import time

from pprint import pprint

try:
    import pulp
except Exception as e:
    print("Instala PuLP: pip install pulp")
    raise

PERF = {
    "CPU": {"GEMM":150.0, "SpMV":10.0, "FFT":50.0},
    "GPU": {"GEMM":5000.0, "SpMV":200.0, "FFT":2000.0}
}
POWER = {"CPU":80.0, "GPU":200.0}
OVERHEAD = {"CPU":0.0, "GPU":0.0005}
MEM_BW = {"H2D":12e9, "D2H":12e9}

def flop_count(op, params):
    if op == "GEMM":
        M=int(params["M"]); N=int(params["N"]); K=int(params["K"])
        return 2.0*M*N*K
    if op == "SpMV":
        nnz=int(params.get("nnz",0)); return 2.0*nnz
    if op == "FFT":
        N=int(params["N"]); return 5.0 * N * math.log2(N)
    raise ValueError("Operación desconocida")

def mem_transfer_bytes(op, params):
    if op=="GEMM":
        M=int(params["M"]); N=int(params["N"]); K=int(params["K"])
        return 4.0*(M*K + K*N + M*N)
    if op=="SpMV":
        rows=int(params.get("rows",0)); nnz=int(params.get("nnz",0))
        return 4.0*(nnz + nnz + (rows+1)) + 4.0*rows + 4.0*rows
    if op=="FFT":
        N=int(params["N"]); return 8.0 * N * 2
    return 0.0

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

    # Pasar parámetros según el tipo de operación
    if op == "GEMM":
        cmd += [str(params["M"]), str(params["N"]), str(params["K"])]
    elif op == "SpMV":
        cmd += [str(params["rows"]), str(params["nnz"])]
    elif op == "FFT":
        cmd += [str(params["N"])]

    start = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    end = time.time()

    # Si el kernel imprime el tiempo, úsalo; si no, calcula diferencia
    try:
        elapsed = float(result.stdout.strip())
        if elapsed <= 0:
            elapsed = end - start
    except ValueError:
        elapsed = end - start

    energy = elapsed * POWER[backend]
    return elapsed, energy

def estimate_time_energy(op, params, backend):
    try:
        # 🔹 Usa medición real si el ejecutable existe o puede compilarse
        return run_kernel_real(op, params, backend)
    except Exception as e:
        # 🔹 Si no se puede medir, cae al estimado teórico
        print(f"[WARN] No se pudo ejecutar kernel real ({backend}): {e}")
        work = flop_count(op, params)
        perf_gflops = PERF[backend][op]
        time_compute = work / (perf_gflops * 1e9)
        mem_bytes = mem_transfer_bytes(op, params) if backend == "GPU" else 0.0
        time_mem = mem_bytes / MEM_BW["H2D"] if mem_bytes > 0 else 0.0
        time_total = OVERHEAD[backend] + time_compute + time_mem
        energy = time_total * POWER[backend]
        return time_total, energy

def get_cplex_solver_instance(msg=False):
    try:
        solver = pulp.CPLEX_PY(msg=msg)
        prob = pulp.LpProblem("test_cplex", pulp.LpMinimize)
        x = pulp.LpVariable("x", lowBound=0)
        prob += x
        res = prob.solve(solver)
        if pulp.LpStatus[prob.status] in ("Optimal","Optimal Solution Found"):
            return pulp.CPLEX_PY(msg=msg)
    except Exception:
        pass

    raise RuntimeError(
        "CPLEX no está disponible. "
    )

def build_and_solve_ilp_cplex(op, params, alpha=0.5, beta=0.5):
    t_cpu, e_cpu = estimate_time_energy(op, params, "CPU")
    t_gpu, e_gpu = estimate_time_energy(op, params, "GPU")

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
        "estimates": {"cpu":{"time_s":t_cpu,"energy_j":e_cpu}, "gpu":{"time_s":t_gpu,"energy_j":e_gpu}},
        "alpha":alpha, "beta":beta, "op":op, "params":params
    }

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["GEMM","SpMV","FFT"], required=True)
    p.add_argument("--M", type=int); p.add_argument("--N", type=int); p.add_argument("--K", type=int)
    p.add_argument("--rows", type=int); p.add_argument("--nnz", type=int)
    p.add_argument("--alpha", type=float, default=1.0); p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--print-json", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    params = {}
    if args.op=="GEMM":
        if args.M is None or args.N is None or args.K is None:
            print("GEMM requiere --M --N --K"); sys.exit(1)
        params={"M":args.M,"N":args.N,"K":args.K}
    elif args.op=="SpMV":
        if args.rows is None or args.nnz is None:
            print("SpMV requiere --rows --nnz"); sys.exit(1)
        params={"rows":args.rows,"nnz":args.nnz}
    elif args.op=="FFT":
        if args.N is None:
            print("FFT requiere --N"); sys.exit(1)
        params={"N":args.N}

    try:
        res = build_and_solve_ilp_cplex(args.op, params, alpha=args.alpha, beta=args.beta)
    except RuntimeError as e:
        print("ERROR:", e)
        sys.exit(2)

    print("Status:", res["status"])
    print("Op:", res["op"], "Params:", res["params"])
    print("Est CPU: time={:.6e}s energy={:.6e}J".format(res["estimates"]["cpu"]["time_s"], res["estimates"]["cpu"]["energy_j"]))
    print("Est GPU: time={:.6e}s energy={:.6e}J".format(res["estimates"]["gpu"]["time_s"], res["estimates"]["gpu"]["energy_j"]))
    print("Decision: CPU={}, GPU={}".format(res["decision"]["x_cpu"], res["decision"]["x_gpu"]))
    print("Objective:", res["objective"])
    if args.print_json:
        print(json.dumps(res, indent=2))

if __name__=="__main__":
    main()

# python ilp_cplex.py --op GEMM --M 1024 --N 1024 --K 1024
# python ilp_cplex.py --op SpMV --rows 1000000 --nnz 5000000
# python ilp_cplex.py --op FFT --N 1048576