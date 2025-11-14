import subprocess

def run_exec(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()

    out = subprocess.check_output(cmd).decode().strip().split()

    nums = []
    for tok in out:
        try:
            nums.append(float(tok))
        except:
            pass
    return nums

def bench_gemm():
    print("\n GEMM")
    sizes = [(256,256,256), (512,512,512), (1024,1024,1024)]

    print(f"{'M':>6} {'N':>6} {'K':>6} | {'CPU Time':>10} {'CPU GFLOPS':>12} {'CPU GB/s':>10} | "
          f"{'GPU Time':>10} {'GPU GFLOPS':>12} {'GPU GB/s':>10}")

    for (M,N,K) in sizes:
        cpu = run_exec(f"kernels\gemm_cpu.exe {M} {N} {K}")
        gpu = run_exec(f"kernels\gemm_gpu.exe {M} {N} {K}")

        print(f"{M:6} {N:6} {K:6} | {cpu[0]:10.6f} {cpu[1]:12.4f} {cpu[2]:10.4f} | "
              f"{gpu[0]:10.6f} {gpu[1]:12.4f} {gpu[2]:10.4f}")

def bench_spmv():
    print("\n SPMV")
    rows = [10000, 20000, 50000]
    nnz   = [100000, 200000, 500000]

    print(f"{'rows':>8} {'nnz':>10} | {'CPU Time':>10} {'CPU GFLOPS':>12} {'CPU GB/s':>10} | "
          f"{'GPU Time':>10} {'GPU GFLOPS':>12} {'GPU GB/s':>10}")

    for r, z in zip(rows, nnz):
        cpu = run_exec(f"kernels\spmv_cpu.exe {r} {z}")
        gpu = run_exec(f"kernels\spmv_gpu.exe {r} {z}")

        print(f"{r:8} {z:10} | {cpu[0]:10.6f} {cpu[1]:12.4f} {cpu[2]:10.4f} | "
              f"{gpu[0]:10.6f} {gpu[1]:12.4f} {gpu[2]:10.4f}")

def bench_fft():
    print("\n FFT")
    sizes = [1024, 4096, 16384, 65536]

    print(f"{'N':>8} | {'CPU Time':>10} {'CPU GFLOPS':>12} {'CPU GB/s':>10} | {'GPU Time':>10} {'GPU GFLOPS':>10} {'GPU GB/s':>10}")

    for N in sizes:
        cpu = run_exec(f"kernels/fft_cpu.exe {N}")
        gpu = run_exec(f"kernels/fft_gpu.exe {N}")

        print(f"{N:8} | {cpu[0]:10.6f} {cpu[1]:12.4f} {cpu[2]:10.4f} | {gpu[0]:10.6f} {gpu[1]:10.4f} {gpu[2]:10.4f}")

if __name__ == "__main__":
    bench_gemm()
    bench_spmv()
    bench_fft()