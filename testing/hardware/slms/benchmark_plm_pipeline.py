"""
Benchmark PLM pipeline stages to identify performance bottlenecks.

Profiles each stage of the PLM display pipeline (phase wrapping, quantization,
electrode mapping, GPU->CPU transfer, rendering) with both NumPy and CuPy backends.
Uses CUDA events for accurate GPU timing.

Usage::

    python benchmark_plm_pipeline.py p67 1
    python benchmark_plm_pipeline.py p67 1 --iterations 5000
    python benchmark_plm_pipeline.py p67 1 --configure-usb
"""

import argparse
import time
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from ti_plm import PLM as TI_PLM
from slmsuite.hardware.slms.texasinstruments import PLM


def bench(fn, inputs, use_cuda=False, warmup=50):
    """
    Benchmark ``fn(input)`` over all inputs, return ms per call.

    Parameters
    ----------
    fn : callable
        Function to benchmark (called once per input).
    inputs : list
        Pre-generated input arrays.
    use_cuda : bool
        If True, use CUDA events for GPU timing.
    warmup : int
        Number of warmup iterations before measurement.

    Returns
    -------
    float
        Milliseconds per call.
    """
    n = len(inputs)
    for i in range(min(warmup, n)):
        fn(inputs[i])

    if use_cuda:
        cp.cuda.Stream.null.synchronize()
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for inp in inputs:
            fn(inp)
        end.record()
        end.synchronize()
        total_ms = cp.cuda.get_elapsed_time(start, end)
    else:
        t0 = time.perf_counter()
        for inp in inputs:
            fn(inp)
        total_ms = (time.perf_counter() - t0) * 1000

    return total_ms / n


def run_ti_plm_benchmarks(ti_plm, shape, iterations):
    """Run quantize and electrode_map benchmarks for ti_plm reference."""
    phases = [
        np.random.rand(*shape).astype(np.float32) * 2 * np.pi
        for _ in range(iterations)
    ]

    timings = {}

    timings["Quantize"] = bench(
        lambda p: ti_plm.quantize(p),
        phases, use_cuda=False
    )

    quantized = [ti_plm.quantize(p) for p in phases]
    timings["Electrode map"] = bench(
        lambda q: ti_plm.electrode_map(q),
        quantized, use_cuda=False
    )

    return timings


def run_benchmarks(slm, xp, iterations, label):
    """Run all pipeline stage benchmarks for a given backend."""
    use_cuda = xp is not None and xp.__name__ == "cupy"

    # Pre-generate random phase arrays
    phases = [
        xp.random.rand(*slm.shape).astype(xp.float32) * 2 * xp.pi
        for _ in range(iterations)
    ]

    timings = {}

    # Stage 1: Phase wrapping
    timings["Phase wrap"] = bench(
        lambda p: p % (2 * np.pi),
        phases, use_cuda
    )

    # Stage 2: Quantization (includes wrapping)
    timings["Quantize"] = bench(
        lambda p: slm._quantize(p),
        phases, use_cuda
    )

    # Stage 3: Electrode mapping only (pre-quantized input)
    quantized = [slm._quantize(p) for p in phases]
    timings["Electrode map"] = bench(
        lambda q: slm._electrode_map(q),
        quantized, use_cuda
    )

    # Stage 4: Bit replication only
    mapped = [slm._electrode_map(q) for q in quantized]
    timings["Bit replicate"] = bench(
        lambda m: m * 255,
        mapped, use_cuda
    )

    # Stage 5: Full _format_phase_hw
    timings["_format_phase_hw"] = bench(
        lambda p: slm._format_phase_hw(p),
        phases, use_cuda
    )

    # Stage 6: GPU->CPU transfer (cupy only)
    if use_cuda:
        formatted = [slm._format_phase_hw(p) for p in phases]
        timings["GPU->CPU xfer"] = bench(
            lambda f: cp.asnumpy(f),
            formatted, use_cuda
        )

    # Stage 7: set_phase without display write
    timings["set_phase(no exec)"] = bench(
        lambda p: slm.set_phase(p, execute=False),
        phases, use_cuda
    )

    # Stage 8: Full set_phase with display
    timings["set_phase(full)"] = bench(
        lambda p: slm.set_phase(p, execute=True, block=True),
        phases, use_cuda
    )

    return timings


def print_results(all_results):
    """Print a formatted comparison table."""
    # Collect all stage names in order
    all_stages = []
    for _, timings in all_results:
        for stage in timings:
            if stage not in all_stages:
                all_stages.append(stage)

    # Find NumPy and CuPy indices for speedup calculation
    numpy_idx = next((i for i, (l, _) in enumerate(all_results) if "NumPy" in l), None)
    cupy_idx = next((i for i, (l, _) in enumerate(all_results) if "CuPy" in l), None)
    show_speedup = numpy_idx is not None and cupy_idx is not None

    # Header
    header = f"{'Stage':<22}"
    for label, _ in all_results:
        header += f" | {label + ' ms':>10}  {label + ' fps':>10}"
    if show_speedup:
        header += f" | {'Speedup':>8}"
    print()
    print(header)
    print("-" * len(header))

    # Rows
    for stage in all_stages:
        row = f"{stage:<22}"
        values = []
        for label, timings in all_results:
            if stage in timings:
                ms = timings[stage]
                fps = 1000.0 / ms if ms > 0 else float("inf")
                row += f" | {ms:>10.3f}  {fps:>10.1f}"
                values.append(ms)
            else:
                row += f" | {'N/A':>10}  {'N/A':>10}"
                values.append(None)
        if show_speedup:
            np_ms = values[numpy_idx]
            cp_ms = values[cupy_idx]
            if np_ms is not None and cp_ms is not None and cp_ms > 0:
                row += f" | {np_ms / cp_ms:>7.1f}x"
            else:
                row += f" | {'N/A':>8}"
        print(row)

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PLM pipeline stages"
    )
    parser.add_argument("device", help="Device name from texas_instruments.json (e.g. p67)")
    parser.add_argument("display", type=int, help="Display number")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Number of iterations per benchmark (default: 1000)")
    parser.add_argument("--configure-usb", action="store_true",
                        help="Configure DLPC900 via USB before benchmarking")
    args = parser.parse_args()

    print(f"Benchmarking PLM '{args.device}' on display {args.display}")
    print(f"Iterations: {args.iterations}")

    all_results = []

    # ti_plm reference benchmark (NumPy only)
    print("\n--- ti_plm (NumPy) reference ---")
    ti_plm = TI_PLM.from_db(args.device)
    shape = tuple(int(s) for s in ti_plm.shape)
    timings_ti = run_ti_plm_benchmarks(ti_plm, shape, args.iterations)
    all_results.append(("ti_plm", timings_ti))

    # NumPy benchmark (force CPU backend)
    print("\n--- slmsuite NumPy (CPU) backend ---")
    slm_cpu = PLM(args.device, args.display, gpu=False,
                  configure_usb=args.configure_usb)
    timings_np = run_benchmarks(slm_cpu, np, args.iterations, "NumPy")
    all_results.append(("NumPy", timings_np))
    slm_cpu.close()

    # CuPy benchmark (GPU backend)
    if cp is not None:
        print("\n--- slmsuite CuPy (GPU) backend ---")
        slm_gpu = PLM(args.device, args.display, gpu=True,
                      configure_usb=args.configure_usb)
        timings_cp = run_benchmarks(slm_gpu, cp, args.iterations, "CuPy")
        all_results.append(("CuPy", timings_cp))
        slm_gpu.close()
    else:
        print("\nCuPy not available, skipping GPU benchmarks.")

    print_results(all_results)


if __name__ == "__main__":
    main()
