"""
Draft tests for ScreenMirrored SLMs.
# TODO: increase rigor; adapt to v0.4.0 testing framework.

Tests both ScreenMirrored and PLM on a multi-monitor setup.
Run: .venv/Scripts/python.exe testing/hardware/slms/test_pyglet.py
"""

import time
import traceback
import numpy as np

from slmsuite.hardware.slms.screenmirrored import ScreenMirrored
from slmsuite.hardware.slms.texasinstruments import PLM

# --- Config ---
DISPLAY_1 = 1
DISPLAY_2 = 2
PLM_DEVICE = "test_monitor"
FREEZE_DELAY = 15       # seconds for freeze test
STRESS_ITERATIONS = 100
REOPEN_CYCLES = 3


def _gradient(shape):
    """Generate a horizontal gradient phase pattern."""
    return np.linspace(0, 2 * np.pi, shape[1], dtype=np.float32)[np.newaxis, :] * np.ones(shape[0], dtype=np.float32)[:, np.newaxis]


results = []

def run_test(name, func):
    """Run a test function and record PASS/FAIL."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        func()
        print(f"  -> PASS")
        results.append((name, "PASS"))
    except Exception:
        traceback.print_exc()
        print(f"  -> FAIL")
        results.append((name, "FAIL"))


# --- Test 1: Display detection ---
def test_display_detection():
    info = ScreenMirrored.info(verbose=True)
    n = len(info)
    print(f"  Found {n} displays")
    assert n >= 3, f"Expected >= 3 displays, got {n}"


# --- Test 2: Basic display ---
def test_basic_display_screenmirrored():
    slm = ScreenMirrored(DISPLAY_1, verbose=True)
    phase = _gradient(slm.shape)
    slm.set_phase(phase)
    time.sleep(1)
    slm.close()

def test_basic_display_plm():
    slm = PLM(PLM_DEVICE, DISPLAY_1, verbose=True)
    phase = _gradient(slm.shape)
    slm.set_phase(phase)
    time.sleep(1)
    slm.close()


# --- Test 3: Freeze test (Issue #117) ---
def test_freeze_screenmirrored():
    slm = ScreenMirrored(DISPLAY_1, verbose=True)
    slm.set_phase(None)
    print(f"  Idling {FREEZE_DELAY}s (interact with other windows)...")
    time.sleep(FREEZE_DELAY)
    print("  Updating phase after idle...")
    slm.set_phase(_gradient(slm.shape))
    time.sleep(0.5)
    slm.close()

def test_freeze_plm():
    slm = PLM(PLM_DEVICE, DISPLAY_1, verbose=True)
    slm.set_phase(None)
    print(f"  Idling {FREEZE_DELAY}s (interact with other windows)...")
    time.sleep(FREEZE_DELAY)
    print("  Updating phase after idle...")
    slm.set_phase(_gradient(slm.shape))
    time.sleep(0.5)
    slm.close()


# --- Test 4: Multi-SLM ---
def test_multi_slm_screenmirrored():
    slm1 = ScreenMirrored(DISPLAY_1, verbose=True)
    slm2 = ScreenMirrored(DISPLAY_2, verbose=True)
    for i in range(4):
        phase1 = np.full(slm1.shape, (i % 2) * np.pi, dtype=np.float32)
        phase2 = np.full(slm2.shape, ((i + 1) % 2) * np.pi, dtype=np.float32)
        slm1.set_phase(phase1)
        slm2.set_phase(phase2)
        time.sleep(0.5)
    slm1.close()
    slm2.close()

def test_multi_slm_plm():
    slm1 = PLM(PLM_DEVICE, DISPLAY_1, verbose=True)
    slm2 = PLM(PLM_DEVICE, DISPLAY_2, verbose=True)
    for i in range(4):
        phase1 = np.full(slm1.shape, (i % 2) * np.pi, dtype=np.float32)
        phase2 = np.full(slm2.shape, ((i + 1) % 2) * np.pi, dtype=np.float32)
        slm1.set_phase(phase1)
        slm2.set_phase(phase2)
        time.sleep(0.5)
    slm1.close()
    slm2.close()

def test_multi_slm_mixed():
    slm1 = ScreenMirrored(DISPLAY_1, verbose=True)
    slm2 = PLM(PLM_DEVICE, DISPLAY_2, verbose=True)
    for i in range(4):
        phase1 = np.full(slm1.shape, (i % 2) * np.pi, dtype=np.float32)
        phase2 = np.full(slm2.shape, ((i + 1) % 2) * np.pi, dtype=np.float32)
        slm1.set_phase(phase1)
        slm2.set_phase(phase2)
        time.sleep(0.5)
    slm1.close()
    slm2.close()


# --- Test 5: Close/reopen ---
def test_close_reopen_screenmirrored():
    for cycle in range(REOPEN_CYCLES):
        print(f"  Cycle {cycle + 1}/{REOPEN_CYCLES}")
        slm = ScreenMirrored(DISPLAY_1, verbose=False)
        slm.set_phase(_gradient(slm.shape))
        time.sleep(0.3)
        slm.close()

def test_close_reopen_plm():
    for cycle in range(REOPEN_CYCLES):
        print(f"  Cycle {cycle + 1}/{REOPEN_CYCLES}")
        slm = PLM(PLM_DEVICE, DISPLAY_1, verbose=False)
        slm.set_phase(_gradient(slm.shape))
        time.sleep(0.3)
        slm.close()


# --- Test 6: Stress test ---
def test_stress_screenmirrored():
    slm = ScreenMirrored(DISPLAY_1, verbose=True)
    phases = [np.random.rand(*slm.shape).astype(np.float32) * 2 * np.pi
              for _ in range(STRESS_ITERATIONS)]
    t0 = time.perf_counter()
    for phase in phases:
        slm.set_phase(phase)
    elapsed = time.perf_counter() - t0
    fps = STRESS_ITERATIONS / elapsed
    print(f"  {STRESS_ITERATIONS} frames in {elapsed:.2f}s = {fps:.1f} fps")
    slm.close()

def test_stress_plm():
    slm = PLM(PLM_DEVICE, DISPLAY_1, verbose=True)
    phases = [np.random.rand(*slm.shape).astype(np.float32) * 2 * np.pi
              for _ in range(STRESS_ITERATIONS)]
    t0 = time.perf_counter()
    for phase in phases:
        slm.set_phase(phase)
    elapsed = time.perf_counter() - t0
    fps = STRESS_ITERATIONS / elapsed
    print(f"  {STRESS_ITERATIONS} frames in {elapsed:.2f}s = {fps:.1f} fps")
    slm.close()


if __name__ == "__main__":
    print("Appendix B Verification Tests")
    print("=" * 60)

    # Uncomment tests as needed. Tests 1-2 are quick; 3+ take longer.
    run_test("1. Display detection", test_display_detection)
    run_test("2a. Basic display (ScreenMirrored)", test_basic_display_screenmirrored)
    run_test("2b. Basic display (PLM)", test_basic_display_plm)
    run_test("3a. Freeze test (ScreenMirrored)", test_freeze_screenmirrored)
    run_test("3b. Freeze test (PLM)", test_freeze_plm)
    run_test("4a. Multi-SLM (ScreenMirrored)", test_multi_slm_screenmirrored)
    run_test("4b. Multi-SLM (PLM)", test_multi_slm_plm)
    run_test("4c. Multi-SLM (mixed)", test_multi_slm_mixed)
    run_test("5a. Close/reopen (ScreenMirrored)", test_close_reopen_screenmirrored)
    run_test("5b. Close/reopen (PLM)", test_close_reopen_plm)
    run_test("6a. Stress test (ScreenMirrored)", test_stress_screenmirrored)
    run_test("6b. Stress test (PLM)", test_stress_plm)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, result in results:
        print(f"  {result}: {name}")
    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    print(f"\n{passed}/{total} passed")
