"""Host-side algorithm gate for sparse_flash_attention_grad (no Ascend needed).

Two independent numpy checks that pin down "the math and the golden are correct",
so any later NPU failure is attributable to the kernel/triton lowering, not the
algorithm:

  1. finite-difference: backward golden vs numerical gradient of the forward golden
     (loss = sum(out * d_out)), for sparse_mode 0 and 3.
  2. kernel-sim: a numpy 1:1 mirror of the triton kernel algorithm (rebuild P from
     saved softmax_max/sum, dS = P*(dO·k - delta)*scale, scatter dk/dkr/dv) vs the
     backward golden, in fp32.

Run:  python verify_grad_algo.py     (exit 0 = all pass)
"""
import sys
import numpy as np

from sparse_flash_attention_numpy import sparse_flash_attention_golden_bsnd as fwd
from sparse_flash_attention_grad_numpy import sparse_flash_attention_grad_golden_bsnd as bwd


def _make_case(rng, B, S1, S2, N1, D, Dr, sc, mode):
    qn = rng.randn(B, S1, N1, D) * 0.3
    kn = rng.randn(B, S2, 1, D) * 0.3
    qr = rng.randn(B, S1, N1, Dr) * 0.3
    kr = rng.randn(B, S2, 1, Dr) * 0.3
    v = kn.copy()
    scale = 1.0 / np.sqrt(D + Dr)
    do = rng.randn(B, S1, N1, D)
    aq, ak = [S1] * B, [S2] * B
    si = np.full((B, S1, 1, sc), -1, np.int32)
    for b in range(B):
        for s in range(S1):
            th = ak[b] if mode == 0 else ak[b] - aq[b] + s + 1
            if th <= 0:
                continue
            n = min(sc, th)
            si[b, s, 0, :n] = np.sort(rng.permutation(th)[:n]).astype(np.int32)
    return qn, kn, qr, kr, v, scale, do, aq, ak, si


def check_finite_diff():
    """backward golden vs central finite-difference of forward golden."""
    rng = np.random.RandomState(0)
    ok = True
    for mode in (0, 3):
        qn, kn, qr, kr, v, scale, do, aq, ak, si = _make_case(
            rng, 1, 4, 8, 2, 16, 8, 4, mode)

        def forward(qn, kn, qr, kr):
            o, _, _ = fwd(qn, kn, v, si, qr, kr, scale, aq, ak,
                          sparse_mode=mode, dtype=np.float32)
            return o

        out = forward(qn, kn, qr, kr)
        dq, dk, dv, dqr, dkr = bwd(qn, kn, v, si, do, out, *fwd(
            qn, kn, v, si, qr, kr, scale, aq, ak, sparse_mode=mode,
            dtype=np.float32)[1:], qr, kr, scale, aq, ak,
            sparse_mode=mode, dtype=np.float32)

        def loss(qn, kn, qr, kr):
            return float(np.sum(forward(qn, kn, qr, kr) * do))

        eps = 1e-4
        names = "qn kn qr kr".split()

        def fd(which, arr, idx):
            a = arr.copy(); a[idx] += eps
            lp = loss(*[a if w == which else x
                        for w, x in zip(names, (qn, kn, qr, kr))])
            a = arr.copy(); a[idx] -= eps
            lm = loss(*[a if w == which else x
                        for w, x in zip(names, (qn, kn, qr, kr))])
            return (lp - lm) / (2 * eps)

        # kn receives BOTH dk and dv (v == kn), so analytic d(kn) = dk + dv.
        tests = [("dq", dq, "qn", qn, (0, 1, 0, 3)),
                 ("dqr", dqr, "qr", qr, (0, 2, 1, 2)),
                 ("dkr", dkr, "kr", kr, (0, 3, 0, 1)),
                 ("dk+dv", dk + dv, "kn", kn, (0, 4, 0, 5))]
        for name, grad, which, arr, idx in tests:
            num = fd(which, arr, idx)
            diff = abs(grad[idx] - num)
            tag = "ok" if diff < 1e-3 else "FAIL"
            ok &= diff < 1e-3
            print(f"  mode{mode} {name:6s} analytic={grad[idx]:+.6f} "
                  f"fd={num:+.6f} diff={diff:.2e} {tag}")
    return ok


def _kernel_sim(qn, kn, qr, kr, si, do, out, smax, ssum, scale, aq, ak, mode):
    """numpy mirror of the triton kernel algorithm (see module docstring)."""
    B, S1, N1, D = qn.shape
    S2 = kn.shape[1]; Dr = qr.shape[-1]; topK = si.shape[-1]
    dq = np.zeros_like(qn); dqr = np.zeros_like(qr)
    dk = np.zeros((B, S2, 1, D)); dkr = np.zeros((B, S2, 1, Dr))
    dv = np.zeros((B, S2, 1, D))
    for b in range(B):
        aqb, akb = int(aq[b]), int(ak[b])
        for s1 in range(S1):
            if not (s1 < aqb):
                continue
            th = akb if mode == 0 else akb - aqb + s1 + 1
            if th <= 0:
                continue
            for g in range(N1):
                delta = float(do[b, s1, g] @ out[b, s1, g])
                acc_dq = np.zeros(D); acc_dqr = np.zeros(Dr)
                for kk in range(topK):
                    tok = int(si[b, s1, 0, kk])
                    if not ((tok != -1) and (tok < th) and (tok < akb)):
                        continue
                    score = (qn[b, s1, g] @ kn[b, tok, 0]
                             + qr[b, s1, g] @ kr[b, tok, 0]) * scale
                    P = np.exp(score - smax[b, 0, s1, g]) / ssum[b, 0, s1, g]
                    dS = P * (float(do[b, s1, g] @ kn[b, tok, 0]) - delta) * scale
                    acc_dq += dS * kn[b, tok, 0]
                    acc_dqr += dS * kr[b, tok, 0]
                    dk[b, tok, 0] += dS * qn[b, s1, g]
                    dkr[b, tok, 0] += dS * qr[b, s1, g]
                    dv[b, tok, 0] += P * do[b, s1, g]
                dq[b, s1, g] = acc_dq; dqr[b, s1, g] = acc_dqr
    return dq, dk, dv, dqr, dkr


def check_kernel_sim():
    """numpy kernel-sim vs backward golden (fp32)."""
    rng = np.random.RandomState(1)
    ok = True
    for mode in (0, 3):
        qn, kn, qr, kr, v, scale, do, aq, ak, si = _make_case(
            rng, 2, 6, 16, 4, 32, 8, 5, mode)
        out, smax, ssum = fwd(qn, kn, v, si, qr, kr, scale, aq, ak,
                              sparse_mode=mode, dtype=np.float32)
        g = bwd(qn, kn, v, si, do, out, smax, ssum, qr, kr, scale, aq, ak,
                sparse_mode=mode, dtype=np.float32)
        k = _kernel_sim(qn, kn, qr, kr, si, do, out, smax, ssum,
                        scale, aq, ak, mode)
        names = ("dq", "dk", "dv", "dqr", "dkr")
        case_ok = True
        for n, a, b in zip(names, g, k):
            close = np.allclose(a, b, rtol=1e-5, atol=1e-6)
            case_ok &= close
        ok &= case_ok
        print(f"  mode{mode} kernel-sim vs golden: "
              f"{'PASS' if case_ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("[1] finite-difference: backward golden vs forward golden")
    fd_ok = check_finite_diff()
    print("[2] kernel-sim: numpy kernel mirror vs backward golden")
    ks_ok = check_kernel_sim()
    allok = fd_ok and ks_ok
    print("ALL PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)
