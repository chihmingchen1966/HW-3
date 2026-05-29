#!/usr/bin/env python3
"""
HW3 Problem 5 - Sample-based subspace diagonalization on LiH.

This script implements a QSCI-style SQD workflow for LiH/STO-3G at R=1.5957 Angstrom.
It uses PySCF to obtain RHF, CCSD amplitudes, and the full active-space Hamiltonian,
then samples determinants from a CCSD/UCCSD-inspired measurement distribution.

Required packages:
    pip install pyscf numpy scipy matplotlib

Run:
    python problem5_main.py --seed 10010022 --outdir problem5_outputs
"""
import argparse
import csv
import itertools
import os
from pathlib import Path

# Avoid BLAS oversubscription on small CI matrices.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

try:
    from pyscf import ao2mo, cc, fci, gto, scf
    from pyscf.fci import cistring, direct_spin1
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PySCF is required for this script. Install with: pip install pyscf\n"
        f"Original import error: {exc}"
    )


def bit_count(x: int) -> int:
    return int(x).bit_count()


def occs_from_string(bitstr: int, norb: int) -> tuple[int, ...]:
    return tuple(i for i in range(norb) if (int(bitstr) >> i) & 1)


def block_spin_bitstring(alpha_str: int, beta_str: int, norb: int) -> str:
    """Return q0...q(2*norb-1) in block-spin order: alpha orbitals then beta orbitals."""
    alpha_bits = ''.join('1' if (int(alpha_str) >> i) & 1 else '0' for i in range(norb))
    beta_bits = ''.join('1' if (int(beta_str) >> i) & 1 else '0' for i in range(norb))
    return alpha_bits + beta_bits


def excite(bitstr: int, remove_orb: int, add_orb: int) -> int:
    return (int(bitstr) & ~(1 << remove_orb)) | (1 << add_orb)


def build_lih_hamiltonian(r_angstrom: float = 1.5957):
    mol = gto.M(
        atom=f"Li 0 0 0; H 0 0 {r_angstrom}",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    mycc = cc.CCSD(mf).run(verbose=0)

    C = mf.mo_coeff
    norb = C.shape[1]
    nelec = mol.nelec
    h1_mo = C.T @ mf.get_hcore() @ C
    eri_mo = ao2mo.restore(1, ao2mo.kernel(mol, C), norb)
    e_nuc = mol.energy_nuc()
    e_fci, ci_vec = direct_spin1.kernel(h1_mo, eri_mo, norb, nelec, ecore=e_nuc)

    # Build the full Hamiltonian matrix in the fixed (N_alpha,N_beta)=(2,2) sector.
    alpha_strings = cistring.make_strings(range(norb), nelec[0])
    beta_strings = cistring.make_strings(range(norb), nelec[1])
    dim = len(alpha_strings) * len(beta_strings)
    h2e_eff = direct_spin1.absorb_h1e(h1_mo, eri_mo, norb, nelec, 0.5)
    H = np.zeros((dim, dim), dtype=float)
    for k in range(dim):
        basis_vec = np.zeros((len(alpha_strings), len(beta_strings)), dtype=float)
        basis_vec.flat[k] = 1.0
        H[:, k] = direct_spin1.contract_2e(h2e_eff, basis_vec, norb, nelec).ravel()
    H = 0.5 * (H + H.T) + np.eye(dim) * e_nuc

    determinants = []
    for a_str, b_str in itertools.product(alpha_strings, beta_strings):
        determinants.append((int(a_str), int(b_str)))
    det_index = {det: i for i, det in enumerate(determinants)}

    return {
        "mol": mol,
        "mf": mf,
        "ccsd": mycc,
        "norb": norb,
        "nelec": nelec,
        "e_nuc": float(e_nuc),
        "e_hf": float(mf.e_tot),
        "e_ccsd": float(mycc.e_tot),
        "e_fci": float(e_fci),
        "H": H,
        "alpha_strings": alpha_strings,
        "beta_strings": beta_strings,
        "determinants": determinants,
        "det_index": det_index,
        "ci_vec": ci_vec,
    }


def ccsd_uccsd_inspired_probabilities(data, epsilon: float = 1e-12):
    """Build a determinant measurement distribution from RHF-reference CCSD amplitudes.

    The reference, single excitations, opposite-spin doubles, and same-spin double excitations
    are assigned amplitudes from the PySCF RHF-CCSD t1/t2 tensors. The probabilities are
    proportional to |amplitude|^2. A tiny epsilon floor prevents numerical zero-probability
    artifacts while remaining negligible for the leading determinants.
    """
    norb = data["norb"]
    nalpha, nbeta = data["nelec"]
    assert nalpha == nbeta == 2, "This helper is specialized to closed-shell LiH with 2 alpha and 2 beta electrons."
    nocc = nalpha
    nvir = norb - nocc
    t1 = data["ccsd"].t1
    t2 = data["ccsd"].t2
    det_index = data["det_index"]
    dim = len(data["determinants"])

    ref = sum(1 << i for i in range(nocc))
    amps = np.zeros(dim, dtype=float)
    amps[det_index[(ref, ref)]] = 1.0

    # alpha and beta singles from the RHF reference.
    for i in range(nocc):
        for av in range(nvir):
            a = nocc + av
            amps[det_index[(excite(ref, i, a), ref)]] += t1[i, av]
            amps[det_index[(ref, excite(ref, i, a))]] += t1[i, av]

    # Opposite-spin doubles alpha(i->a), beta(j->b).
    for i in range(nocc):
        for j in range(nocc):
            for av in range(nvir):
                for bv in range(nvir):
                    a = nocc + av
                    b = nocc + bv
                    amps[det_index[(excite(ref, i, a), excite(ref, j, b))]] += t2[i, j, av, bv]

    # Same-spin doubles. For a closed shell, use antisymmetrized spatial t2 as a compact proxy.
    for av in range(nvir):
        for bv in range(av + 1, nvir):
            a = nocc + av
            b = nocc + bv
            two_virtual = (1 << a) | (1 << b)
            val = 0.5 * (t2[0, 1, av, bv] - t2[0, 1, bv, av])
            amps[det_index[(two_virtual, ref)]] += val
            amps[det_index[(ref, two_virtual)]] += val

    probs = amps**2 + epsilon
    probs /= probs.sum()
    return probs, amps


def run_qsci(data, shot_budgets, seed: int, epsilon: float):
    probs, amps = ccsd_uccsd_inspired_probabilities(data, epsilon=epsilon)
    rng = np.random.default_rng(seed)
    H = data["H"]
    e_fci = data["e_fci"]
    results = []

    for shots in shot_budgets:
        sampled_indices = rng.choice(len(probs), size=shots, replace=True, p=probs)
        unique_indices = np.unique(sampled_indices)
        H_sub = H[np.ix_(unique_indices, unique_indices)]
        e_sqd = float(np.linalg.eigvalsh(H_sub)[0])
        results.append(
            {
                "shots": int(shots),
                "d": int(len(unique_indices)),
                "E_SQD_Ha": e_sqd,
                "abs_error_Ha": abs(e_sqd - e_fci),
                "kept_fraction": 1.0,  # sampling distribution is already projected to N_alpha=N_beta=2
            }
        )
    return results, probs, amps


def make_plots(results, outdir: Path):
    import matplotlib.pyplot as plt

    shots = np.array([r["shots"] for r in results], dtype=float)
    d = np.array([r["d"] for r in results], dtype=float)
    err = np.array([r["abs_error_Ha"] for r in results], dtype=float)

    fig1 = outdir / "problem5_error_vs_subspace_dimension.png"
    plt.figure(figsize=(6, 4))
    plt.loglog(d, err, marker="o")
    plt.xlabel("Subspace dimension d")
    plt.ylabel("|E_SQD - E_FCI| (Ha)")
    plt.title("LiH QSCI/SQD convergence vs subspace dimension")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    for x, y, s in zip(d, err, shots):
        plt.annotate(f"{int(s):g} shots", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.close()

    fig2 = outdir / "problem5_error_vs_shots.png"
    plt.figure(figsize=(6, 4))
    plt.loglog(shots, err, marker="o")
    plt.xlabel("Shot count")
    plt.ylabel("|E_SQD - E_FCI| (Ha)")
    plt.title("LiH QSCI/SQD convergence vs shot count")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    for x, y, dd in zip(shots, err, d):
        plt.annotate(f"d={int(dd)}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()
    return fig1, fig2


def save_results(results, data, probs, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "problem5_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shots", "d", "E_SQD_Ha", "abs_error_Ha", "kept_fraction"])
        for r in results:
            writer.writerow([
                r["shots"],
                r["d"],
                f"{r['E_SQD_Ha']:.12f}",
                f"{r['abs_error_Ha']:.12e}",
                f"{r['kept_fraction']:.6f}",
            ])

    # Top sampled determinants for auditability.
    top_path = outdir / "problem5_top_determinants.csv"
    order = np.argsort(probs)[::-1]
    with top_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "probability", "alpha_occ", "beta_occ", "block_spin_bitstring"])
        for rank, idx in enumerate(order[:20], start=1):
            a_str, b_str = data["determinants"][idx]
            writer.writerow([
                rank,
                f"{probs[idx]:.12e}",
                str(occs_from_string(a_str, data["norb"])),
                str(occs_from_string(b_str, data["norb"])),
                block_spin_bitstring(a_str, b_str, data["norb"]),
            ])
    return csv_path, top_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10010022, help="Random seed; use numerical student ID.")
    parser.add_argument("--outdir", type=str, default="problem5_outputs")
    parser.add_argument("--epsilon", type=float, default=1e-12, help="Tiny probability floor.")
    parser.add_argument("--shots", type=int, nargs="+", default=[100, 1000, 10000, 100000])
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = build_lih_hamiltonian()
    results, probs, amps = run_qsci(data, args.shots, args.seed, args.epsilon)
    csv_path, top_path = save_results(results, data, probs, outdir)
    fig1, fig2 = make_plots(results, outdir)

    print("HW3 Problem 5 - LiH QSCI-style SQD")
    print(f"Seed = {args.seed}")
    print(f"norb = {data['norb']}, nelec = {data['nelec']}, full sector dimension = {len(data['determinants'])}")
    print(f"E_nuc  = {data['e_nuc']:.12f} Ha")
    print(f"E_HF   = {data['e_hf']:.12f} Ha")
    print(f"E_CCSD = {data['e_ccd'] if False else data['e_ccsd']:.12f} Ha")
    print(f"E_FCI  = {data['e_fci']:.12f} Ha")
    print("\nshots,d,E_SQD_Ha,abs_error_Ha")
    for r in results:
        print(f"{r['shots']},{r['d']},{r['E_SQD_Ha']:.12f},{r['abs_error_Ha']:.12e}")
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {top_path}")
    print(f"Wrote: {fig1}")
    print(f"Wrote: {fig2}")


if __name__ == "__main__":
    main()
