"""
HW3 Problem 3 reproducibility script
Minimal-basis H2 integrals, qubit Hamiltonians, and ansatz specifications.

Conventions:
  - Geometry: H2, R = 0.7414 Angstrom, STO-3G basis.
  - Spin-orbital order: 0 = sigma_g alpha, 1 = sigma_g beta,
    2 = sigma_u alpha, 3 = sigma_u beta.
  - ERI convention in the report: Dirac/spatial <pq|rs>.
    PySCF stores chemists' notation (pq|rs); the script converts it by
    <pq|rs> = (p r | q s)_chemist for real orbitals.
  - Printed Pauli labels are in ascending qubit-index order.
    Example: XYYX means X_0 Y_1 Y_2 X_3.

The script tries to use PySCF. If PySCF is unavailable, it uses the embedded
reference integrals reported in the PDF so the remaining verification can run.

Run:
    python problem3_main.py
Optional:
    python problem3_main.py --require-pyscf
"""

from __future__ import annotations

import argparse
import itertools
from typing import Dict, Iterable, List, Tuple

import numpy as np

STUDENT_ID_NUMERIC_SEED = 10010022
np.random.seed(STUDENT_ID_NUMERIC_SEED)

# -----------------------------------------------------------------------------
# Reference numerical values reported in HW3_Problem3_Report_Final.pdf
# -----------------------------------------------------------------------------

E_NUC_REF = 0.713753993688
H_MO_REF = np.array(
    [
        [-1.253309786646, 0.0],
        [0.0, -0.475068848772],
    ],
    dtype=float,
)
# Spatial Dirac ERIs <pq|rs>, p,q,r,s in {0,1} for sigma_g, sigma_u.
ERI_DIRAC_REF = np.zeros((2, 2, 2, 2), dtype=float)
unique_dirac = {
    (0, 0, 0, 0): 0.676181192522,  # <11|11>
    (0, 0, 0, 1): 0.0,             # <11|12>
    (0, 0, 1, 1): 0.181288808211,  # <11|22>
    (0, 1, 0, 1): 0.663434443056,  # <12|12>
    (0, 1, 1, 1): 0.0,             # <12|22>
    (1, 1, 1, 1): 0.695634034524,  # <22|22>
}
# Fill the real-orbital Dirac permutation symmetries.
for (p, q, r, s), value in unique_dirac.items():
    for idx in [
        (p, q, r, s), (q, p, s, r), (r, s, p, q), (s, r, q, p),
        (r, q, p, s), (p, s, r, q), (q, r, s, p), (s, p, q, r),
    ]:
        ERI_DIRAC_REF[idx] = value


def get_integrals(require_pyscf: bool = False) -> Tuple[str, float, np.ndarray, np.ndarray]:
    """Return source label, E_nuc, h_mo, and spatial Dirac ERIs <pq|rs>."""
    try:
        from pyscf import ao2mo, gto, scf  # type: ignore
        import pyscf  # type: ignore
    except ModuleNotFoundError:
        if require_pyscf:
            raise SystemExit("PySCF is required but not installed. Run: pip install pyscf")
        return "embedded reference values (PySCF unavailable)", E_NUC_REF, H_MO_REF.copy(), ERI_DIRAC_REF.copy()

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.7414",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    c_mo = mf.mo_coeff
    h_mo = c_mo.T @ mf.get_hcore() @ c_mo
    eri_chemist = ao2mo.restore(1, ao2mo.kernel(mol, c_mo), c_mo.shape[1])

    # Convert PySCF chemists' notation to Dirac <pq|rs> = (p r | q s)_chemist.
    n_spatial = h_mo.shape[0]
    eri_dirac = np.zeros_like(eri_chemist)
    for p, q, r, s in itertools.product(range(n_spatial), repeat=4):
        eri_dirac[p, q, r, s] = eri_chemist[p, r, q, s]

    return f"PySCF {pyscf.__version__}", float(mol.energy_nuc()), h_mo, eri_dirac


def spin_orbital_integrals(h_mo: np.ndarray, eri_dirac: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build spin-orbital h[p,q] and <pq|rs> in Problem-1 order."""
    # 0=g alpha, 1=g beta, 2=u alpha, 3=u beta
    spatial = [0, 0, 1, 1]
    spin = [0, 1, 0, 1]
    n = 4
    h = np.zeros((n, n), dtype=float)
    g = np.zeros((n, n, n, n), dtype=float)
    for p, q in itertools.product(range(n), repeat=2):
        if spin[p] == spin[q]:
            h[p, q] = h_mo[spatial[p], spatial[q]]
    for p, q, r, s in itertools.product(range(n), repeat=4):
        if spin[p] == spin[r] and spin[q] == spin[s]:
            g[p, q, r, s] = eri_dirac[spatial[p], spatial[q], spatial[r], spatial[s]]
    return h, g


def annihilate(state: int, p: int) -> Tuple[int | None, int]:
    if not ((state >> p) & 1):
        return None, 0
    phase = (-1) ** (bin(state & ((1 << p) - 1)).count("1"))
    return state & ~(1 << p), phase


def create(state: int, p: int) -> Tuple[int | None, int]:
    if (state >> p) & 1:
        return None, 0
    phase = (-1) ** (bin(state & ((1 << p) - 1)).count("1"))
    return state | (1 << p), phase


def fermionic_matrix(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    n = h.shape[0]
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for ket in range(dim):
        for p, q in itertools.product(range(n), repeat=2):
            st, ph1 = annihilate(ket, q)
            if st is None:
                continue
            st, ph2 = create(st, p)
            if st is not None:
                H[st, ket] += h[p, q] * ph1 * ph2
        for p, q, r, s in itertools.product(range(n), repeat=4):
            st, ph1 = annihilate(ket, r)
            if st is None:
                continue
            st, ph2 = annihilate(st, s)
            if st is None:
                continue
            st, ph3 = create(st, q)
            if st is None:
                continue
            st, ph4 = create(st, p)
            if st is not None:
                H[st, ket] += 0.5 * g[p, q, r, s] * ph1 * ph2 * ph3 * ph4
    return H


A_JW = np.eye(4, dtype=int)
A_BK = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]], dtype=int)
A_PAR = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=int)


def bits(state: int, n: int) -> np.ndarray:
    return np.array([(state >> j) & 1 for j in range(n)], dtype=int)


def bitint(bs: Iterable[int]) -> int:
    return sum(int(b) << j for j, b in enumerate(bs))


def mapped_state(occ_state: int, A: np.ndarray) -> int:
    return bitint((A @ bits(occ_state, A.shape[1])) % 2)


def transform_matrix(H_occ: np.ndarray, A: np.ndarray) -> np.ndarray:
    dim = H_occ.shape[0]
    q_to_n: Dict[int, int] = {}
    for n_state in range(dim):
        q_to_n[mapped_state(n_state, A)] = n_state
    Hq = np.zeros_like(H_occ)
    for qi in range(dim):
        for qj in range(dim):
            Hq[qi, qj] = H_occ[q_to_n[qi], q_to_n[qj]]
    return Hq


def neutral_sector_indices() -> List[int]:
    sector = []
    for s in range(16):
        b = bits(s, 4)
        if b[0] + b[2] == 1 and b[1] + b[3] == 1:
            sector.append(s)
    return sector


def mapped_sector(A: np.ndarray) -> List[int]:
    return [mapped_state(s, A) for s in neutral_sector_indices()]


def taper_parity(H4: np.ndarray) -> np.ndarray:
    # q2 = q3 = 1; keep q0, q1.
    idx = [q0 + (q1 << 1) + (1 << 2) + (1 << 3) for q0 in [0, 1] for q1 in [0, 1]]
    return H4[np.ix_(idx, idx)]


I2 = np.array([[1, 0], [0, 1]], complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron_label(label: str) -> np.ndarray:
    mat = np.array([[1]], dtype=complex)
    for j in reversed(range(len(label))):
        mat = np.kron(mat, PAULI[label[j]])
    return mat


def pauli_expansion(H: np.ndarray, tol: float = 1e-10) -> List[Tuple[str, float]]:
    n = int(np.log2(H.shape[0]))
    out: List[Tuple[str, float]] = []
    for chars in itertools.product("IXYZ", repeat=n):
        label = "".join(chars)
        coeff = np.trace(kron_label(label).conj().T @ H) / (2 ** n)
        if abs(coeff) > tol:
            out.append((label, float(coeff.real)))
    return out


def pretty(label: str) -> str:
    if label == "I" * len(label):
        return "I"
    return " ".join(f"{ch}_{i}" for i, ch in enumerate(label) if ch != "I")


def print_terms(title: str, terms: List[Tuple[str, float]]) -> None:
    print(f"\n{title}")
    for label, c in terms:
        print(f"  {c:+.12f}  {label:<4s}   # {pretty(label)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-pyscf", action="store_true")
    args = parser.parse_args()

    source, e_nuc, h_mo, eri_dirac = get_integrals(args.require_pyscf)
    h, g = spin_orbital_integrals(h_mo, eri_dirac)
    Hf = fermionic_matrix(h, g)

    Hjw = transform_matrix(Hf, A_JW)
    Hbk = transform_matrix(Hf, A_BK)
    Hpar4 = transform_matrix(Hf, A_PAR)
    Hpar2 = taper_parity(Hpar4)

    print("HW3 Problem 3 verification")
    print(f"Seed = {STUDENT_ID_NUMERIC_SEED}")
    print(f"Integral source = {source}")
    print(f"E_nuc = {e_nuc:.12f} Ha")
    print("\nOne-electron integrals h_pq in canonical RHF spatial MO basis:")
    print(h_mo)
    print("\nSix symmetry-unique spatial Dirac ERIs <pq|rs>:")
    for key in [(0,0,0,0),(0,0,0,1),(0,0,1,1),(0,1,0,1),(0,1,1,1),(1,1,1,1)]:
        print(f"  <{key[0]+1}{key[1]+1}|{key[2]+1}{key[3]+1}> = {eri_dirac[key]:+.12f}")

    print_terms("JW electronic Hamiltonian", pauli_expansion(Hjw))
    print_terms("BK electronic Hamiltonian", pauli_expansion(Hbk))
    print_terms("Two-qubit parity-tapered electronic Hamiltonian", pauli_expansion(Hpar2))

    print("\nExact diagonalization in N_alpha=1, N_beta=1 sector:")
    for name, H, idx in [
        ("Fermionic", Hf, neutral_sector_indices()),
        ("JW", Hjw, mapped_sector(A_JW)),
        ("BK", Hbk, mapped_sector(A_BK)),
        ("Parity4", Hpar4, mapped_sector(A_PAR)),
    ]:
        eig = np.linalg.eigvalsh(H[np.ix_(idx, idx)])
        print(f"  {name:<9s} E_elec_min = {eig[0]:+.12f}, E_total_min = {eig[0] + e_nuc:+.12f}")
    eig2 = np.linalg.eigvalsh(Hpar2)
    print(f"  {'Parity2':<9s} E_elec_min = {eig2[0]:+.12f}, E_total_min = {eig2[0] + e_nuc:+.12f}")

    print("\nAnsatz specifications used in the report:")
    print("  UCCSD/JW: one essential double-excitation parameter theta for |sigma_g^2> <-> |sigma_u^2>.")
    print("  UCCSD/parity: the same physics becomes a one-parameter two-qubit entangling rotation.")
    print("  HEA/JW: initial |1100>, RY rotations, linear CNOT chain, 2 layers, final RY layer, 12 parameters.")
    print("  HEA/parity: initial |11>, RY rotations, CNOT(0,1), 2 layers, final RY layer, 6 parameters.")


if __name__ == "__main__":
    main()
