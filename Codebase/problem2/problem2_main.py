"""
HW3 Problem 2 reproducibility script
Qubit mappings for minimal-basis H2 at R = 0.7414 Angstrom.

This script generates the numerical data reported in HW3_Problem2_Report_Final.pdf:
- Jordan-Wigner, Bravyi-Kitaev, and two-qubit parity-tapered Pauli Hamiltonians
- fixed parity sector for neutral H2, N_alpha = N_beta = 1
- eigenvalue comparison in the physical sector
- mapping resource summary

Conventions:
- Mode/qubit order before mapping:
    0 = sigma_g alpha, 1 = sigma_g beta, 2 = sigma_u alpha, 3 = sigma_u beta
- Pauli labels are printed in ascending qubit-index order.
  Example: XZYI means X_0 Z_1 Y_2 I_3.
- The printed Hamiltonian coefficients are electronic energies. Add E_nuc for total energies.

Dependencies:
    pip install numpy pyscf

Run:
    python problem2_main.py
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from pyscf import ao2mo, gto, scf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PySCF is required to run this script. Install it with:\n"
        "    pip install pyscf numpy\n"
    ) from exc


# -----------------------------------------------------------------------------
# 1. Molecular integrals: H2/STO-3G at R = 0.7414 Angstrom
# -----------------------------------------------------------------------------

def build_h2_integrals() -> Tuple[float, np.ndarray, np.ndarray]:
    """Return Enuc, one-electron MO integrals, and spatial ERIs in PySCF convention."""
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
    eri_chem = ao2mo.restore(1, ao2mo.kernel(mol, c_mo), c_mo.shape[1])
    e_nuc = mol.energy_nuc()
    return e_nuc, h_mo, eri_chem


def build_spin_orbital_integrals(h_mo: np.ndarray, eri_chem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build h[p,q] and Dirac spin-orbital <pq|rs> in Problem-1 order."""
    # Problem-1 order: 0=g alpha, 1=g beta, 2=u alpha, 3=u beta
    spatial = [0, 0, 1, 1]
    spin = [0, 1, 0, 1]  # alpha=0, beta=1
    n_modes = 4

    h = np.zeros((n_modes, n_modes))
    g = np.zeros((n_modes, n_modes, n_modes, n_modes))

    for p, q in itertools.product(range(n_modes), repeat=2):
        if spin[p] == spin[q]:
            h[p, q] = h_mo[spatial[p], spatial[q]]

    # PySCF spatial ERIs are in chemists' notation (mu nu | lambda sigma).
    # For real orbitals, eri_chem[p_spatial, r_spatial, q_spatial, s_spatial]
    # corresponds to Dirac <pq|rs> when spin selection rules are satisfied.
    for p, q, r, s in itertools.product(range(n_modes), repeat=4):
        if spin[p] == spin[r] and spin[q] == spin[s]:
            g[p, q, r, s] = eri_chem[spatial[p], spatial[r], spatial[q], spatial[s]]

    return h, g


# -----------------------------------------------------------------------------
# 2. Fermionic Hamiltonian matrix in the occupation-number basis
# -----------------------------------------------------------------------------

def annihilate(state: int, p: int) -> Tuple[int | None, int]:
    """Apply a_p to occupation basis state encoded as an integer."""
    if not ((state >> p) & 1):
        return None, 0
    phase = (-1) ** (bin(state & ((1 << p) - 1)).count("1"))
    return state & ~(1 << p), phase


def create(state: int, p: int) -> Tuple[int | None, int]:
    """Apply a_p^dagger to occupation basis state encoded as an integer."""
    if (state >> p) & 1:
        return None, 0
    phase = (-1) ** (bin(state & ((1 << p) - 1)).count("1"))
    return state | (1 << p), phase


def build_fermionic_matrix(h: np.ndarray, g: np.ndarray, n_modes: int = 4) -> np.ndarray:
    """Build H = sum h_pq a^dag_p a_q + 1/2 sum <pq|rs> a^dag_p a^dag_q a_s a_r."""
    dim = 2 ** n_modes
    h_f = np.zeros((dim, dim), dtype=complex)

    for ket in range(dim):
        for p, q in itertools.product(range(n_modes), repeat=2):
            st, ph1 = annihilate(ket, q)
            if st is None:
                continue
            st, ph2 = create(st, p)
            if st is not None:
                h_f[st, ket] += h[p, q] * ph1 * ph2

        for p, q, r, s in itertools.product(range(n_modes), repeat=4):
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
                h_f[st, ket] += 0.5 * g[p, q, r, s] * ph1 * ph2 * ph3 * ph4

    return h_f


# -----------------------------------------------------------------------------
# 3. Binary basis transforms for JW, BK, and parity mapping
# -----------------------------------------------------------------------------

# Jordan-Wigner: qubit bits are occupation bits.
A_JW = np.eye(4, dtype=int)

# Four-mode Bravyi-Kitaev binary transform used in the report.
A_BK = np.array(
    [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 1],
    ],
    dtype=int,
)

# Parity mapping chosen to expose spin parities:
# q2 = n0+n2 = alpha parity, q3 = n1+n3 = beta parity.
A_PARITY = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ],
    dtype=int,
)


def bit_list(state: int, n_bits: int) -> np.ndarray:
    return np.array([(state >> j) & 1 for j in range(n_bits)], dtype=int)


def bits_to_int(bits: Iterable[int]) -> int:
    return sum(int(bit) << j for j, bit in enumerate(bits))


def occupation_to_mapped_state(n_state: int, transform: np.ndarray) -> int:
    n_bits = bit_list(n_state, transform.shape[1])
    q_bits = (transform @ n_bits) % 2
    return bits_to_int(q_bits)


def transformed_matrix(h_occ: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply q = transform @ n mod 2, where n is the occupation bit vector."""
    dim = h_occ.shape[0]
    q_to_n: Dict[int, int] = {}

    for n_state in range(dim):
        q_state = occupation_to_mapped_state(n_state, transform)
        q_to_n[q_state] = n_state

    h_qubit = np.zeros_like(h_occ)
    for qi in range(dim):
        for qj in range(dim):
            h_qubit[qi, qj] = h_occ[q_to_n[qi], q_to_n[qj]]
    return h_qubit


def parity_taper_indices() -> List[int]:
    """Keep q2=q3=1 for neutral H2: N_alpha=N_beta=1."""
    return [q0 + (q1 << 1) + (1 << 2) + (1 << 3) for q0 in [0, 1] for q1 in [0, 1]]


def taper_parity_hamiltonian(h_parity_4q: np.ndarray) -> np.ndarray:
    kept_indices = parity_taper_indices()
    return h_parity_4q[np.ix_(kept_indices, kept_indices)]


# -----------------------------------------------------------------------------
# 4. Pauli expansion and sector comparison
# -----------------------------------------------------------------------------

I2 = np.array([[1, 0], [0, 1]], complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron_label(label: str) -> np.ndarray:
    """Build matrix for a Pauli label written in ascending qubit-index order."""
    mat = np.array([[1]], complex)
    for j in reversed(range(len(label))):
        mat = np.kron(mat, PAULI[label[j]])
    return mat


def pauli_expansion(h_qubit: np.ndarray, tol: float = 1.0e-10) -> List[Tuple[str, float]]:
    n_qubits = int(np.log2(h_qubit.shape[0]))
    terms: List[Tuple[str, float]] = []
    for chars in itertools.product("IXYZ", repeat=n_qubits):
        label = "".join(chars)
        coeff = np.trace(kron_label(label).conj().T @ h_qubit) / (2 ** n_qubits)
        if abs(coeff) > tol:
            terms.append((label, float(coeff.real)))
    return terms


def h2_neutral_occupation_sector() -> List[int]:
    """Occupation basis states with N_alpha=n0+n2=1 and N_beta=n1+n3=1."""
    sector = []
    for state in range(16):
        occ = bit_list(state, 4)
        if occ[0] + occ[2] == 1 and occ[1] + occ[3] == 1:
            sector.append(state)
    return sector


def mapped_sector_indices(transform: np.ndarray) -> List[int]:
    """Physical sector indices after applying a binary qubit mapping."""
    return [occupation_to_mapped_state(state, transform) for state in h2_neutral_occupation_sector()]


def pauli_weight(label: str) -> int:
    return sum(ch != "I" for ch in label)


def pretty_label(label: str) -> str:
    """Convert a compact label such as IZYY to Z_1 Y_2 Y_3 for readability."""
    if set(label) == {"I"}:
        return "I"
    return " ".join(f"{ch}_{j}" for j, ch in enumerate(label) if ch != "I")


def print_terms(name: str, terms: Iterable[Tuple[str, float]]) -> None:
    print(f"\n{name} Pauli terms, electronic Hamiltonian:")
    for label, coeff in terms:
        print(f"  {coeff:+.12f}  {label:4s}    # {pretty_label(label)}")


def main() -> None:
    e_nuc, h_mo, eri_chem = build_h2_integrals()
    h, g = build_spin_orbital_integrals(h_mo, eri_chem)
    h_f = build_fermionic_matrix(h, g)

    h_jw = transformed_matrix(h_f, A_JW)
    h_bk = transformed_matrix(h_f, A_BK)
    h_parity_4q = transformed_matrix(h_f, A_PARITY)
    h_parity_2q = taper_parity_hamiltonian(h_parity_4q)

    jw_terms = pauli_expansion(h_jw)
    bk_terms = pauli_expansion(h_bk)
    parity_terms = pauli_expansion(h_parity_2q)

    print(f"PySCF nuclear repulsion E_nuc = {e_nuc:.12f} Ha")
    print_terms("JW", jw_terms)
    print_terms("BK", bk_terms)
    print_terms("Two-qubit parity-tapered", parity_terms)

    occ_sector = h2_neutral_occupation_sector()
    jw_sector = mapped_sector_indices(A_JW)
    bk_sector = mapped_sector_indices(A_BK)
    parity_sector = mapped_sector_indices(A_PARITY)

    eig_f = np.linalg.eigvalsh(h_f[np.ix_(occ_sector, occ_sector)])
    eig_jw = np.linalg.eigvalsh(h_jw[np.ix_(jw_sector, jw_sector)])
    eig_bk = np.linalg.eigvalsh(h_bk[np.ix_(bk_sector, bk_sector)])
    eig_parity4 = np.linalg.eigvalsh(h_parity_4q[np.ix_(parity_sector, parity_sector)])
    eig_parity2 = np.linalg.eigvalsh(h_parity_2q)

    print("\nFour electronic eigenvalues in the N_alpha=1, N_beta=1 sector:")
    for name, eig in [
        ("Fermionic", eig_f),
        ("JW", eig_jw),
        ("BK", eig_bk),
        ("Parity4", eig_parity4),
        ("Parity2", eig_parity2),
    ]:
        print(f"  {name:10s}: " + "  ".join(f"{x:+.12f}" for x in eig))

    print("\nGround-state molecular total energy:")
    print(f"  E_FCI,total = {eig_parity2[0] + e_nuc:+.12f} Ha")

    print("\nMapping summary:")
    for name, n_qubits, terms in [("JW", 4, jw_terms), ("BK", 4, bk_terms), ("Parity2", 2, parity_terms)]:
        non_identity = [(label, c) for label, c in terms if label != "I" * n_qubits]
        max_weight = max(pauli_weight(label) for label, _ in non_identity)
        print(
            f"  {name:8s}: qubits={n_qubits}, "
            f"non-identity terms={len(non_identity)}, max Pauli weight={max_weight}"
        )


if __name__ == "__main__":
    main()
