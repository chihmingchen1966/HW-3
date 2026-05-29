#!/usr/bin/env python3
"""
HW3 Problem 1 - Minimal-basis H2 helper script

Purpose
-------
Problem 1 is mainly a derivation problem. This script provides a minimal
reproducible code component for the report by constructing the requested
two-determinant closed-shell FCI Hamiltonian block and checking Hermiticity.

Seed policy
-----------
HW3 requires the numerical part of the student ID as the random seed for
randomized steps in later problems. Problem 1 has no randomized step, but the
seed is still recorded here for consistency.

Seed used: 10010022
"""

import numpy as np

SEED = 10010022
np.random.seed(SEED)


def build_closed_shell_fci_block(h, eri):
    """
    Build the two-determinant closed-shell FCI block for

        |Psi0> = |chi1 chi2>
        |PsiD> = |chi3 chi4>

    using the spin-orbital convention

        <pq|rs> = int dx1 dx2 chi_p*(x1) chi_q*(x2) r12^-1 chi_r(x1) chi_s(x2)

    Parameters
    ----------
    h : ndarray, shape (4, 4)
        One-electron spin-orbital integral matrix h[p, q].
        Python indices 0,1,2,3 correspond to chi1, chi2, chi3, chi4.

    eri : ndarray, shape (4, 4, 4, 4)
        Two-electron spin-orbital integral tensor eri[p, q, r, s] = <pq|rs>.

    Returns
    -------
    H : ndarray, shape (2, 2)
        Closed-shell FCI block:

        [[h11+h22+<12|12>-<12|21>,  <12|34>-<12|43>],
         [<34|12>-<34|21>,          h33+h44+<34|34>-<34|43>]]
    """

    # Convert report labels chi1, chi2, chi3, chi4 to zero-based Python indices.
    i, j = 0, 1   # chi1, chi2
    a, b = 2, 3   # chi3, chi4

    h11 = h[i, i] + h[j, j] + eri[i, j, i, j] - eri[i, j, j, i]
    h22 = h[a, a] + h[b, b] + eri[a, b, a, b] - eri[a, b, b, a]

    h12 = eri[i, j, a, b] - eri[i, j, b, a]
    h21 = eri[a, b, i, j] - eri[a, b, j, i]

    return np.array([[h11, h12],
                     [h21, h22]], dtype=complex)


def demo_with_hermitian_test_integrals():
    """
    Use simple artificial integrals that satisfy the Hermitian relation

        <pq|rs>* = <rs|pq>

    This is not meant to reproduce a numerical H2 calculation. It only verifies
    the algebraic structure required in Problem 1.
    """

    h = np.zeros((4, 4), dtype=complex)
    h[0, 0] = -1.0
    h[1, 1] = -1.0
    h[2, 2] = -0.2
    h[3, 3] = -0.2

    eri = np.zeros((4, 4, 4, 4), dtype=complex)

    # Diagonal Coulomb/exchange-like terms.
    eri[0, 1, 0, 1] = 0.70
    eri[0, 1, 1, 0] = 0.10
    eri[2, 3, 2, 3] = 0.60
    eri[2, 3, 3, 2] = 0.05

    # Off-diagonal coupling terms.
    eri[0, 1, 2, 3] = 0.18
    eri[0, 1, 3, 2] = 0.02

    # Enforce Hermitian conjugate partners: <pq|rs>* = <rs|pq>.
    eri[2, 3, 0, 1] = np.conjugate(eri[0, 1, 2, 3])
    eri[3, 2, 0, 1] = np.conjugate(eri[0, 1, 3, 2])

    # Electron-label interchange symmetry used in the report:
    # <43|12> = <34|21>.
    eri[2, 3, 1, 0] = eri[3, 2, 0, 1]

    H = build_closed_shell_fci_block(h, eri)

    print("HW3 Problem 1 - Minimal-basis H2")
    print(f"Seed used: {SEED}")
    print()
    print("Two-determinant closed-shell FCI block:")
    print(H)
    print()
    print("Hermiticity check: H == H.conj().T")
    print(np.allclose(H, H.conjugate().T))
    print()
    print("Eigenvalues of this demonstration block:")
    print(np.linalg.eigvalsh(H))


if __name__ == "__main__":
    demo_with_hermitian_test_integrals()
