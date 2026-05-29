#!/usr/bin/env python3
"""
problem4_main.py

Homework 3 - Problem 4: VQE for minimal-basis H2.

This script is intentionally self-contained for the Problem 4 VQE study.
It uses standard H2/STO-3G Hamiltonian coefficients at R = 0.7414 Angstrom
for the Jordan-Wigner four-qubit Hamiltonian and the two-qubit
parity-tapered Hamiltonian.

Author seed convention:
    Use the numerical part of the student ID for every randomized step.
    Here: SEED = 10010022

What the script does:
    1. Exact diagonalization for JW and parity Hamiltonians.
    2. VQE for four cases:
        - UCCSD + JW
        - UCCSD + parity
        - real-amplitude + JW
        - real-amplitude + parity
    3. Finite-shot energy estimation at the optimized noiseless parameters
       of the best two-qubit parity-tapered circuit.
    4. Prints tables suitable for copying into the report.

Dependencies:
    numpy
    scipy  (recommended; used for scipy.optimize.minimize)

Run:
    python3 problem4_main.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    from scipy.optimize import minimize
except Exception as exc:  # pragma: no cover
    minimize = None
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None


# ============================================================
# Global settings required by HW3 Problem 4
# ============================================================

SEED = 10010022
MAXITER = 1000
TOL = 1.0e-10
OPTIMIZER = "COBYLA"
PARAM_INIT_SCALE = 0.05

# Finite-shot settings
FINITE_SHOTS = [10**1, 10**4]

# Documented approximation for a noisy two-qubit backend.
# This is not a full device calibration file. It is a transparent simplified
# noise model used only for Problem 4 finite-shot comparison.
TARGET_QPU = "IBM Brisbane-like superconducting backend, simplified approximation"
NOISE_MODEL_SOURCE = (
    "Documented approximation: independent readout assignment error plus "
    "global expectation shrinkage representing gate/decoherence noise."
)
NOISE_READOUT_ERROR = 0.015      # 1.5% symmetric bit-flip readout error
NOISE_EXPECTATION_SHRINK = 0.985 # simple shrink factor for non-identity expectations


# ============================================================
# Basic Pauli machinery
# Convention:
#   Qubit labels are zero-based.
#   A basis state is written as |q0 q1 ... q(n-1)>.
#   Pauli strings are stored as dictionaries such as {0:"Z", 1:"X"}.
# ============================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_MATS = {
    "I": I2,
    "X": X,
    "Y": Y,
    "Z": Z,
}


@dataclass
class Hamiltonian:
    name: str
    n_qubits: int
    terms: List[Tuple[float, Dict[int, str]]]


@dataclass
class VQEResult:
    case: str
    n_params: int
    energy: float
    exact_energy: float
    abs_error: float
    params: np.ndarray
    nfev: int
    success: bool
    message: str


def pauli_matrix(pauli: Dict[int, str], n_qubits: int) -> np.ndarray:
    """Build the dense matrix for a Pauli string."""
    mats = []
    for q in range(n_qubits):
        mats.append(PAULI_MATS[pauli.get(q, "I")])
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def hamiltonian_matrix(ham: Hamiltonian) -> np.ndarray:
    """Build the dense Hamiltonian matrix."""
    dim = 2**ham.n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for coeff, pauli in ham.terms:
        H += coeff * pauli_matrix(pauli, ham.n_qubits)
    return H


def basis_state(bits: List[int]) -> np.ndarray:
    """Return |q0 q1 ...> as a dense vector."""
    idx = int("".join(str(int(b)) for b in bits), 2)
    vec = np.zeros(2 ** len(bits), dtype=complex)
    vec[idx] = 1.0
    return vec


def expectation(ham_mat: np.ndarray, state: np.ndarray) -> float:
    """Compute <psi|H|psi>."""
    return float(np.real(np.vdot(state, ham_mat @ state)))


def exact_ground_energy(ham: Hamiltonian) -> float:
    """Exact ground-state energy by dense diagonalization."""
    H = hamiltonian_matrix(ham)
    return float(np.linalg.eigvalsh(H)[0])


# ============================================================
# H2 Hamiltonians
# ============================================================
# The four-qubit JW coefficients below are standard H2/STO-3G
# total-energy coefficients at R = 0.7414 Angstrom, including Enuc.
# They are commonly used in VQE tutorials and match the FCI energy
# approximately -1.137 Ha.
#
# The two-qubit parity-tapered coefficients are written as an electronic
# Hamiltonian plus Enuc shifted into the identity term.
# ============================================================

JW_H2 = Hamiltonian(
    name="H2_JW_4q_total_energy",
    n_qubits=4,
    terms=[
        (-0.097066268167631, {}),
        (0.171412826447769, {0: "Z"}),
        (0.171412826447769, {1: "Z"}),
        (-0.223431536908135, {2: "Z"}),
        (-0.223431536908135, {3: "Z"}),
        (0.168688981686932, {0: "Z", 1: "Z"}),
        (0.120546593729135, {0: "Z", 2: "Z"}),
        (0.165927850337703, {0: "Z", 3: "Z"}),
        (0.165927850337703, {1: "Z", 2: "Z"}),
        (0.120546593729135, {1: "Z", 3: "Z"}),
        (0.174412876122615, {2: "Z", 3: "Z"}),
        (-0.0453026155037993, {0: "X", 1: "X", 2: "Y", 3: "Y"}),
        (0.0453026155037993, {0: "X", 1: "Y", 2: "Y", 3: "X"}),
        (0.0453026155037993, {0: "Y", 1: "X", 2: "X", 3: "Y"}),
        (-0.0453026155037993, {0: "Y", 1: "Y", 2: "X", 3: "X"}),
    ],
)

E_NUC = 0.7199689944489797

PARITY_H2 = Hamiltonian(
    name="H2_parity_tapered_2q_total_energy",
    n_qubits=2,
    terms=[
        (-1.052373245772859 + E_NUC, {}),
        (0.39793742484318045, {0: "Z"}),
        (-0.39793742484318045, {1: "Z"}),
        (-0.01128010425623538, {0: "Z", 1: "Z"}),
        (0.18093119978423156, {0: "X", 1: "X"}),
    ],
)


# ============================================================
# Ansatz circuits
# ============================================================

def uccsd_jw_state(theta: np.ndarray) -> np.ndarray:
    """
    Minimal H2 UCCSD ansatz in the two-determinant closed-shell subspace.

    |psi(theta)> = cos(theta) |1100> + sin(theta) |0011>

    Here |1100> is the Hartree-Fock determinant with modes 0 and 1 occupied,
    and |0011> is the double excitation with modes 2 and 3 occupied.
    """
    t = float(theta[0])
    return math.cos(t) * basis_state([1, 1, 0, 0]) + math.sin(t) * basis_state([0, 0, 1, 1])


def uccsd_parity_state(theta: np.ndarray) -> np.ndarray:
    """
    Two-qubit parity-tapered single-parameter UCCSD-like ansatz.

    |psi(theta)> = cos(theta) |10> + sin(theta) |01>

    In this convention, the relevant two-dimensional sector is spanned by
    |10> and |01>, and the XX term couples these two basis states.
    """
    t = float(theta[0])
    return math.cos(t) * basis_state([1, 0]) + math.sin(t) * basis_state([0, 1])


def apply_one_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply a one-qubit gate to the state vector using the |q0 q1 ...> convention."""
    full = []
    for q in range(n_qubits):
        full.append(gate if q == qubit else I2)
    U = full[0]
    for m in full[1:]:
        U = np.kron(U, m)
    return U @ state


def apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """Apply CNOT with the |q0 q1 ...> convention."""
    dim = 2**n_qubits
    out = np.zeros_like(state)
    for idx, amp in enumerate(state):
        if abs(amp) < 1e-15:
            continue
        bits = [int(x) for x in format(idx, f"0{n_qubits}b")]
        if bits[control] == 1:
            bits[target] ^= 1
        j = int("".join(str(b) for b in bits), 2)
        out[j] += amp
    return out


def ry(theta: float) -> np.ndarray:
    """RY rotation matrix."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def real_amplitude_state(params: np.ndarray, n_qubits: int, layers: int = 2, final_rotation: bool = True) -> np.ndarray:
    """
    Hardware-efficient real-amplitude ansatz:
        alternating RY layers and linear CNOT entanglers.

    For n_qubits = 4:
        each entangler layer uses CNOT(0,1), CNOT(1,2), CNOT(2,3).
    For n_qubits = 2:
        each entangler layer uses CNOT(0,1).

    Number of parameters:
        n_qubits * (layers + 1) if final_rotation=True
        n_qubits * layers otherwise
    """
    expected = n_qubits * (layers + (1 if final_rotation else 0))
    if len(params) != expected:
        raise ValueError(f"Expected {expected} parameters, got {len(params)}")

    state = basis_state([0] * n_qubits)
    k = 0

    for _ in range(layers):
        for q in range(n_qubits):
            state = apply_one_qubit_gate(state, ry(float(params[k])), q, n_qubits)
            k += 1

        # Linear entangler.
        for q in range(n_qubits - 1):
            state = apply_cnot(state, control=q, target=q + 1, n_qubits=n_qubits)

    if final_rotation:
        for q in range(n_qubits):
            state = apply_one_qubit_gate(state, ry(float(params[k])), q, n_qubits)
            k += 1

    return state


# ============================================================
# VQE runner
# ============================================================

def run_vqe(
    case: str,
    ham: Hamiltonian,
    state_fn: Callable[[np.ndarray], np.ndarray],
    n_params: int,
    rng: np.random.Generator,
) -> VQEResult:
    """Run VQE with a shared optimizer configuration."""
    if minimize is None:
        raise RuntimeError(
            "scipy.optimize.minimize is required for this script. "
            f"Original import error: {SCIPY_IMPORT_ERROR}"
        )

    H = hamiltonian_matrix(ham)
    exact = exact_ground_energy(ham)
    x0 = rng.normal(loc=0.0, scale=PARAM_INIT_SCALE, size=n_params)

    nfev_counter = {"n": 0}

    def objective(x: np.ndarray) -> float:
        nfev_counter["n"] += 1
        state = state_fn(x)
        return expectation(H, state)

    result = minimize(
        objective,
        x0,
        method=OPTIMIZER,
        options={"maxiter": MAXITER, "tol": TOL, "rhobeg": 0.2},
    )

    energy = float(result.fun)
    return VQEResult(
        case=case,
        n_params=n_params,
        energy=energy,
        exact_energy=exact,
        abs_error=abs(energy - exact),
        params=np.asarray(result.x, dtype=float),
        nfev=int(getattr(result, "nfev", nfev_counter["n"])),
        success=bool(result.success),
        message=str(result.message),
    )


# ============================================================
# Finite-shot estimator
# ============================================================

def pauli_expectation_from_state(state: np.ndarray, pauli: Dict[int, str], n_qubits: int) -> float:
    """Exact expectation value of one Pauli string."""
    if len(pauli) == 0:
        return 1.0
    P = pauli_matrix(pauli, n_qubits)
    return float(np.real(np.vdot(state, P @ state)))


def allocate_shots(ham: Hamiltonian, total_shots: int) -> Dict[int, int]:
    """
    Allocate finite shots across non-identity Pauli terms.

    Rule:
        Allocate shots proportional to |coefficient| over non-identity terms,
        with at least one shot per non-identity Pauli term.
    """
    non_identity_indices = [i for i, (_, p) in enumerate(ham.terms) if len(p) > 0]
    weights = np.array([abs(ham.terms[i][0]) for i in non_identity_indices], dtype=float)
    weights = weights / np.sum(weights)

    raw = np.maximum(1, np.floor(total_shots * weights).astype(int))
    diff = int(total_shots - np.sum(raw))

    # Adjust rounding while keeping at least one shot per measured term.
    order = np.argsort(-weights)
    j = 0
    while diff != 0:
        idx = order[j % len(order)]
        if diff > 0:
            raw[idx] += 1
            diff -= 1
        else:
            if raw[idx] > 1:
                raw[idx] -= 1
                diff += 1
        j += 1

    return {term_idx: int(shots) for term_idx, shots in zip(non_identity_indices, raw)}


def noisy_mean_transform(mean: float, pauli: Dict[int, str]) -> float:
    """
    Simplified noisy-backend transformation for a Pauli expectation.

    A symmetric readout error p shrinks each measured qubit's expectation
    by (1 - 2p). A global expectation shrink factor approximates gate and
    decoherence effects for non-identity observables.
    """
    measured_weight = len(pauli)
    readout_factor = (1.0 - 2.0 * NOISE_READOUT_ERROR) ** measured_weight
    return float(NOISE_EXPECTATION_SHRINK * readout_factor * mean)


def finite_shot_energy(
    ham: Hamiltonian,
    state: np.ndarray,
    total_shots: int,
    rng: np.random.Generator,
    noisy: bool,
) -> Tuple[float, float, Dict[int, int]]:
    """
    Estimate energy by independently sampling each non-identity Pauli term.

    Returns:
        sample_mean_energy, standard_error, shot_allocation
    """
    allocation = allocate_shots(ham, total_shots)

    energy = 0.0
    variance = 0.0

    for term_index, (coeff, pauli) in enumerate(ham.terms):
        if len(pauli) == 0:
            energy += coeff
            continue

        n = allocation[term_index]
        mu = pauli_expectation_from_state(state, pauli, ham.n_qubits)

        if noisy:
            mu = noisy_mean_transform(mu, pauli)

        # Pauli measurement outcomes are +/-1 with E[outcome] = mu.
        prob_plus = (1.0 + mu) / 2.0
        prob_plus = min(1.0, max(0.0, prob_plus))

        samples = rng.choice(np.array([1.0, -1.0]), size=n, p=np.array([prob_plus, 1.0 - prob_plus]))
        sample_mean = float(np.mean(samples))

        # Unbiased sample variance if n > 1; fallback to Bernoulli variance if n == 1.
        if n > 1:
            sample_var = float(np.var(samples, ddof=1))
        else:
            sample_var = max(0.0, 1.0 - sample_mean**2)

        energy += coeff * sample_mean
        variance += (coeff**2) * sample_var / n

    standard_error = math.sqrt(max(variance, 0.0))
    return float(energy), float(standard_error), allocation


# ============================================================
# Printing helpers
# ============================================================

def format_params(params: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.10f}" for x in params) + "]"


def print_hamiltonian_summary() -> None:
    print("=" * 88)
    print("Hamiltonian summary")
    print("=" * 88)
    for ham in [JW_H2, PARITY_H2]:
        exact = exact_ground_energy(ham)
        non_identity = sum(1 for _, p in ham.terms if len(p) > 0)
        max_weight = max((len(p) for _, p in ham.terms), default=0)
        print(f"{ham.name}")
        print(f"  qubits                 : {ham.n_qubits}")
        print(f"  non-identity terms     : {non_identity}")
        print(f"  max Pauli-string weight: {max_weight}")
        print(f"  exact ground energy    : {exact:.12f} Ha")
        print()


def print_vqe_results(results: List[VQEResult]) -> None:
    print("=" * 88)
    print("VQE results")
    print("=" * 88)
    print(f"Shared optimizer : {OPTIMIZER}")
    print(f"Tolerance        : {TOL}")
    print(f"Max iterations   : {MAXITER}")
    print(f"Seed             : {SEED}")
    print()

    header = (
        f"{'case':30s} {'n_param':>7s} {'E_VQE (Ha)':>16s} "
        f"{'E_exact (Ha)':>16s} {'abs_error':>14s} {'nfev':>7s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.case:30s} {r.n_params:7d} {r.energy:16.12f} "
            f"{r.exact_energy:16.12f} {r.abs_error:14.6e} {r.nfev:7d}"
        )
    print()

    print("Optimized parameters")
    print("-" * 88)
    for r in results:
        print(f"{r.case}:")
        print(f"  success : {r.success}")
        print(f"  message : {r.message}")
        print(f"  params  : {format_params(r.params)}")
    print()


def print_finite_shot_results(best_result: VQEResult) -> None:
    print("=" * 88)
    print("Finite-shot estimate at optimized noiseless parameters")
    print("=" * 88)
    print(f"Best two-qubit parity-tapered circuit: {best_result.case}")
    print(f"Target QPU/noisy backend             : {TARGET_QPU}")
    print(f"Noise-model source                   : {NOISE_MODEL_SOURCE}")
    print(f"Main noise parameters                : readout_error={NOISE_READOUT_ERROR}, "
          f"expectation_shrink={NOISE_EXPECTATION_SHRINK}")
    print("Shot allocation rule                 : proportional to |Pauli coefficient|")
    print()

    if "UCCSD" in best_result.case:
        state = uccsd_parity_state(best_result.params)
    else:
        state = real_amplitude_state(best_result.params, n_qubits=2, layers=2, final_rotation=True)

    rng_ideal = np.random.default_rng(SEED + 101)
    rng_noisy = np.random.default_rng(SEED + 202)

    header = f"{'shots':>10s} {'backend':>12s} {'sample mean E (Ha)':>22s} {'standard error':>18s}"
    print(header)
    print("-" * len(header))

    last_allocation = None
    for shots in FINITE_SHOTS:
        e_ideal, se_ideal, alloc = finite_shot_energy(PARITY_H2, state, shots, rng_ideal, noisy=False)
        e_noisy, se_noisy, _ = finite_shot_energy(PARITY_H2, state, shots, rng_noisy, noisy=True)
        last_allocation = alloc

        print(f"{shots:10d} {'ideal':>12s} {e_ideal:22.12f} {se_ideal:18.6e}")
        print(f"{shots:10d} {'noisy':>12s} {e_noisy:22.12f} {se_noisy:18.6e}")

    print()
    print("Final shot allocation example for the largest shot budget:")
    if last_allocation is not None:
        for term_index, shots in sorted(last_allocation.items()):
            coeff, pauli = PARITY_H2.terms[term_index]
            label = "".join(f"{op}{q}" for q, op in sorted(pauli.items()))
            print(f"  term {term_index:2d}: {coeff:+.12f} * {label:<6s} -> {shots} shots")
    print()


# ============================================================
# Main
# ============================================================

def main() -> None:
    print_hamiltonian_summary()

    rng = np.random.default_rng(SEED)

    results: List[VQEResult] = []

    # Case 1: UCCSD + JW
    results.append(
        run_vqe(
            case="UCCSD + JW",
            ham=JW_H2,
            state_fn=uccsd_jw_state,
            n_params=1,
            rng=rng,
        )
    )

    # Case 2: UCCSD + parity
    results.append(
        run_vqe(
            case="UCCSD + parity",
            ham=PARITY_H2,
            state_fn=uccsd_parity_state,
            n_params=1,
            rng=rng,
        )
    )

    # Case 3: real-amplitude + JW
    results.append(
        run_vqe(
            case="real-amplitude + JW",
            ham=JW_H2,
            state_fn=lambda x: real_amplitude_state(x, n_qubits=4, layers=2, final_rotation=True),
            n_params=4 * 3,
            rng=rng,
        )
    )

    # Case 4: real-amplitude + parity
    results.append(
        run_vqe(
            case="real-amplitude + parity",
            ham=PARITY_H2,
            state_fn=lambda x: real_amplitude_state(x, n_qubits=2, layers=2, final_rotation=True),
            n_params=2 * 3,
            rng=rng,
        )
    )

    print_vqe_results(results)

    parity_results = [r for r in results if "parity" in r.case]
    best_parity = min(parity_results, key=lambda r: r.abs_error)
    print_finite_shot_results(best_parity)


if __name__ == "__main__":
    main()
