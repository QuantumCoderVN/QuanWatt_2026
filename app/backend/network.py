# ============================================================
# network.py
# Network construction and validation utilities
# ============================================================

import numpy as np
from typing import Dict, List, Tuple, Optional


# MATPOWER-style bus types
PQ = 1
PV = 2
REF = 3

# Bus column indices
BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
GS = 4
BS = 5
VM = 6
VA = 7

# Gen column indices
GEN_BUS = 0
PG = 1
QG = 2
VG = 3

# Branch column indices
F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
TAP = 5
SHIFT = 6
BR_STATUS = 7


def validate_case(case: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate network case data.

    Returns:
        (is_valid, error_message)
    """
    try:
        bus = case["bus"]
        branch = case["branch"]
        base_mva = case["base_mva"]

        # Check base MVA is positive
        if base_mva <= 0:
            return False, "Base MVA must be positive"

        # Check finite inputs
        if not np.all(np.isfinite(bus)):
            return False, "Bus data contains non-finite values"

        if not np.all(np.isfinite(branch)):
            return False, "Branch data contains non-finite values"

        # Check positive voltage magnitudes
        if not np.all(bus[:, VM] > 0):
            return False, "All voltage magnitudes must be positive"

        # Check exactly one slack bus
        slack_count = np.sum(bus[:, BUS_TYPE] == REF)
        if slack_count != 1:
            return False, f"Must have exactly one slack bus, found {slack_count}"

        # Check valid branch endpoints
        bus_ids = set(bus[:, BUS_I].astype(int))
        for br in branch:
            if int(br[BR_STATUS]) == 0:
                continue
            f = int(br[F_BUS])
            t = int(br[T_BUS])
            if f not in bus_ids or t not in bus_ids:
                return False, f"Branch endpoint {f}-{t} references non-existent bus"

        # Check connectivity (simplified - could be more thorough)
        n_buses = len(bus)
        if n_buses < 2:
            return False, "Must have at least 2 buses"

        # Check for non-zero resistance and reactance in active branches
        for br in branch:
            if int(br[BR_STATUS]) == 0:
                continue
            r = br[BR_R]
            x = br[BR_X]
            if abs(r) < 1e-12 and abs(x) < 1e-12:
                return False, f"Branch {int(br[F_BUS])}-{int(br[T_BUS])} has zero impedance"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def make_ybus(case: Dict) -> np.ndarray:
    """
    Build complex bus-admittance matrix from case data.
    Includes branch impedances, charging, and real transformer taps.
    """
    bus = case["bus"]
    branch = case["branch"]

    nb = bus.shape[0]
    bus_ids = bus[:, BUS_I].astype(int)

    # Create mapping from bus ID to index
    bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}

    Ybus = np.zeros((nb, nb), dtype=complex)

    # Add shunt admittances
    for i in range(nb):
        gs = bus[i, GS]
        bs = bus[i, BS]
        Ybus[i, i] += gs + 1j * bs

    # Add branch admittances
    for br in branch:
        if int(br[BR_STATUS]) == 0:
            continue

        f_id = int(br[F_BUS])
        t_id = int(br[T_BUS])

        if f_id not in bus_id_to_idx or t_id not in bus_id_to_idx:
            continue

        f = bus_id_to_idx[f_id]
        t = bus_id_to_idx[t_id]

        r = br[BR_R]
        x = br[BR_X]
        b = br[BR_B]
        tap = br[TAP] if br[TAP] != 0 else 1.0

        z = r + 1j * x
        y = 1 / z
        y_shunt = 1j * b / 2

        # Handle transformer tap ratio (real taps only, no phase shift)
        tap2 = tap * tap

        Ybus[f, f] += y / tap2 + y_shunt
        Ybus[t, t] += y + y_shunt

        Ybus[f, t] -= y / tap
        Ybus[t, f] -= y / tap

    return Ybus


def make_B_matrices(case: Dict, Ybus: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build FDLS decoupled matrices:
        B'  = -imag(Ybus) reduced to non-slack buses
        B'' = -imag(Ybus) reduced to PQ buses
    """
    B = -Ybus.imag

    non_slack = case["non_slack"]
    pq = case["pq"]

    Bprime = B[np.ix_(non_slack, non_slack)]
    Bdouble = B[np.ix_(pq, pq)]

    return Bprime, Bdouble


def validate_B_matrices(Bprime: np.ndarray, Bdouble: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Validate that FDLS matrices are symmetric and nonsingular.
    """
    # Check Bprime
    if len(Bprime) > 0:
        if not np.allclose(Bprime, Bprime.T):
            return False, "B' matrix is not symmetric"

        try:
            cond = np.linalg.cond(Bprime)
            if cond > 1e12:
                return False, f"B' matrix is nearly singular (cond={cond:.2e})"
        except np.linalg.LinAlgError:
            return False, "B' matrix is singular"

    # Check Bdouble
    if len(Bdouble) > 0:
        if not np.allclose(Bdouble, Bdouble.T):
            return False, "B'' matrix is not symmetric"

        try:
            cond = np.linalg.cond(Bdouble)
            if cond > 1e12:
                return False, f"B'' matrix is nearly singular (cond={cond:.2e})"
        except np.linalg.LinAlgError:
            return False, "B'' matrix is singular"

    return True, None


def make_specified_power(case: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate specified active and reactive power injections in per-unit.
    """
    bus = case["bus"]
    gen = case.get("gen", np.array([]))
    base_mva = case["base_mva"]

    nb = bus.shape[0]

    # Net injection = generation - load
    P_spec = -bus[:, PD] / base_mva
    Q_spec = -bus[:, QD] / base_mva

    # Add generation
    for g in gen:
        bus_idx = int(g[GEN_BUS]) - 1
        if 0 <= bus_idx < nb:
            P_spec[bus_idx] += g[PG] / base_mva
            Q_spec[bus_idx] += g[QG] / base_mva

    return P_spec, Q_spec


def initial_voltage(case: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract initial voltage magnitudes and angles from case data.
    """
    bus = case["bus"]

    vm = bus[:, VM].copy()
    va = np.deg2rad(bus[:, VA].copy())

    return vm, va


def calculated_power(Ybus: np.ndarray, vm: np.ndarray, va: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate active and reactive power injections from voltages.
    """
    V = vm * np.exp(1j * va)
    S = V * np.conj(Ybus @ V)

    P = S.real
    Q = S.imag

    return P, Q


def prepare_case_indices(case: Dict) -> Dict:
    """
    Prepare bus indices for FDLS: slack, PV, PQ, non-slack.
    Modifies case dict in-place.
    """
    bus = case["bus"]

    case["slack"] = np.where(bus[:, BUS_TYPE] == REF)[0]
    case["pv"] = np.where(bus[:, BUS_TYPE] == PV)[0]
    case["pq"] = np.where(bus[:, BUS_TYPE] == PQ)[0]
    case["non_slack"] = np.where(bus[:, BUS_TYPE] != REF)[0]

    return case


def default_case() -> Dict:
    """
    Return the default IEEE-14-derived 3-bus test case.
    """
    base_mva = 100.0

    # bus_i, type, Pd, Qd, Gs, Bs, Vm, Va(deg)
    bus = np.array([
        [1, REF,  0.0,  0.0, 0.0, 0.0, 1.060, 0.0],
        [2, PV,  21.7, 12.7, 0.0, 0.0, 1.045, 0.0],
        [3, PQ,  94.2, 19.0, 0.0, 0.0, 1.010, 0.0],
    ], dtype=float)

    # gen_bus, Pg, Qg, Vg
    gen = np.array([
        [1, 232.4, -16.9, 1.060],
        [2,  40.0,  42.4, 1.045],
    ], dtype=float)

    # fbus, tbus, r, x, b, tap, shift, status
    branch = np.array([
        [1, 2, 0.01938, 0.05917, 0.0528, 0.0, 0.0, 1],
        [2, 3, 0.04699, 0.19797, 0.0438, 0.0, 0.0, 1],
    ], dtype=float)

    case = {
        "base_mva": base_mva,
        "bus": bus,
        "gen": gen,
        "branch": branch,
    }

    prepare_case_indices(case)

    return case
