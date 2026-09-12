# ============================================================
# ieee_case.py
# Full IEEE 14-bus test case (MATPOWER case14 data)
# ============================================================

import numpy as np


# MATPOWER-style bus type
PQ = 1
PV = 2
REF = 3

# bus columns
BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
GS = 4
BS = 5
VM = 6
VA = 7

# gen columns
GEN_BUS = 0
PG = 1
QG = 2
VG = 3

# branch columns
F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
TAP = 5
SHIFT = 6
BR_STATUS = 7


def load_ieee14():
    """
    Full IEEE 14-bus power-flow case.

    Data are taken from MATPOWER case14 and represented with the same
    compact columns used by the original 3-bus program.

    Bus types:
        Bus 1: REF/slack
        Buses 2, 3, 6, 8: PV
        Buses 4, 5, 7, 9, 10, 11, 12, 13, 14: PQ
    """

    base_mva = 100.0

    # bus_i, type, Pd, Qd, Gs, Bs, Vm, Va(deg)
    bus = np.array([
        [ 1, REF,  0.0,   0.0, 0.0,  0.0, 1.060,   0.00],
        [ 2, PV,  21.7,  12.7, 0.0,  0.0, 1.045,  -4.98],
        [ 3, PV,  94.2,  19.0, 0.0,  0.0, 1.010, -12.72],
        [ 4, PQ,  47.8,  -3.9, 0.0,  0.0, 1.019, -10.33],
        [ 5, PQ,   7.6,   1.6, 0.0,  0.0, 1.020,  -8.78],
        [ 6, PV,  11.2,   7.5, 0.0,  0.0, 1.070, -14.22],
        [ 7, PQ,   0.0,   0.0, 0.0,  0.0, 1.062, -13.37],
        [ 8, PV,   0.0,   0.0, 0.0,  0.0, 1.090, -13.36],
        [ 9, PQ,  29.5,  16.6, 0.0, 19.0, 1.056, -14.94],
        [10, PQ,   9.0,   5.8, 0.0,  0.0, 1.051, -15.10],
        [11, PQ,   3.5,   1.8, 0.0,  0.0, 1.057, -14.79],
        [12, PQ,   6.1,   1.6, 0.0,  0.0, 1.055, -15.07],
        [13, PQ,  13.5,   5.8, 0.0,  0.0, 1.050, -15.16],
        [14, PQ,  14.9,   5.0, 0.0,  0.0, 1.036, -16.04],
    ], dtype=float)

    # gen_bus, Pg, Qg, Vg
    gen = np.array([
        [1, 232.4, -16.9, 1.060],
        [2,  40.0,  42.4, 1.045],
        [3,   0.0,  23.4, 1.010],
        [6,   0.0,  12.2, 1.070],
        [8,   0.0,  17.4, 1.090],
    ], dtype=float)

    # fbus, tbus, r, x, b, tap, shift(deg), status
    # tap = 0 means a unity tap ratio, following MATPOWER convention.
    branch = np.array([
        [ 1,  2, 0.01938, 0.05917, 0.0528, 0.000, 0.0, 1],
        [ 1,  5, 0.05403, 0.22304, 0.0492, 0.000, 0.0, 1],
        [ 2,  3, 0.04699, 0.19797, 0.0438, 0.000, 0.0, 1],
        [ 2,  4, 0.05811, 0.17632, 0.0340, 0.000, 0.0, 1],
        [ 2,  5, 0.05695, 0.17388, 0.0346, 0.000, 0.0, 1],
        [ 3,  4, 0.06701, 0.17103, 0.0128, 0.000, 0.0, 1],
        [ 4,  5, 0.01335, 0.04211, 0.0000, 0.000, 0.0, 1],
        [ 4,  7, 0.00000, 0.20912, 0.0000, 0.978, 0.0, 1],
        [ 4,  9, 0.00000, 0.55618, 0.0000, 0.969, 0.0, 1],
        [ 5,  6, 0.00000, 0.25202, 0.0000, 0.932, 0.0, 1],
        [ 6, 11, 0.09498, 0.19890, 0.0000, 0.000, 0.0, 1],
        [ 6, 12, 0.12291, 0.25581, 0.0000, 0.000, 0.0, 1],
        [ 6, 13, 0.06615, 0.13027, 0.0000, 0.000, 0.0, 1],
        [ 7,  8, 0.00000, 0.17615, 0.0000, 0.000, 0.0, 1],
        [ 7,  9, 0.00000, 0.11001, 0.0000, 0.000, 0.0, 1],
        [ 9, 10, 0.03181, 0.08450, 0.0000, 0.000, 0.0, 1],
        [ 9, 14, 0.12711, 0.27038, 0.0000, 0.000, 0.0, 1],
        [10, 11, 0.08205, 0.19207, 0.0000, 0.000, 0.0, 1],
        [12, 13, 0.22092, 0.19988, 0.0000, 0.000, 0.0, 1],
        [13, 14, 0.17093, 0.34802, 0.0000, 0.000, 0.0, 1],
    ], dtype=float)

    case = {
        "base_mva": base_mva,
        "bus": bus,
        "gen": gen,
        "branch": branch,
    }

    case["slack"] = np.where(bus[:, BUS_TYPE] == REF)[0]
    case["pv"] = np.where(bus[:, BUS_TYPE] == PV)[0]
    case["pq"] = np.where(bus[:, BUS_TYPE] == PQ)[0]
    case["non_slack"] = np.where(bus[:, BUS_TYPE] != REF)[0]

    return case


def make_ybus(case):
    """
    Construct Ybus, including:
        - line series admittance,
        - half-line charging at both ends,
        - off-nominal transformer tap and phase shift,
        - bus shunt conductance/susceptance.
    """

    bus = case["bus"]
    branch = case["branch"]
    base_mva = case["base_mva"]

    nb = bus.shape[0]
    Ybus = np.zeros((nb, nb), dtype=complex)

    for br in branch:
        if int(br[BR_STATUS]) == 0:
            continue

        f = int(br[F_BUS]) - 1
        t = int(br[T_BUS]) - 1

        r = br[BR_R]
        x = br[BR_X]
        b = br[BR_B]

        z = r + 1j * x
        if abs(z) < 1e-14:
            raise ValueError(f"Branch {f + 1}-{t + 1} has zero impedance.")

        y = 1 / z
        y_shunt = 1j * b / 2

        tap_ratio = br[TAP]
        if abs(tap_ratio) < 1e-14:
            tap_ratio = 1.0

        shift_rad = np.deg2rad(br[SHIFT])
        tap = tap_ratio * np.exp(1j * shift_rad)

        Yff = (y + y_shunt) / (tap * np.conj(tap))
        Yft = -y / np.conj(tap)
        Ytf = -y / tap
        Ytt = y + y_shunt

        Ybus[f, f] += Yff
        Ybus[f, t] += Yft
        Ybus[t, f] += Ytf
        Ybus[t, t] += Ytt

    # MATPOWER convention: Gs and Bs are specified in MW/MVAr at V=1 p.u.
    Ybus[np.diag_indices(nb)] += (
        bus[:, GS] + 1j * bus[:, BS]
    ) / base_mva

    return Ybus


def make_B_matrices(case, Ybus):
    """
    Keep the same simplified FDLS construction as the original code:
        B'  = -imag(Ybus), reduced to non-slack buses
        B'' = -imag(Ybus), reduced to PQ buses
    """

    B = -Ybus.imag

    non_slack = case["non_slack"]
    pq = case["pq"]

    Bprime = B[np.ix_(non_slack, non_slack)]
    Bdouble = B[np.ix_(pq, pq)]

    return Bprime, Bdouble


def make_specified_power(case):
    bus = case["bus"]
    gen = case["gen"]
    base_mva = case["base_mva"]

    nb = bus.shape[0]

    P_spec = -bus[:, PD] / base_mva
    Q_spec = -bus[:, QD] / base_mva

    for g in gen:
        bus_idx = int(g[GEN_BUS]) - 1
        P_spec[bus_idx] += g[PG] / base_mva
        Q_spec[bus_idx] += g[QG] / base_mva

    return P_spec, Q_spec


def initial_voltage(case):
    bus = case["bus"]

    vm = bus[:, VM].copy()
    va = np.deg2rad(bus[:, VA].copy())

    return vm, va


def calculated_power(Ybus, vm, va):
    V = vm * np.exp(1j * va)
    S = V * np.conj(Ybus @ V)

    P = S.real
    Q = S.imag

    return P, Q


def print_ieee_data(case, Ybus, Bprime, Bdouble):
    np.set_printoptions(precision=8, suppress=True)

    print("\n" + "=" * 80)
    print("FULL IEEE 14-BUS DATA")
    print("=" * 80)

    print("\nBus data:")
    print("columns: bus_i, type, Pd, Qd, Gs, Bs, Vm, Va")
    print(case["bus"])

    print("\nGen data:")
    print("columns: gen_bus, Pg, Qg, Vg")
    print(case["gen"])

    print("\nBranch data:")
    print("columns: fbus, tbus, r, x, b, tap, shift, status")
    print(case["branch"])

    print("\nYbus:")
    print(Ybus)

    print("\nBprime = -imag(Ybus), reduced non-slack:")
    print(Bprime)

    print("\nBdouble = -imag(Ybus), reduced PQ:")
    print(Bdouble)

    print("\nSlack index:", case["slack"])
    print("PV index:   ", case["pv"])
    print("PQ index:   ", case["pq"])
    print("Non-slack:  ", case["non_slack"])