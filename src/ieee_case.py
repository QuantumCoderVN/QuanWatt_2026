# ============================================================
# ieee_case.py
# IEEE-14 reduced 3-bus test case
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


def load_ieee14_reduced_3bus():
    """
    IEEE-14 reduced 3-bus.

    Bus 1: Slack
    Bus 2: PV
    Bus 3: PQ

    Dữ liệu nhánh lấy theo cặp nhánh IEEE-14:
    1-2 và 2-3.
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

    case["slack"] = np.where(bus[:, BUS_TYPE] == REF)[0]
    case["pv"] = np.where(bus[:, BUS_TYPE] == PV)[0]
    case["pq"] = np.where(bus[:, BUS_TYPE] == PQ)[0]
    case["non_slack"] = np.where(bus[:, BUS_TYPE] != REF)[0]

    return case


def make_ybus(case):
    bus = case["bus"]
    branch = case["branch"]

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
        y = 1 / z
        y_shunt = 1j * b / 2

        Ybus[f, f] += y + y_shunt
        Ybus[t, t] += y + y_shunt

        Ybus[f, t] -= y
        Ybus[t, f] -= y

    return Ybus


def make_B_matrices(case, Ybus):
    """
    FDLS:
        B'  = -imag(Ybus) reduced non-slack
        B'' = -imag(Ybus) reduced PQ
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
    print("IEEE-14 REDUCED 3-BUS DATA")
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