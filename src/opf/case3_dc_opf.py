# ============================================================
# case3_dc_opf.py
# IEEE14-subcase-3bus data loader
# ============================================================

import numpy as np

from pypower.case14 import case14

from pypower.idx_bus import BUS_I, BUS_TYPE, REF, PD
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PMAX, PMIN
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_cost import MODEL, NCOST

from config_qopf3 import SELECTED_BUSES


def _copy_ppc(ppc):
    out = {}
    for k, v in ppc.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


def extract_case14_3bus(selected_buses=SELECTED_BUSES):
    """
    Lấy dữ liệu thật từ PYPOWER case14.

    Giữ:
        - bus thuộc selected_buses
        - branch có cả from-bus và to-bus trong selected_buses
        - generator online nằm trong selected_buses
        - gencost tương ứng với generator đó

    Đây không phải case tự tạo thủ công.
    """

    ppc = case14()

    baseMVA = float(ppc["baseMVA"])
    bus = np.asarray(ppc["bus"], dtype=float)
    gen = np.asarray(ppc["gen"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float)
    gencost = np.asarray(ppc["gencost"], dtype=float)

    selected_buses = list(selected_buses)
    selected_set = set(selected_buses)

    bus_mask = np.array([
        int(row[BUS_I]) in selected_set
        for row in bus
    ])

    bus_sub = bus[bus_mask].copy()

    branch_mask = np.array([
        int(row[F_BUS]) in selected_set and int(row[T_BUS]) in selected_set
        for row in branch
    ])

    branch_sub = branch[branch_mask].copy()

    gen_mask = np.array([
        int(row[GEN_BUS]) in selected_set and row[GEN_STATUS] > 0
        for row in gen
    ])

    gen_sub = gen[gen_mask].copy()
    gencost_sub = gencost[gen_mask].copy()

    if bus_sub.shape[0] != len(selected_buses):
        raise ValueError("Không lấy đủ số bus đã chọn từ case14.")

    if branch_sub.shape[0] == 0:
        raise ValueError("Không có branch nào nối giữa các bus đã chọn.")

    if gen_sub.shape[0] == 0:
        raise ValueError("Không có generator online nào trong các bus đã chọn.")

    # Đảm bảo có slack bus.
    ref_idx = np.where(bus_sub[:, BUS_TYPE] == REF)[0]

    if len(ref_idx) == 0:
        bus_sub[0, BUS_TYPE] = REF
        slack_bus_number = int(bus_sub[0, BUS_I])
    else:
        slack_bus_number = int(bus_sub[ref_idx[0], BUS_I])

    ppc3 = {
        "version": ppc.get("version", "2"),
        "baseMVA": baseMVA,
        "bus": bus_sub,
        "gen": gen_sub,
        "branch": branch_sub,
        "gencost": gencost_sub,
    }

    return ppc3, selected_buses, slack_bus_number


def load_case3_dc_opf_from_ieee14():
    ppc3, selected_buses, slack_bus_number = extract_case14_3bus()

    return {
        "name": "ieee14_subcase_3bus",
        "baseMVA": float(ppc3["baseMVA"]),
        "ppc": _copy_ppc(ppc3),
        "bus": ppc3["bus"].copy(),
        "gen": ppc3["gen"].copy(),
        "branch": ppc3["branch"].copy(),
        "gencost": ppc3["gencost"].copy(),
        "selected_buses": selected_buses,
        "slack_bus_number": slack_bus_number,
    }


def extract_quadratic_cost_pu(case):
    """
    PYPOWER gencost dùng đơn vị MW:

        cost(Pg_MW) = c2 Pg_MW^2 + c1 Pg_MW + c0

    Code của ta dùng biến Pg_pu:

        Pg_MW = baseMVA * Pg_pu

    Do đó:

        cost(Pg_pu)
        = c2 * baseMVA^2 * Pg_pu^2
        + c1 * baseMVA * Pg_pu
        + c0

    QP chuẩn:

        0.5 z^T H z + f^T z

    nên:

        H_ii = 2 * c2 * baseMVA^2
        f_i  = c1 * baseMVA
    """

    baseMVA = float(case["baseMVA"])
    gencost = np.asarray(case["gencost"], dtype=float)

    ng = gencost.shape[0]

    c2_pu = np.zeros(ng)
    c1_pu = np.zeros(ng)
    c0 = np.zeros(ng)

    for i in range(ng):
        model = int(gencost[i, MODEL])
        ncost = int(gencost[i, NCOST])

        if model != 2:
            raise ValueError("Chỉ hỗ trợ polynomial gencost MODEL = 2.")

        coeffs = gencost[i, -ncost:]

        if ncost == 3:
            c2_mw, c1_mw, c0_i = coeffs
        elif ncost == 2:
            c2_mw = 0.0
            c1_mw, c0_i = coeffs
        elif ncost == 1:
            c2_mw = 0.0
            c1_mw = 0.0
            c0_i = coeffs[0]
        else:
            raise ValueError("Chỉ hỗ trợ cost bậc 0, 1 hoặc 2.")

        c2_pu[i] = c2_mw * baseMVA**2
        c1_pu[i] = c1_mw * baseMVA
        c0[i] = c0_i

    return c2_pu, c1_pu, c0


def objective_value(z, case):
    ng = case["gen"].shape[0]
    Pg_pu = np.asarray(z[:ng], dtype=float)

    c2_pu, c1_pu, c0 = extract_quadratic_cost_pu(case)

    return float(np.sum(c2_pu * Pg_pu**2 + c1_pu * Pg_pu + c0))


def make_variable_names(case):
    gen = case["gen"]
    bus = case["bus"]
    slack_bus_number = case["slack_bus_number"]

    names = []

    for i in range(gen.shape[0]):
        gen_bus = int(gen[i, GEN_BUS])
        names.append(f"Pg_gen{i+1}_bus{gen_bus}_pu")

    for b in bus[:, BUS_I]:
        b = int(b)
        if b != slack_bus_number:
            names.append(f"theta_bus{b}_rad")

    return names


def parse_solution(z, case):
    baseMVA = float(case["baseMVA"])
    gen = case["gen"]
    bus = case["bus"]

    ng = gen.shape[0]

    Pg_pu = np.asarray(z[:ng], dtype=float)
    theta_non_slack = np.asarray(z[ng:], dtype=float)

    slack_bus_number = case["slack_bus_number"]
    bus_numbers = [int(v) for v in bus[:, BUS_I]]

    theta_by_bus = {slack_bus_number: 0.0}

    non_slack_buses = [
        b for b in bus_numbers
        if b != slack_bus_number
    ]

    for b, theta in zip(non_slack_buses, theta_non_slack):
        theta_by_bus[b] = float(theta)

    parsed = {}

    for i in range(ng):
        gen_bus = int(gen[i, GEN_BUS])
        parsed[f"Pg_gen{i+1}_bus{gen_bus}_pu"] = float(Pg_pu[i])
        parsed[f"Pg_gen{i+1}_bus{gen_bus}_MW"] = float(Pg_pu[i] * baseMVA)
        parsed[f"Pg_gen{i+1}_bus{gen_bus}_Pmin_MW"] = float(gen[i, PMIN])
        parsed[f"Pg_gen{i+1}_bus{gen_bus}_Pmax_MW"] = float(gen[i, PMAX])

    for b in bus_numbers:
        theta = theta_by_bus[b]
        parsed[f"theta_bus{b}_rad"] = float(theta)
        parsed[f"theta_bus{b}_deg"] = float(np.rad2deg(theta))

    total_generation_pu = float(np.sum(Pg_pu))
    total_load_pu = float(np.sum(bus[:, PD]) / baseMVA)

    parsed["total_generation_pu"] = total_generation_pu
    parsed["total_generation_MW"] = total_generation_pu * baseMVA
    parsed["total_load_pu"] = total_load_pu
    parsed["total_load_MW"] = total_load_pu * baseMVA

    return parsed