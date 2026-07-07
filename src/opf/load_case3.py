import re
import html
import urllib.request
import numpy as np

np.set_printoptions(
    precision=6,
    suppress=True,
    linewidth=220
)

# ============================================================
# 1. Download MATPOWER case3sc source from official MATPOWER docs
# ============================================================

MATPOWER_CASE3SC_URL = (
    "https://matpower.org/docs/ref/matpower7.1/extras/sdp_pf/case3sc.html"
)


def download_case3sc_html(url=MATPOWER_CASE3SC_URL):
    """
    Download official MATPOWER case3sc source page.

    Fix HTTP 406 by sending browser-like headers.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    request = urllib.request.Request(
        url,
        headers=headers,
        method="GET"
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        raw_html = response.read().decode("utf-8", errors="ignore")

    # Convert HTML to plain text
    text = re.sub(r"<[^>]+>", "", raw_html)
    text = html.unescape(text)

    return text
def clean_matpower_line(line):
    """
    Clean one line from MATPOWER m-file source.
    Remove line numbers, comments, semicolons.
    """
    # Remove source-code line number such as: 0026
    line = re.sub(r"^\s*\d{4}\s*", "", line)

    # Remove MATLAB comments
    line = line.split("%")[0]

    # Remove semicolon
    line = line.replace(";", " ")

    return line.strip()

def parse_matpower_matrix(text, field_name):
    """
    Parse matrix block like:
        mpc.bus = [
            ...
        ];
    """
    pattern = rf"mpc\.{field_name}\s*=\s*\[(.*?)\];"
    match = re.search(pattern, text, flags=re.DOTALL)

    if match is None:
        raise ValueError(f"Cannot find mpc.{field_name} in MATPOWER source.")

    block = match.group(1)

    rows = []
    for line in block.splitlines():
        line = clean_matpower_line(line)

        if not line:
            continue

        values = [float(x) for x in line.split()]
        rows.append(values)

    return np.array(rows, dtype=float)


def parse_base_mva(text):
    """
    Parse:
        mpc.baseMVA = 100;
    """
    pattern = r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)\s*;"
    match = re.search(pattern, text)

    if match is None:
        raise ValueError("Cannot find mpc.baseMVA in MATPOWER source.")

    return float(match.group(1))


def load_matpower_case3sc_from_official_html():
    """
    Actually load case3sc from MATPOWER official source page.
    """
    text = download_case3sc_html()

    mpc = {
        "baseMVA": parse_base_mva(text),
        "bus": parse_matpower_matrix(text, "bus"),
        "gen": parse_matpower_matrix(text, "gen"),
        "branch": parse_matpower_matrix(text, "branch"),
        "gencost": parse_matpower_matrix(text, "gencost"),
    }

    return mpc


# ============================================================
# 2. Build a simple DC-OPF KKT matrix from loaded MATPOWER case
# ============================================================

def build_dc_kkt_from_matpower_case(mpc):
    """
    Build simple DC-OPF equality-constrained KKT system.

    Variables:
        x = [theta_non_slack, Pg_1, Pg_2, ..., Pg_ng]

    KKT unknown:
        y = [x, lambda]

    KKT matrix:
        [ H      Aeq.T ]
        [ Aeq    0    ]

    Convention:
        baseMVA * Bbus * theta - Cg * Pg = -Pd
    """

    baseMVA = mpc["baseMVA"]
    bus = mpc["bus"]
    gen = mpc["gen"]
    branch = mpc["branch"]
    gencost = mpc["gencost"]

    n_bus = bus.shape[0]
    n_gen = gen.shape[0]

    # -----------------------------
    # MATPOWER column indices
    # -----------------------------
    BUS_I = 0
    BUS_TYPE = 1
    PD = 2

    GEN_BUS = 0

    F_BUS = 0
    T_BUS = 1
    BR_X = 3

    COST_C2 = 4
    COST_C1 = 5

    # -----------------------------
    # Find slack bus
    # MATPOWER bus type 3 = reference/slack
    # -----------------------------
    slack_candidates = np.where(bus[:, BUS_TYPE] == 3)[0]
    if len(slack_candidates) == 0:
        raise ValueError("No slack bus found. MATPOWER bus type 3 is required.")

    slack_bus_internal = int(slack_candidates[0])

    angle_buses = [i for i in range(n_bus) if i != slack_bus_internal]

    # -----------------------------
    # Build Bbus from branch x
    # -----------------------------
    Bbus = np.zeros((n_bus, n_bus), dtype=float)

    # Map MATPOWER external bus number to internal row index
    bus_number_to_index = {
        int(bus[i, BUS_I]): i
        for i in range(n_bus)
    }

    for br in branch:
        f_ext = int(br[F_BUS])
        t_ext = int(br[T_BUS])

        f = bus_number_to_index[f_ext]
        t = bus_number_to_index[t_ext]

        x = br[BR_X]
        bij = 1.0 / x

        Bbus[f, f] += bij
        Bbus[t, t] += bij
        Bbus[f, t] -= bij
        Bbus[t, f] -= bij

    # -----------------------------
    # Build generator incidence Cg
    # Cg[i, g] = 1 if generator g is at bus i
    # -----------------------------
    Cg = np.zeros((n_bus, n_gen), dtype=float)

    for g in range(n_gen):
        gen_bus_ext = int(gen[g, GEN_BUS])
        gen_bus_internal = bus_number_to_index[gen_bus_ext]
        Cg[gen_bus_internal, g] = 1.0

    # -----------------------------
    # Build Aeq x = beq
    # x = [theta_non_slack, Pg]
    # -----------------------------
    Bred = Bbus[:, angle_buses]

    Aeq = np.hstack([
        baseMVA * Bred,
        -Cg
    ])

    Pd = bus[:, PD]
    beq = -Pd

    # -----------------------------
    # Build quadratic cost Hessian H and gradient g
    # f(Pg) = c2 Pg^2 + c1 Pg + c0
    # H(Pg_i, Pg_i) = 2*c2_i
    # grad linear part = c1_i
    # -----------------------------
    n_angle = len(angle_buses)
    n_x = n_angle + n_gen

    H = np.zeros((n_x, n_x), dtype=float)
    grad = np.zeros(n_x, dtype=float)

    for g_idx in range(n_gen):
        x_idx = n_angle + g_idx

        c2 = gencost[g_idx, COST_C2]
        c1 = gencost[g_idx, COST_C1]

        H[x_idx, x_idx] = 2.0 * c2
        grad[x_idx] = c1

    # -----------------------------
    # KKT system
    # [H Aeq.T] [x     ] = [-grad]
    # [Aeq  0  ] [lambda]   [beq  ]
    # -----------------------------
    KKT = np.block([
        [H, Aeq.T],
        [Aeq, np.zeros((n_bus, n_bus))]
    ])

    rhs = np.concatenate([
        -grad,
        beq
    ])

    return {
        "Bbus": Bbus,
        "Aeq": Aeq,
        "H": H,
        "KKT": KKT,
        "rhs": rhs,
        "slack_bus_internal": slack_bus_internal,
        "angle_buses": angle_buses,
    }


# ============================================================
# 3. Main
# ============================================================

if __name__ == "__main__":
    print("Loading MATPOWER case3sc from:")
    print(MATPOWER_CASE3SC_URL)

    mpc = load_matpower_case3sc_from_official_html()

    print("\nLoaded MATPOWER data:")
    print("baseMVA =", mpc["baseMVA"])
    print("bus shape     =", mpc["bus"].shape)
    print("gen shape     =", mpc["gen"].shape)
    print("branch shape  =", mpc["branch"].shape)
    print("gencost shape =", mpc["gencost"].shape)

    print("\nmpc.bus =")
    print(mpc["bus"])

    print("\nmpc.gen =")
    print(mpc["gen"])

    print("\nmpc.branch =")
    print(mpc["branch"])

    print("\nmpc.gencost =")
    print(mpc["gencost"])

    result = build_dc_kkt_from_matpower_case(mpc)

    print("\nBbus =")
    print(result["Bbus"])

    print("\nAeq =")
    print(result["Aeq"])

    print("\nH =")
    print(result["H"])

    print("\nA_OPF_KKT =")
    print(result["KKT"])

    print("\nb_OPF_rhs =")
    print(result["rhs"])

    print("\nShape of A_OPF_KKT:", result["KKT"].shape)
    print("Condition number:", np.linalg.cond(result["KKT"]))