from pathlib import Path
import sys, os, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
CLASS_DIR = ROOT / "class_geo_math_lab"
sys.path.insert(0, str(CLASS_DIR))

from classy import Class

DATA_DIR = CLASS_DIR / "data"
PANTHEON_DIR = DATA_DIR / "pantheon"
BAO_ROOT = DATA_DIR / "bao"

CSV_DIR = ROOT / "resultados" / "csv"
LOG_DIR = ROOT / "resultados" / "logs"
PLOT_DIR = ROOT / "docs" / "plots" / "geo_01_master_test"

for d in [CSV_DIR, LOG_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SN_FILE = PANTHEON_DIR / "Pantheon+SH0ES.dat"
SN_COV_FILE = PANTHEON_DIR / "Pantheon+SH0ES_STAT+SYS.cov"

OMEGA_B = 0.05
S8_BASE_OBS, S8_BASE_ERR = 0.776, 0.0325
RD_OBS, RD_ERR = 147.1, 0.3
OMBH2_OBS, OMBH2_ERR = 0.0224, 0.0001
OMMH2_OBS, OMMH2_ERR = 0.143, 0.002
LOGA_OBS, LOGA_ERR = 3.044, 0.014

WL_PRIORS = {
    "KiDS_like": (0.776, 0.0325),
    "DES_HSC_like": (0.759, 0.024),
    "DES_Y3_like": (0.776, 0.032),
}

FS8_DATA = [
    (0.02, 0.360, 0.040), (0.067, 0.423, 0.055),
    (0.10, 0.370, 0.130), (0.15, 0.490, 0.050),
    (0.17, 0.510, 0.060), (0.22, 0.420, 0.070),
    (0.25, 0.351, 0.058), (0.32, 0.384, 0.095),
    (0.37, 0.460, 0.038), (0.38, 0.430, 0.054),
    (0.44, 0.413, 0.080), (0.51, 0.452, 0.057),
    (0.57, 0.444, 0.038), (0.60, 0.390, 0.063),
    (0.61, 0.457, 0.052), (0.73, 0.437, 0.072),
    (0.80, 0.470, 0.080), (0.86, 0.400, 0.110),
    (1.40, 0.482, 0.116), (1.52, 0.426, 0.077),
    (1.944, 0.364, 0.106),
]

z_fs8 = np.array([x[0] for x in FS8_DATA])
fs8_obs = np.array([x[1] for x in FS8_DATA])
fs8_err = np.array([x[2] for x in FS8_DATA])


def load_pantheon():
    data = np.genfromtxt(SN_FILE, names=True, dtype=None, encoding=None)
    names = data.dtype.names
    z_col = "zHD" if "zHD" in names else ("zcmb" if "zcmb" in names else names[1])

    if "m_b_corr" in names:
        mag_col = "m_b_corr"
    elif "MU_SH0ES" in names:
        mag_col = "MU_SH0ES"
    elif "MU" in names:
        mag_col = "MU"
    else:
        raise ValueError("No encuentro columna MU/magnitud en Pantheon")

    z = np.array(data[z_col], dtype=float)
    m = np.array(data[mag_col], dtype=float)

    raw = np.loadtxt(SN_COV_FILE)
    if raw.ndim == 1:
        cov = raw[1:].reshape(len(z), len(z)) if int(raw[0]) == len(z) else raw.reshape(len(z), len(z))
    else:
        cov = raw

    cho = cho_factor(cov, lower=True, check_finite=False)
    ones = np.ones(len(z))
    print("Pantheon cargado:", len(z))
    return z, m, cho, ones


z_sn, m_sn, cov_sn_cho, ones_sn = load_pantheon()


def read_numbers(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            nums = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", line)
            if nums:
                rows.append([float(x) for x in nums])

    if not rows:
        raise ValueError("sin numeros")

    lens = [len(r) for r in rows]
    if len(set(lens)) == 1:
        arr = np.array(rows)
        return arr[:, 0] if arr.shape[1] == 1 else arr

    return np.array([x for r in rows for x in r])


def infer_z(path, mean_raw=None):
    name = os.path.basename(path).lower()
    arr = np.asarray(mean_raw) if mean_raw is not None else None

    if arr is not None and arr.ndim == 1 and len(arr) == 2 and 0.01 < arr[0] < 4:
        return float(arr[0])

    if arr is not None and arr.ndim == 2 and arr.shape[1] >= 2:
        z_col = arr[:, 0]
        if np.all((z_col > 0.01) & (z_col < 4)):
            return float(np.mean(z_col))

    if "bgs" in name: return 0.295
    if "lrg+elg" in name: return 0.934
    if "elg" in name: return 1.321
    if "qso" in name: return 1.484
    if "lya" in name: return 2.33
    return None


def parse_bao_pair(mean_path, cov_path):
    mean_raw = read_numbers(mean_path)
    cov_raw = read_numbers(cov_path)

    name = os.path.basename(mean_path).lower()
    z = infer_z(mean_path, mean_raw)
    mean_arr = np.asarray(mean_raw)
    cov_arr = np.asarray(cov_raw)

    if mean_arr.ndim == 1 and len(mean_arr) == 2 and cov_arr.ndim == 1 and len(cov_arr) == 1:
        return {"z": z, "obs": ["DV"], "mean": np.array([mean_arr[1]]), "cov": np.array([[cov_arr[0]]]), "name": os.path.basename(mean_path)}

    mean_vec = mean_arr[:, -1] if mean_arr.ndim == 2 and mean_arr.shape[1] >= 2 else mean_arr.flatten()

    if cov_arr.ndim == 1:
        if len(cov_arr) == len(mean_vec):
            cov = np.diag(cov_arr)
        elif len(cov_arr) == len(mean_vec) ** 2:
            cov = cov_arr.reshape(len(mean_vec), len(mean_vec))
        else:
            raise ValueError("cov incompatible")
    else:
        cov = cov_arr

    if "lya" in name:
        obs = ["DH", "DM"]
    elif len(mean_vec) == 2:
        obs = ["DM", "DH"]
    elif len(mean_vec) == 1:
        obs = ["DV"]
    else:
        raise ValueError("obs no interpretable")

    return {"z": z, "obs": obs, "mean": mean_vec, "cov": cov, "name": os.path.basename(mean_path)}


def load_bao(token="desi_gaussian_bao"):
    files = glob.glob(str(BAO_ROOT / "**" / "*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    means = [f for f in files if f.lower().endswith("_mean.txt") or f.lower().endswith("_mean")]
    covs = [f for f in files if f.lower().endswith("_cov.txt") or f.lower().endswith("_cov")]

    entries = []
    for mf in means:
        low = mf.lower()
        if token not in low or "all_gccomb" in low:
            continue

        base_m = re.sub(r"_mean(\.txt)?$", "", mf, flags=re.I)
        for cf in covs:
            base_c = re.sub(r"_cov(\.txt)?$", "", cf, flags=re.I)
            if os.path.normcase(base_m) == os.path.normcase(base_c):
                try:
                    entries.append(parse_bao_pair(mf, cf))
                except Exception as e:
                    print("SKIP BAO:", os.path.basename(mf), e)
                break
    return entries


BAO = load_bao()
print("BAO puntos:", sum(len(e["mean"]) for e in BAO))


def run_class(h, omega_m, logA, param, model):
    A_s = np.exp(logA) / 1e10
    omega_b = OMEGA_B * h**2
    omega_cdm = (omega_m - OMEGA_B) * h**2

    if omega_cdm <= 0:
        raise ValueError("omega_cdm <= 0")

    if model == "LCDM":
        geo_xi, geo_mu, geo_mode = 1.0, 1.0, 0
    elif model == "GEO_CLASS":
        geo_xi, geo_mu, geo_mode = param, 1.0, 0
    elif model == "FREE_MU":
        geo_xi, geo_mu, geo_mode = 1.0, param, 1
    else:
        raise ValueError(model)

    cosmo = Class()
    cosmo.set({
        "h": h,
        "omega_b": omega_b,
        "omega_cdm": omega_cdm,
        "A_s": A_s,
        "n_s": 0.965,
        "tau_reio": 0.054,
        "output": "mPk",
        "P_k_max_1/Mpc": 2.0,
        "z_max_pk": 2.5,
        "geo_xi": geo_xi,
        "geo_mu": geo_mu,
        "geo_mode": geo_mode,
    })
    cosmo.compute()
    return cosmo, A_s


def fs8_class(cosmo, z):
    return float(cosmo.scale_independent_growth_factor_f(z) * cosmo.sigma8() * cosmo.scale_independent_growth_factor(z))


def chi2_fs8(cosmo):
    pred = np.array([fs8_class(cosmo, z) for z in z_fs8])
    return float(np.sum(((fs8_obs - pred) / fs8_err) ** 2))


def S8_val(cosmo, omega_m):
    return float(cosmo.sigma8() * np.sqrt(omega_m / 0.3))


def chi2_s8(cosmo, omega_m, s8_obs, s8_err):
    return float(((S8_val(cosmo, omega_m) - s8_obs) / s8_err) ** 2)


def chi2_rd(cosmo):
    try:
        rd = float(cosmo.rs_drag())
    except Exception:
        rd = RD_OBS
    return float(((rd - RD_OBS) / RD_ERR) ** 2), rd


def chi2_ombh2(h):
    return float(((OMEGA_B * h**2 - OMBH2_OBS) / OMBH2_ERR) ** 2)


def chi2_ommh2(h, omega_m):
    return float(((omega_m * h**2 - OMMH2_OBS) / OMMH2_ERR) ** 2)


def chi2_As(logA):
    return float(((logA - LOGA_OBS) / LOGA_ERR) ** 2)


def chi2_sn(cosmo):
    dl = np.array([cosmo.luminosity_distance(float(z)) for z in z_sn])
    mu_th = 5.0 * np.log10(dl) + 25.0
    diff = m_sn - mu_th

    Cinv_d = cho_solve(cov_sn_cho, diff, check_finite=False)
    Cinv_1 = cho_solve(cov_sn_cho, ones_sn, check_finite=False)

    a = float(diff @ Cinv_d)
    b = float(diff @ Cinv_1)
    cval = float(ones_sn @ Cinv_1)

    return a - b * b / cval


def bao_predict(cosmo, z, obs, rd):
    DM = (1.0 + z) * cosmo.angular_distance(z)
    DH = 1.0 / cosmo.Hubble(z)
    DV = (z * DM * DM * DH) ** (1.0 / 3.0)

    if obs == "DM": return DM / rd
    if obs == "DH": return DH / rd
    if obs == "DV": return DV / rd
    raise ValueError(obs)


def chi2_bao(cosmo):
    try:
        rd = float(cosmo.rs_drag())
    except Exception:
        rd = RD_OBS

    total = 0.0
    for e in BAO:
        pred = np.array([bao_predict(cosmo, e["z"], ob, rd) for ob in e["obs"]])
        diff = e["mean"] - pred
        inv = np.linalg.pinv(e["cov"])
        total += float(diff.T @ inv @ diff)
    return total


def xi_equiv_from_mu(mu_eff, omega_m):
    od = omega_m - OMEGA_B
    if od <= 0:
        return np.nan
    return (mu_eff * omega_m - OMEGA_B) / od


def fit_model(model, test_config):
    s8_obs = test_config["s8_obs"]
    s8_err = test_config["s8_err"]

    flags = {k: test_config[k] for k in ["use_bao", "use_sn", "use_fs8", "use_s8", "use_as", "use_cmb_priors"]}

    if model == "LCDM":
        bounds, k = [(0.62, 0.75), (0.22, 0.42), (2.8, 3.3)], 3
    elif model == "GEO_CLASS":
        bounds, k = [(0.62, 0.75), (0.22, 0.42), (2.8, 3.3), (0.0, 1.0)], 4
    elif model == "FREE_MU":
        bounds, k = [(0.62, 0.75), (0.22, 0.42), (2.8, 3.3), (0.0, 1.5)], 4
    else:
        raise ValueError(model)

    def unpack(x):
        if model == "LCDM":
            h, om, logA = x
            param = 1.0
        else:
            h, om, logA, param = x
        return h, om, logA, param

    def obj(x):
        h, om, logA, param = unpack(x)
        if om <= OMEGA_B:
            return 1e30

        try:
            cosmo, _ = run_class(h, om, logA, param, model)
            total = 0.0

            if flags["use_bao"]: total += chi2_bao(cosmo)
            if flags["use_sn"]: total += chi2_sn(cosmo)
            if flags["use_fs8"]: total += chi2_fs8(cosmo)
            if flags["use_s8"]: total += chi2_s8(cosmo, om, s8_obs, s8_err)
            if flags["use_as"]: total += chi2_As(logA)

            if flags["use_cmb_priors"]:
                chi_rd_val, _ = chi2_rd(cosmo)
                total += chi_rd_val + chi2_ombh2(h) + chi2_ommh2(h, om)

            cosmo.struct_cleanup()
            cosmo.empty()
            return float(total)
        except Exception:
            return 1e30

    res = differential_evolution(
        obj, bounds, seed=123, polish=True, tol=1e-4,
        maxiter=test_config["maxiter"], popsize=test_config["popsize"]
    )

    h, om, logA, param = unpack(res.x)
    cosmo, A_s = run_class(h, om, logA, param, model)

    parts = {
        "BAO": chi2_bao(cosmo) if flags["use_bao"] else 0.0,
        "SN": chi2_sn(cosmo) if flags["use_sn"] else 0.0,
        "fs8": chi2_fs8(cosmo) if flags["use_fs8"] else 0.0,
        "S8": chi2_s8(cosmo, om, s8_obs, s8_err) if flags["use_s8"] else 0.0,
        "As": chi2_As(logA) if flags["use_as"] else 0.0,
    }

    chi_rd_val, rd = chi2_rd(cosmo)
    parts["rd"] = chi_rd_val if flags["use_cmb_priors"] else 0.0
    parts["ombh2"] = chi2_ombh2(h) if flags["use_cmb_priors"] else 0.0
    parts["ommh2"] = chi2_ommh2(h, om) if flags["use_cmb_priors"] else 0.0

    chi_total = sum(parts.values())

    if model == "LCDM":
        geo_xi, geo_mu, mu_eff, xi_equiv = 1.0, 1.0, 1.0, 1.0
    elif model == "GEO_CLASS":
        geo_xi = param
        geo_mu = 1.0
        mu_eff = (OMEGA_B + geo_xi * (om - OMEGA_B)) / om
        xi_equiv = geo_xi
    else:
        geo_xi = 1.0
        geo_mu = param
        mu_eff = geo_mu
        xi_equiv = xi_equiv_from_mu(mu_eff, om)

    n = 0
    if flags["use_bao"]: n += sum(len(e["mean"]) for e in BAO)
    if flags["use_sn"]: n += len(z_sn)
    if flags["use_fs8"]: n += len(FS8_DATA)
    if flags["use_s8"]: n += 1
    if flags["use_as"]: n += 1
    if flags["use_cmb_priors"]: n += 3

    AIC = chi_total + 2 * k
    BIC = chi_total + k * np.log(max(n, 2))

    result = {
        "model": model,
        "H0": float(100 * h),
        "h": float(h),
        "Omega_m": float(om),
        "Omega_b": float(OMEGA_B),
        "Omega_d": float(om - OMEGA_B),
        "logA": float(logA),
        "A_s": float(A_s),
        "sigma8": float(cosmo.sigma8()),
        "S8": S8_val(cosmo, om),
        "rd": float(rd),
        "geo_xi": float(geo_xi),
        "geo_mu": float(geo_mu),
        "f_out": float(1.0 - geo_xi) if model == "GEO_CLASS" else 0.0,
        "mu_eff": float(mu_eff),
        "xi_equiv": float(xi_equiv),
        "Omega_growth": float(mu_eff * om),
        "chi_total": float(chi_total),
        "AIC": float(AIC),
        "BIC": float(BIC),
        **{f"chi_{k2}": float(v) for k2, v in parts.items()},
    }

    cosmo.struct_cleanup()
    cosmo.empty()
    return result


TESTS = [
    {
        "name": "01_growth_S8",
        "use_bao": False, "use_sn": False, "use_fs8": True, "use_s8": True,
        "use_as": True, "use_cmb_priors": True,
        "s8_obs": S8_BASE_OBS, "s8_err": S8_BASE_ERR,
        "maxiter": 45, "popsize": 8,
    },
    {
        "name": "02_BAO_SN_full",
        "use_bao": True, "use_sn": True, "use_fs8": True, "use_s8": True,
        "use_as": True, "use_cmb_priors": True,
        "s8_obs": S8_BASE_OBS, "s8_err": S8_BASE_ERR,
        "maxiter": 35, "popsize": 7,
    },
]

for wl_name, (s8obs, s8err) in WL_PRIORS.items():
    TESTS.append({
        "name": f"03_weak_lensing_prior_{wl_name}",
        "use_bao": True, "use_sn": True, "use_fs8": True, "use_s8": True,
        "use_as": True, "use_cmb_priors": True,
        "s8_obs": s8obs, "s8_err": s8err,
        "maxiter": 30, "popsize": 7,
    })


def save_outputs(summary):
    df = pd.DataFrame(summary)
    csv_path = CSV_DIR / "geo_01_master_test_summary.csv"
    df.to_csv(csv_path, index=False)

    tests = df["test"].unique()
    pivot = df.pivot(index="test", columns="model", values="BIC")
    delta_geo = pivot["GEO_CLASS"] - pivot["LCDM"]
    delta_mu = pivot["FREE_MU"] - pivot["LCDM"]

    plt.figure(figsize=(11, 6))
    x = np.arange(len(tests))
    plt.bar(x - 0.2, delta_geo.loc[tests], width=0.4, label="GEO_CLASS - LCDM")
    plt.bar(x + 0.2, delta_mu.loc[tests], width=0.4, label="FREE_MU - LCDM")
    plt.axhline(0)
    plt.xticks(x, tests, rotation=45, ha="right")
    plt.ylabel("Delta BIC")
    plt.title("GEO 01 — Master Test: Delta BIC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_delta_bic.png", dpi=200)
    plt.close()

    geo = df[df["model"] == "GEO_CLASS"].copy()

    plt.figure(figsize=(11, 6))
    plt.plot(geo["test"], geo["S8"], "o-")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("S8")
    plt.title("GEO 01 — S8 GEO_CLASS")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_s8_geo_class.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.bar(geo["test"], geo["geo_xi"])
    plt.axhline(np.sqrt(3 / 5), linestyle="--", label="sqrt(3/5)")
    plt.axhline(3 / 4, linestyle="--", label="3/4")
    plt.axhline(np.pi / 4, linestyle="--", label="pi/4")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("geo_xi")
    plt.title("GEO 01 — Active geometric coupling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_geo_xi.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.bar(geo["test"], geo["f_out"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("f_out")
    plt.title("GEO 01 — Complementary fraction")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_f_out.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(geo["test"], geo["Omega_growth"], "o-")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Omega_growth")
    plt.title("GEO 01 — Effective growth source")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_omega_growth.png", dpi=200)
    plt.close()

    log_path = LOG_DIR / "geo_01_master_test.log"
    log_path.write_text(
        "GEO 01 Master Test completed.\n"
        f"CSV: {csv_path}\n"
        f"Plots: {PLOT_DIR}\n",
        encoding="utf-8",
    )


def main():
    print("\n================================================")
    print("GEO 01 — MASTER TEST")
    print("LCDM vs GEO_CLASS vs FREE_MU")
    print("================================================")

    summary = []

    for test in TESTS:
        print("\n\n################################################")
        print("TEST:", test["name"])
        print("################################################")

        results = [
            fit_model("LCDM", test),
            fit_model("GEO_CLASS", test),
            fit_model("FREE_MU", test),
        ]

        lcdm, geo, mu = results
        best = min(results, key=lambda x: x["BIC"])

        for r in results:
            row = {"test": test["name"], **r}
            summary.append(row)

            print("\nModelo:", r["model"])
            for key in ["H0", "Omega_m", "logA", "sigma8", "S8", "rd", "geo_xi", "f_out", "geo_mu", "mu_eff", "xi_equiv", "Omega_growth", "chi_total", "AIC", "BIC"]:
                print(f"{key:14s} =", r[key])

        print("\nRESUMEN TEST:", test["name"])
        print("Mejor BIC =", best["model"])
        print("Delta BIC GEO - LCDM =", geo["BIC"] - lcdm["BIC"])
        print("Delta BIC MU  - LCDM =", mu["BIC"] - lcdm["BIC"])
        print("Delta BIC MU  - GEO  =", mu["BIC"] - geo["BIC"])
        print("GEO geo_xi =", geo["geo_xi"])
        print("GEO f_out  =", geo["f_out"])
        print("GEO mu_eff =", geo["mu_eff"])
        print("MU xi_equiv =", mu["xi_equiv"])
        print("MU físico GEO? =", 0.0 <= mu["xi_equiv"] <= 1.0)

    save_outputs(summary)

    print("\nOutputs:")
    print(" - resultados/csv/geo_01_master_test_summary.csv")
    print(" - docs/plots/geo_01_master_test/")
    print(" - resultados/logs/geo_01_master_test.log")


if __name__ == "__main__":
    main()
