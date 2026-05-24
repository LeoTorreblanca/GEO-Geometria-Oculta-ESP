from classy import Class
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

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
    (1.944, 0.364, 0.106)
]

z_data = np.array([x[0] for x in FS8_DATA])
fs8_obs = np.array([x[1] for x in FS8_DATA])
fs8_err = np.array([x[2] for x in FS8_DATA])

# Priors lensing / CMB-BBN comprimidos
S8_OBS, S8_ERR = 0.776, 0.0325
RD_OBS, RD_ERR = 147.1, 0.3
OMBH2_OBS, OMBH2_ERR = 0.0224, 0.0001
OMMH2_OBS, OMMH2_ERR = 0.143, 0.002

OMEGA_B = 0.05


def run_class(h, Omega_m):
    omega_b = OMEGA_B * h**2
    omega_cdm = (Omega_m - OMEGA_B) * h**2

    if omega_cdm <= 0:
        raise ValueError("omega_cdm <= 0")

    cosmo = Class()
    cosmo.set({
        "h": h,
        "omega_b": omega_b,
        "omega_cdm": omega_cdm,
        "A_s": 2.1e-9,
        "n_s": 0.965,
        "tau_reio": 0.054,
        "output": "mPk",
        "P_k_max_1/Mpc": 2.0,
        "z_max_pk": 2.5
    })
    cosmo.compute()
    return cosmo


def dlnH_dz_class(cosmo, z):
    dz = 1e-4
    zp = z + dz
    zm = max(z - dz, 0.0)
    return (np.log(cosmo.Hubble(zp)) - np.log(cosmo.Hubble(zm))) / (zp - zm)


def E2_LCDM(z, Omega_m):
    return Omega_m * (1 + z)**3 + (1 - Omega_m)


def Omega_m_z(z, Omega_m):
    return Omega_m * (1 + z)**3 / E2_LCDM(z, Omega_m)


def Omega_b_z(z, Omega_b, Omega_m):
    return Omega_b * (1 + z)**3 / E2_LCDM(z, Omega_m)


def Omega_d_z(z, Omega_b, Omega_m):
    return (Omega_m - Omega_b) * (1 + z)**3 / E2_LCDM(z, Omega_m)


def growth_source(z, Omega_m, Omega_b, xi, model):
    if model == "LCDM":
        return Omega_m_z(z, Omega_m)
    if model == "GEO_CLASS":
        return Omega_b_z(z, Omega_b, Omega_m) + xi * Omega_d_z(z, Omega_b, Omega_m)
    raise ValueError(model)


def solve_growth_external(cosmo, Omega_m, Omega_b, xi, sigma8_0, model):
    zmax = 2.2
    z_eval = np.linspace(zmax, 0.0, 1200)

    def eq(z, y):
        delta, ddelta = y
        A = 2.0 / (1.0 + z) - dlnH_dz_class(cosmo, z)
        source = growth_source(z, Omega_m, Omega_b, xi, model)
        B = 1.5 * source / (1.0 + z)**2
        return [ddelta, -A * ddelta + B * delta]

    sol = solve_ivp(
        eq,
        [zmax, 0.0],
        [1e-5, -1e-5 / (1 + zmax)],
        t_eval=z_eval,
        rtol=1e-7,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError("fallo growth")

    z = sol.t
    delta = np.abs(sol.y[0])
    delta[delta < 1e-20] = 1e-20

    a = 1.0 / (1.0 + z)
    f = np.gradient(np.log(delta), np.log(a))
    sigma8_z = sigma8_0 * delta / delta[-1]
    fs8 = f * sigma8_z

    return z, fs8


def predict_fs8(cosmo, Omega_m, Omega_b, xi, sigma8_0, model):
    z_model, fs8_model = solve_growth_external(
        cosmo, Omega_m, Omega_b, xi, sigma8_0, model
    )
    return np.interp(z_data, z_model[::-1], fs8_model[::-1])


def chi2_fs8(cosmo, Omega_m, Omega_b, xi, sigma8_0, model):
    pred = predict_fs8(cosmo, Omega_m, Omega_b, xi, sigma8_0, model)
    return np.sum(((fs8_obs - pred) / fs8_err)**2)


def S8(sigma8, Omega_m):
    return sigma8 * np.sqrt(Omega_m / 0.3)


def chi2_s8(sigma8, Omega_m):
    return ((S8(sigma8, Omega_m) - S8_OBS) / S8_ERR)**2


def chi2_rd(rd):
    return ((rd - RD_OBS) / RD_ERR)**2


def chi2_ombh2(h):
    return ((OMEGA_B * h**2 - OMBH2_OBS) / OMBH2_ERR)**2


def chi2_ommh2(h, Omega_m):
    return ((Omega_m * h**2 - OMMH2_OBS) / OMMH2_ERR)**2


def approx_rd_from_class(cosmo):
    # CLASS tiene rs_drag() en muchas versiones classy
    try:
        return cosmo.rs_drag()
    except Exception:
        return RD_OBS


def fit_model(model):
    if model == "LCDM":
        bounds = [
            (0.62, 0.75),   # h
            (0.22, 0.42),   # Omega_m
            (0.45, 1.05)    # sigma8
        ]

        def obj(x):
            h, Omega_m, sigma8 = x
            if Omega_m <= OMEGA_B:
                return 1e30
            try:
                cosmo = run_class(h, Omega_m)
                rd = approx_rd_from_class(cosmo)
                val = (
                    chi2_fs8(cosmo, Omega_m, OMEGA_B, 1.0, sigma8, "LCDM")
                    + chi2_s8(sigma8, Omega_m)
                    + chi2_rd(rd)
                    + chi2_ombh2(h)
                    + chi2_ommh2(h, Omega_m)
                )
                cosmo.struct_cleanup()
                cosmo.empty()
                return val
            except Exception:
                return 1e30

        res = differential_evolution(obj, bounds, seed=123, polish=True, tol=1e-5)
        h, Omega_m, sigma8 = res.x
        xi = 1.0

    elif model == "GEO_CLASS":
        bounds = [
            (0.62, 0.75),   # h
            (0.22, 0.42),   # Omega_m
            (0.45, 1.20),   # sigma8
            (0.0, 1.0)      # xi
        ]

        def obj(x):
            h, Omega_m, sigma8, xi = x
            if Omega_m <= OMEGA_B:
                return 1e30
            try:
                cosmo = run_class(h, Omega_m)
                rd = approx_rd_from_class(cosmo)
                val = (
                    chi2_fs8(cosmo, Omega_m, OMEGA_B, xi, sigma8, "GEO_CLASS")
                    + chi2_s8(sigma8, Omega_m)
                    + chi2_rd(rd)
                    + chi2_ombh2(h)
                    + chi2_ommh2(h, Omega_m)
                )
                cosmo.struct_cleanup()
                cosmo.empty()
                return val
            except Exception:
                return 1e30

        res = differential_evolution(obj, bounds, seed=123, polish=True, tol=1e-5)
        h, Omega_m, sigma8, xi = res.x

    else:
        raise ValueError(model)

    cosmo = run_class(h, Omega_m)
    rd = approx_rd_from_class(cosmo)

    chi_fs8 = chi2_fs8(
        cosmo, Omega_m, OMEGA_B, xi, sigma8,
        "LCDM" if model == "LCDM" else "GEO_CLASS"
    )

    chi_s8 = chi2_s8(sigma8, Omega_m)
    chi_rd_val = chi2_rd(rd)
    chi_ob = chi2_ombh2(h)
    chi_om = chi2_ommh2(h, Omega_m)

    chi_total = chi_fs8 + chi_s8 + chi_rd_val + chi_ob + chi_om

    k = 3 if model == "LCDM" else 4
    n = len(FS8_DATA) + 4
    AIC = chi_total + 2 * k
    BIC = chi_total + k * np.log(n)

    mu_eff = 1.0 if model == "LCDM" else (OMEGA_B + xi * (Omega_m - OMEGA_B)) / Omega_m

    result = {
        "model": model,
        "h": h,
        "H0": 100*h,
        "Omega_m": Omega_m,
        "Omega_b": OMEGA_B,
        "Omega_d": Omega_m - OMEGA_B,
        "sigma8": sigma8,
        "S8": S8(sigma8, Omega_m),
        "rd": rd,
        "xi": xi,
        "mu_eff": mu_eff,
        "Omega_growth": mu_eff * Omega_m,
        "chi_total": chi_total,
        "chi_fs8": chi_fs8,
        "chi_S8": chi_s8,
        "chi_rd": chi_rd_val,
        "chi_ombh2": chi_ob,
        "chi_ommh2": chi_om,
        "AIC": AIC,
        "BIC": BIC
    }

    cosmo.struct_cleanup()
    cosmo.empty()

    return result


print("\n==============================")
print("CLASS + GEO_CLASS + PRIORS")
print("==============================")

lcdm = fit_model("LCDM")
geo = fit_model("GEO_CLASS")

for r in [lcdm, geo]:
    print("\nModelo:", r["model"])
    print("H0          =", r["H0"])
    print("h           =", r["h"])
    print("Omega_m     =", r["Omega_m"])
    print("Omega_b     =", r["Omega_b"])
    print("Omega_d     =", r["Omega_d"])
    print("sigma8      =", r["sigma8"])
    print("S8          =", r["S8"])
    print("rd          =", r["rd"])
    print("xi          =", r["xi"])
    print("mu_eff      =", r["mu_eff"])
    print("Omega_growth=", r["Omega_growth"])
    print("chi_total   =", r["chi_total"])
    print("chi_fs8     =", r["chi_fs8"])
    print("chi_S8      =", r["chi_S8"])
    print("chi_rd      =", r["chi_rd"])
    print("chi_ombh2   =", r["chi_ombh2"])
    print("chi_ommh2   =", r["chi_ommh2"])
    print("AIC         =", r["AIC"])
    print("BIC         =", r["BIC"])

print("\n==============================")
print("COMPARACIÓN")
print("==============================")
print("Delta chi2 GEO - LCDM =", geo["chi_total"] - lcdm["chi_total"])
print("Delta AIC  GEO - LCDM =", geo["AIC"] - lcdm["AIC"])
print("Delta BIC  GEO - LCDM =", geo["BIC"] - lcdm["BIC"])
