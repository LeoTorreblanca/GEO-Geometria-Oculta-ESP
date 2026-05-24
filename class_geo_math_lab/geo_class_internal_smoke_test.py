"""
GEO_CLASS — Internal Smoke Test

Purpose:
Validate that the modified CLASS engine accepts the GEO_CLASS parameter `geo_xi`
and that changing it produces a measurable response in sigma8 and P(k).

This is not a cosmological fit.
It is only a minimal technical test of the GEO_CLASS coupling channel.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from classy import Class


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "outputs" / "csv"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
LOGS_DIR = BASE_DIR / "outputs" / "logs"

for folder in (CSV_DIR, PLOTS_DIR, LOGS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


BASE_PARAMS = {
    "h": 0.67,
    "omega_b": 0.022,
    "omega_cdm": 0.12,
    "A_s": 2.1e-9,
    "n_s": 0.965,
    "tau_reio": 0.054,
    "output": "mPk",
    "P_k_max_1/Mpc": 2.0,
    "z_max_pk": 2.0,
    "geo_mu": 1.0,
    "geo_mode": 0,
}


def run_case(geo_xi: float) -> dict:
    params = dict(BASE_PARAMS)
    params["geo_xi"] = geo_xi

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    result = {
        "geo_xi": geo_xi,
        "sigma8": float(cosmo.sigma8()),
        "pk_k01_z0": float(cosmo.pk(0.1, 0.0)),
        "pk_k01_z1": float(cosmo.pk(0.1, 1.0)),
    }

    cosmo.struct_cleanup()
    cosmo.empty()
    return result


def save_outputs(df: pd.DataFrame) -> None:
    csv_path = CSV_DIR / "geo_class_internal_smoke_test.csv"
    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(df["geo_xi"], df["sigma8"], marker="o")
    plt.xlabel("geo_xi")
    plt.ylabel("sigma8")
    plt.title("GEO_CLASS smoke test — sigma8 response")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geo_class_internal_smoke_test_sigma8.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(df["geo_xi"], df["pk_k01_z0"], marker="o", label="P(k=0.1,z=0)")
    plt.plot(df["geo_xi"], df["pk_k01_z1"], marker="o", label="P(k=0.1,z=1)")
    plt.xlabel("geo_xi")
    plt.ylabel("P(k)")
    plt.title("GEO_CLASS smoke test — matter power response")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geo_class_internal_smoke_test_pk.png", dpi=160)
    plt.close()

    log_path = LOGS_DIR / "geo_class_internal_smoke_test.log"
    log_path.write_text(
        "GEO_CLASS internal smoke test completed successfully.\n"
        f"CSV: {csv_path}\n"
        f"Plots: {PLOTS_DIR}\n",
        encoding="utf-8",
    )


def main() -> None:
    print("\n=== GEO_CLASS INTERNAL SMOKE TEST ===")
    print("Testing response of CLASS to the GEO_CLASS coupling parameter: geo_xi")
    print("Mode: geo_mode = 0 | geo_mu = 1.0")
    print("-" * 78)

    df = pd.DataFrame([run_case(x) for x in (1.0, 0.07, 0.0)])

    print(df.to_string(index=False))
    print("-" * 78)

    save_outputs(df)

    print("STATUS: OK if sigma8 / P(k) change with geo_xi.")
    print("NOTE: This is a technical engine test, not a final physical validation.")
    print("Saved outputs:")
    print(" - outputs/csv/geo_class_internal_smoke_test.csv")
    print(" - outputs/plots/geo_class_internal_smoke_test_sigma8.png")
    print(" - outputs/plots/geo_class_internal_smoke_test_pk.png\n")


if __name__ == "__main__":
    main()
