from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from geo_01_master_test import (
    TESTS,
    OMEGA_B,
    fit_model,
    run_class,
    fs8_class,
    S8_val,
)

CSV_DIR = ROOT / "resultados" / "csv"
LOG_DIR = ROOT / "resultados" / "logs"
PLOT_DIR = ROOT / "docs" / "plots" / "geo_04_prediction_law_base"

for d in [CSV_DIR, LOG_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FC = math.sqrt(3.0 / 5.0)
ETA = FC ** 2
ALPHA = 1.0 / 3.0

ZS = [0.1, 0.3, 0.5, 0.7, 1.0]


def main():
    print("\n================================================")
    print("GEO 04 — BASE PREDICTION LAW")
    print("fc=sqrt(3/5), eta=3/5, R=mu_eff^(1/3)")
    print("================================================")

    summary_rows = []
    redshift_rows = []

    for test in TESTS:
        print("\n\n################################################")
        print("TEST:", test["name"])
        print("################################################")

        lcdm_fit = fit_model("LCDM", test)

        h = lcdm_fit["h"]
        om = lcdm_fit["Omega_m"]
        logA = lcdm_fit["logA"]

        od = om - OMEGA_B

        mu_eff = (OMEGA_B + FC * od) / om
        R_pred = mu_eff ** ALPHA

        lcdm_cosmo, _ = run_class(h, om, logA, 1.0, "LCDM")
        geo_cosmo, _ = run_class(h, om, logA, FC, "GEO_CLASS")

        s8_lcdm = S8_val(lcdm_cosmo, om)
        s8_geo_class = S8_val(geo_cosmo, om)
        s8_geo_pred = s8_lcdm * R_pred

        s8_err = abs(s8_geo_class - s8_geo_pred) / max(abs(s8_geo_class), 1e-12)

        print("\nGEOMETRY")
        print("fc                       =", FC)
        print("f_out                    =", 1.0 - FC)
        print("eta=fc^2                 =", ETA)
        print("Omega_m                  =", om)
        print("Omega_b                  =", OMEGA_B)
        print("Omega_d                  =", od)
        print("mu_eff                   =", mu_eff)
        print("R_pred=mu_eff^(1/3)      =", R_pred)

        print("\nS8")
        print("S8 LCDM                  =", s8_lcdm)
        print("S8 GEO_CLASS             =", s8_geo_class)
        print("S8 GEO predicted         =", s8_geo_pred)
        print("ratio CLASS              =", s8_geo_class / s8_lcdm)
        print("ratio predicted          =", R_pred)
        print("relative error           =", s8_err)

        errors = []

        print("\nfσ8 by redshift")
        print("z      LCDM       GEO_CLASS  GEO_pred   ratio_CLASS  ratio_pred   err%")

        for z in ZS:
            fs8_lcdm = fs8_class(lcdm_cosmo, z)
            fs8_geo_class = fs8_class(geo_cosmo, z)
            fs8_geo_pred = fs8_lcdm * R_pred

            ratio_class = fs8_geo_class / fs8_lcdm
            rel_err = abs(fs8_geo_class - fs8_geo_pred) / max(abs(fs8_geo_class), 1e-12)
            errors.append(rel_err)

            redshift_rows.append({
                "test": test["name"],
                "z": z,
                "fs8_LCDM": fs8_lcdm,
                "fs8_GEO_CLASS": fs8_geo_class,
                "fs8_GEO_pred": fs8_geo_pred,
                "ratio_CLASS": ratio_class,
                "ratio_pred": R_pred,
                "relative_error": rel_err,
            })

            print(
                f"{z:3.1f}  "
                f"{fs8_lcdm:.6f}  "
                f"{fs8_geo_class:.6f}  "
                f"{fs8_geo_pred:.6f}  "
                f"{ratio_class:.6f}     "
                f"{R_pred:.6f}    "
                f"{100 * rel_err:.3f}"
            )

        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))
        score = math.exp(-(mean_err + s8_err) / 0.03)

        print("\nTEST SUMMARY")
        print("mean fs8 error       =", mean_err)
        print("max fs8 error        =", max_err)
        print("S8 error             =", s8_err)
        print("prediction score     =", 100 * score)

        summary_rows.append({
            "test": test["name"],
            "fc": FC,
            "f_out": 1.0 - FC,
            "eta": ETA,
            "Omega_m": om,
            "Omega_b": OMEGA_B,
            "Omega_d": od,
            "mu_eff": mu_eff,
            "R_pred": R_pred,
            "S8_LCDM": s8_lcdm,
            "S8_GEO_CLASS": s8_geo_class,
            "S8_GEO_pred": s8_geo_pred,
            "S8_relative_error": s8_err,
            "mean_fs8_error": mean_err,
            "max_fs8_error": max_err,
            "prediction_score": score,
        })

        lcdm_cosmo.struct_cleanup()
        lcdm_cosmo.empty()
        geo_cosmo.struct_cleanup()
        geo_cosmo.empty()

    summary_df = pd.DataFrame(summary_rows)
    redshift_df = pd.DataFrame(redshift_rows)

    summary_path = CSV_DIR / "geo_04_prediction_law_base_summary.csv"
    redshift_path = CSV_DIR / "geo_04_prediction_law_base_redshift.csv"

    summary_df.to_csv(summary_path, index=False)
    redshift_df.to_csv(redshift_path, index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(summary_df["R_pred"], summary_df["S8_GEO_CLASS"] / summary_df["S8_LCDM"], s=90)
    mn = min(summary_df["R_pred"].min(), (summary_df["S8_GEO_CLASS"] / summary_df["S8_LCDM"]).min())
    mx = max(summary_df["R_pred"].max(), (summary_df["S8_GEO_CLASS"] / summary_df["S8_LCDM"]).max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", label="y=x")
    plt.xlabel("R predicted")
    plt.ylabel("R CLASS = S8_GEO_CLASS / S8_LCDM")
    plt.title("GEO 04 — Base law: R predicted vs CLASS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_R_pred_vs_CLASS.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(summary_df["test"], summary_df["S8_LCDM"], "o-", label="S8 LCDM")
    plt.plot(summary_df["test"], summary_df["S8_GEO_pred"], "o-", label="S8 GEO predicted")
    plt.plot(summary_df["test"], summary_df["S8_GEO_CLASS"], "o-", label="S8 GEO CLASS")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("S8")
    plt.title("GEO 04 — Base prediction of S8")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_S8_prediction.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.bar(summary_df["test"], 100 * summary_df["S8_relative_error"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("relative error %")
    plt.title("GEO 04 — Base law relative error")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_relative_error.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.bar(summary_df["test"], 100 * (1.0 - summary_df["R_pred"]))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("suppression %")
    plt.title("GEO 04 — Predicted geometric suppression")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_growth_suppression.png", dpi=200)
    plt.close()

    log_path = LOG_DIR / "geo_04_prediction_law_base.log"
    log_path.write_text(
        "GEO 04 Base Prediction Law completed.\n"
        f"fc = {FC}\n"
        f"eta = {ETA}\n"
        "Law: R = mu_eff^(1/3)\n"
        f"Mean prediction score = {float(summary_df['prediction_score'].mean())}\n"
        f"Summary CSV: {summary_path}\n"
        f"Redshift CSV: {redshift_path}\n"
        f"Plots: {PLOT_DIR}\n",
        encoding="utf-8",
    )

    print("\n\n================================================")
    print("GLOBAL RESULT GEO 04")
    print("================================================")
    print("score mean =", 100 * summary_df["prediction_score"].mean())
    print("score min  =", 100 * summary_df["prediction_score"].min())
    print("score max  =", 100 * summary_df["prediction_score"].max())

    print("\nOutputs:")
    print(" - resultados/csv/geo_04_prediction_law_base_summary.csv")
    print(" - resultados/csv/geo_04_prediction_law_base_redshift.csv")
    print(" - docs/plots/geo_04_prediction_law_base/")
    print(" - resultados/logs/geo_04_prediction_law_base.log")


if __name__ == "__main__":
    main()
