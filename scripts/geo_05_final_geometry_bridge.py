from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

CSV_DIR = ROOT / "resultados" / "csv"
LOG_DIR = ROOT / "resultados" / "logs"
PLOT_DIR = ROOT / "docs" / "plots" / "geo_05_final_geometry_bridge"

for d in [CSV_DIR, LOG_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INPUT_01 = CSV_DIR / "geo_01_master_test_summary.csv"
INPUT_03 = CSV_DIR / "geo_03_architecture_strong_test_summary.csv"
INPUT_04 = CSV_DIR / "geo_04_prediction_law_base_summary.csv"

FC_NODE = math.sqrt(3.0 / 5.0)
FC_BASE = 3.0 / 4.0
FC_PI4 = math.pi / 4.0


def geo_bridge_metrics(fc):
    fout = 1.0 - fc
    eta = fc**2

    active_angle = 360.0 * fc
    outside_angle = 360.0 * fout

    active_rad = 2.0 * math.pi * fc
    outside_rad = 2.0 * math.pi * fout

    tension = abs(fc - FC_NODE)
    compression = abs(fc - FC_BASE)

    active_radius = math.sqrt(fc)
    volume_radius = fc ** (1.0 / 3.0)

    return {
        "fc": fc,
        "f_out": fout,
        "eta": eta,
        "active_angle_deg": active_angle,
        "outside_angle_deg": outside_angle,
        "active_angle_rad": active_rad,
        "outside_angle_rad": outside_rad,
        "tension_to_node": tension,
        "compression_to_base": compression,
        "active_radius": active_radius,
        "volume_radius": volume_radius,
    }


def classify_geometry(fc):
    d_node = abs(fc - FC_NODE)
    d_base = abs(fc - FC_BASE)
    d_pi4 = abs(fc - FC_PI4)

    best = min([
        ("NODE_sqrt35", d_node),
        ("BASE_3_4", d_base),
        ("PI4", d_pi4),
    ], key=lambda x: x[1])

    return best[0], best[1]


def main():
    print("\n================================================")
    print("GEO 05 — FINAL GEOMETRY BRIDGE")
    print("================================================")

    if not INPUT_01.exists():
        raise FileNotFoundError(INPUT_01)

    if not INPUT_03.exists():
        raise FileNotFoundError(INPUT_03)

    if not INPUT_04.exists():
        raise FileNotFoundError(INPUT_04)

    df1 = pd.read_csv(INPUT_01)
    df3 = pd.read_csv(INPUT_03)
    df4 = pd.read_csv(INPUT_04)

    geo = df1[df1["model"] == "GEO_CLASS"].copy()

    rows = []

    print("\nObserved GEO_CLASS structures:")
    print("------------------------------------------------------------")

    for _, row in geo.iterrows():
        fc = float(row["geo_xi"])

        metrics = geo_bridge_metrics(fc)
        family, distance = classify_geometry(fc)

        out = {
            "test": row["test"],
            "family": family,
            "family_distance": distance,
            "H0": row["H0"],
            "Omega_m": row["Omega_m"],
            "sigma8": row["sigma8"],
            "S8": row["S8"],
            "mu_eff": row["mu_eff"],
            "Omega_growth": row["Omega_growth"],
            "chi_total": row["chi_total"],
            "AIC": row["AIC"],
            "BIC": row["BIC"],
            **metrics,
        }

        rows.append(out)

        print(f"\nTEST: {row['test']}")
        print("family                =", family)
        print("distance              =", distance)
        print("fc                    =", metrics["fc"])
        print("f_out                 =", metrics["f_out"])
        print("eta                   =", metrics["eta"])
        print("active_angle_deg      =", metrics["active_angle_deg"])
        print("outside_angle_deg     =", metrics["outside_angle_deg"])
        print("active_radius         =", metrics["active_radius"])
        print("volume_radius         =", metrics["volume_radius"])
        print("tension_to_node       =", metrics["tension_to_node"])
        print("compression_to_base   =", metrics["compression_to_base"])

    bridge_df = pd.DataFrame(rows)

    bridge_csv = CSV_DIR / "geo_05_final_geometry_bridge.csv"
    bridge_df.to_csv(bridge_csv, index=False)

    mean_fc = float(bridge_df["fc"].mean())
    std_fc = float(bridge_df["fc"].std())

    mean_eta = float(bridge_df["eta"].mean())
    mean_mu = float(bridge_df["mu_eff"].mean())

    mean_growth = float(bridge_df["Omega_growth"].mean())

    node_dist = float(np.mean(np.abs(bridge_df["fc"] - FC_NODE)))
    base_dist = float(np.mean(np.abs(bridge_df["fc"] - FC_BASE)))
    pi4_dist = float(np.mean(np.abs(bridge_df["fc"] - FC_PI4)))

    summary = pd.DataFrame([{
        "mean_fc": mean_fc,
        "std_fc": std_fc,
        "mean_eta": mean_eta,
        "mean_mu_eff": mean_mu,
        "mean_Omega_growth": mean_growth,
        "distance_to_node": node_dist,
        "distance_to_base": base_dist,
        "distance_to_pi4": pi4_dist,
        "best_global_geometry":
            min([
                ("NODE_sqrt35", node_dist),
                ("BASE_3_4", base_dist),
                ("PI4", pi4_dist),
            ], key=lambda x: x[1])[0],
    }])

    summary_csv = CSV_DIR / "geo_05_final_geometry_bridge_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n================================================")
    print("GLOBAL GEOMETRY")
    print("================================================")
    print("mean_fc              =", mean_fc)
    print("std_fc               =", std_fc)
    print("mean_eta             =", mean_eta)
    print("mean_mu_eff          =", mean_mu)
    print("mean_Omega_growth    =", mean_growth)
    print("distance_to_node     =", node_dist)
    print("distance_to_base     =", base_dist)
    print("distance_to_pi4      =", pi4_dist)
    print("best_global_geometry =", summary.iloc[0]["best_global_geometry"])

    plt.figure(figsize=(11, 6))
    plt.plot(bridge_df["test"], bridge_df["fc"], "o-", linewidth=3)
    plt.axhline(FC_NODE, linestyle="--", label="sqrt(3/5)")
    plt.axhline(FC_BASE, linestyle="--", label="3/4")
    plt.axhline(FC_PI4, linestyle="--", label="pi/4")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("fc")
    plt.title("GEO 05 — Final geometric bridge")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_fc_bridge.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.bar(bridge_df["test"], bridge_df["eta"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("eta = fc^2")
    plt.title("GEO 05 — Geometric efficiency")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_eta.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.bar(bridge_df["test"], bridge_df["active_angle_deg"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("degrees")
    plt.title("GEO 05 — Active angular sector")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_active_angle.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.bar(bridge_df["test"], bridge_df["mu_eff"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mu_eff")
    plt.title("GEO 05 — Effective coupling")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_mu_eff.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.bar(bridge_df["test"], bridge_df["Omega_growth"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Omega_growth")
    plt.title("GEO 05 — Effective growth source")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_growth_source.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(bridge_df["fc"], bridge_df["mu_eff"], s=100)

    for _, row in bridge_df.iterrows():
        plt.annotate(row["test"], (row["fc"], row["mu_eff"]), fontsize=8)

    plt.xlabel("fc")
    plt.ylabel("mu_eff")
    plt.title("GEO 05 — Geometry vs effective coupling")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_fc_vs_mu.png", dpi=200)
    plt.close()

    log_path = LOG_DIR / "geo_05_final_geometry_bridge.log"

    log_path.write_text(
        "GEO 05 Final Geometry Bridge completed.\n"
        f"mean_fc = {mean_fc}\n"
        f"mean_eta = {mean_eta}\n"
        f"mean_mu_eff = {mean_mu}\n"
        f"best_global_geometry = {summary.iloc[0]['best_global_geometry']}\n"
        f"Bridge CSV: {bridge_csv}\n"
        f"Summary CSV: {summary_csv}\n"
        f"Plots: {PLOT_DIR}\n",
        encoding="utf-8",
    )

    print("\nOutputs:")
    print(" - resultados/csv/geo_05_final_geometry_bridge.csv")
    print(" - resultados/csv/geo_05_final_geometry_bridge_summary.csv")
    print(" - docs/plots/geo_05_final_geometry_bridge/")
    print(" - resultados/logs/geo_05_final_geometry_bridge.log")


if __name__ == "__main__":
    main()
