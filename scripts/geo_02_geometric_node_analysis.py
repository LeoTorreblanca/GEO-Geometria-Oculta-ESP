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
PLOT_DIR = ROOT / "docs" / "plots" / "geo_02_geometric_node_analysis"

for d in [CSV_DIR, LOG_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INPUT_CSV = CSV_DIR / "geo_01_master_test_summary.csv"

CANDIDATES = {
    "sqrt(3/5)": math.sqrt(3 / 5),
    "pi/4": math.pi / 4,
    "3/4": 3 / 4,
    "cos(30deg)": math.cos(math.radians(30)),
    "sqrt(2/3)": math.sqrt(2 / 3),
    "1/sqrt(2)": 1 / math.sqrt(2),
    "phi_inv": (math.sqrt(5) - 1) / 2,
    "2/pi": 2 / math.pi,
    "sqrt(pi)/2": math.sqrt(math.pi) / 2,
}


def score_candidate(values, candidate):
    diffs = values - candidate
    abs_diffs = np.abs(diffs)
    rel_diffs = abs_diffs / candidate
    rmse = math.sqrt(np.mean(diffs**2))
    mae = float(np.mean(abs_diffs))
    maxerr = float(np.max(abs_diffs))
    rel_mean = float(np.mean(rel_diffs))
    return rmse, mae, maxerr, rel_mean


def angle_from_fraction(fc):
    return 360.0 * fc


def projection_angle(fc, ideal):
    ratio = fc / ideal
    if ratio < -1 or ratio > 1:
        return np.nan
    return math.degrees(math.acos(ratio))


def main():
    print("\n================================================")
    print("GEO 02 — GEOMETRIC NODE ANALYSIS")
    print("================================================")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    geo = df[df["model"] == "GEO_CLASS"].copy()

    if geo.empty:
        raise ValueError("No GEO_CLASS rows found in geo_01_master_test_summary.csv")

    labels = geo["test"].tolist()
    fc = geo["geo_xi"].to_numpy(dtype=float)
    fout = 1.0 - fc
    eta = fc**2

    rows = []

    print("\nValores GEO_CLASS observados:")
    for lab, val in zip(labels, fc):
        print(
            f"{lab:35s} "
            f"geo_xi={val:.15f}  "
            f"f_out={1-val:.15f}  "
            f"eta={val**2:.15f}  "
            f"angle={angle_from_fraction(val):.3f}°"
        )

    print("\nResumen estadístico:")
    print("N                  =", len(fc))
    print("geo_xi mean        =", np.mean(fc))
    print("geo_xi median      =", np.median(fc))
    print("geo_xi std         =", np.std(fc, ddof=1) if len(fc) > 1 else 0.0)
    print("geo_xi min         =", np.min(fc))
    print("geo_xi max         =", np.max(fc))
    print("f_out mean         =", np.mean(fout))
    print("activo/fuera mean  =", np.mean(fc / fout))

    for name, cand in CANDIDATES.items():
        rmse, mae, maxerr, rel_mean = score_candidate(fc, cand)
        rows.append({
            "candidate": name,
            "value": cand,
            "RMSE": rmse,
            "MAE": mae,
            "max_err": maxerr,
            "rel_mean": rel_mean,
        })

    ranking = pd.DataFrame(rows).sort_values("RMSE")
    ranking_path = CSV_DIR / "geo_02_geometric_node_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    detail_rows = []
    for lab, val in zip(labels, fc):
        detail = {
            "test": lab,
            "geo_xi": val,
            "f_out": 1.0 - val,
            "eta": val**2,
            "theta_active_deg": angle_from_fraction(val),
            "theta_out_deg": angle_from_fraction(1.0 - val),
            "r_area": math.sqrt(val),
            "r_volume": val ** (1 / 3),
        }
        for name, cand in CANDIDATES.items():
            detail[f"diff_{name}"] = val - cand
            detail[f"abs_diff_{name}"] = abs(val - cand)
        detail["projection_angle_pi4"] = projection_angle(val, math.pi / 4)
        detail["projection_angle_sqrt35"] = projection_angle(val, math.sqrt(3 / 5))
        detail_rows.append(detail)

    detail_df = pd.DataFrame(detail_rows)
    detail_path = CSV_DIR / "geo_02_geometric_node_details.csv"
    detail_df.to_csv(detail_path, index=False)

    print("\nRanking por RMSE:")
    print(ranking.to_string(index=False))

    x = np.arange(len(fc))

    candidates_main = {
        "3/4": 3 / 4,
        "sqrt(3/5)": math.sqrt(3 / 5),
        "pi/4": math.pi / 4,
    }

    plt.figure(figsize=(11, 6))
    plt.plot(labels, fc, "o-", linewidth=3, markersize=8, label="geo_xi observado")
    for name, val in candidates_main.items():
        plt.axhline(val, linestyle="--", label=f"{name} = {val:.6f}")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("geo_xi")
    plt.title("GEO 02 — geo_xi observado vs candidatos geométricos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_geo_xi_vs_candidates.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    width = 0.25
    for i, (name, val) in enumerate(candidates_main.items()):
        plt.bar(x + (i - 1) * width, np.abs(fc - val), width=width, label=name)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("|geo_xi - candidato|")
    plt.title("GEO 02 — Distancia absoluta a candidatos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_distance_candidates.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(fc, fout, s=100)
    for i, lab in enumerate(labels):
        plt.annotate(lab, (fc[i], fout[i]), fontsize=8)
    plt.xlabel("geo_xi activo")
    plt.ylabel("f_out complementario")
    plt.title("GEO 02 — Dualidad activa/complementaria")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_geo_xi_vs_f_out.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.hist(fc, bins=min(5, len(fc)), edgecolor="black")
    for name, val in candidates_main.items():
        plt.axvline(val, linestyle="--", label=name)
    plt.xlabel("geo_xi")
    plt.ylabel("frecuencia")
    plt.title("GEO 02 — Distribución geométrica de geo_xi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_hist_geo_xi.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    names = list(candidates_main.keys())
    rmse_vals = [math.sqrt(np.mean((fc - candidates_main[n]) ** 2)) for n in names]
    plt.bar(names, rmse_vals)
    plt.ylabel("RMSE")
    plt.title("GEO 02 — Ranking geométrico principal")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_rmse_candidates.png", dpi=200)
    plt.close()

    best = ranking.iloc[0]
    second = ranking.iloc[1]

    log_path = LOG_DIR / "geo_02_geometric_node_analysis.log"
    log_path.write_text(
        "GEO 02 Geometric Node Analysis completed.\n"
        f"Input CSV: {INPUT_CSV}\n"
        f"Best candidate: {best['candidate']} = {best['value']}\n"
        f"Second candidate: {second['candidate']} = {second['value']}\n"
        f"Ranking CSV: {ranking_path}\n"
        f"Details CSV: {detail_path}\n"
        f"Plots: {PLOT_DIR}\n",
        encoding="utf-8",
    )

    print("\nDiagnóstico:")
    print("Mejor candidato por RMSE:", best["candidate"], "=", best["value"])
    print("Segundo candidato:", second["candidate"], "=", second["value"])
    print("RMSE mejor:", best["RMSE"])
    print("RMSE segundo:", second["RMSE"])

    print("\nOutputs:")
    print(" - resultados/csv/geo_02_geometric_node_ranking.csv")
    print(" - resultados/csv/geo_02_geometric_node_details.csv")
    print(" - docs/plots/geo_02_geometric_node_analysis/")
    print(" - resultados/logs/geo_02_geometric_node_analysis.log")


if __name__ == "__main__":
    main()
