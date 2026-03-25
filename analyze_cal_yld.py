from pathlib import Path
import argparse
import logging
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


LOGLEVEL = "INFO"

SCENARIO_ORDER = {"SSP126": 0, "SSP370": 1, "SSP585": 2}
WATER_LABELS = {"HRLM": "rainfed", "HILM": "irrigated"}


def parse_var_name(name: str):
    # Example: cal_yld_RES05-YCX_FP2140_GFDL-ESM4_SSP126_HILM
    m = re.match(r"^cal_yld_(RES\d{2}-[A-Z]{3})_([A-Z0-9]+)_([^_]+)_([^_]+)_([A-Z0-9]+)$", name)
    if not m:
        return None
    variable_code, period, model, scenario, water = m.groups()
    return {
        "var_name": name,
        "variable_code": variable_code,
        "period": period,
        "model": model,
        "scenario": scenario,
        "water": water,
    }


def pick_input_nc(requested: Path) -> Path:
    if requested.exists():
        return requested
    fallback = Path("cal_yld_RES05-YCX.nc")
    if fallback.exists():
        logging.warning(f"Input {requested} not found, using {fallback} instead")
        return fallback
    raise FileNotFoundError(f"Input NetCDF not found: {requested}")


def collect_entries(ds: xr.Dataset):
    entries = []
    for name in ds.data_vars:
        info = parse_var_name(name)
        if info is not None:
            entries.append(info)
    return entries


def choose_observed(entries, period: str, water: str):
    cands = [e for e in entries if e["water"] == water and e["period"] == period]
    if not cands:
        raise RuntimeError(f"No observed candidate for period={period}, water={water}")

    preferred = [e for e in cands if e["scenario"] in {"HIST", "observed", "OBSERVED"}]
    if preferred:
        cands = preferred

    model_priority = {"AGERA5": 0, "ENSEMBLE": 1}
    cands = sorted(cands, key=lambda e: (model_priority.get(e["model"], 9), e["model"]))
    return cands[0]


def ordered_future_maps(entries, water: str):
    fut = [
        e
        for e in entries
        if e["water"] == water and e["period"] == "FP2140" and e["scenario"] in SCENARIO_ORDER
    ]
    fut = sorted(
        fut,
        key=lambda e: (
            SCENARIO_ORDER[e["scenario"]],
            0 if e["model"] == "ENSEMBLE" else 1,
            e["model"],
        ),
    )
    return fut


def summarize_vector(vec: np.ndarray):
    valid = vec[np.isfinite(vec)]
    if valid.size == 0:
        return {
            "mean": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q10": np.nan,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "frac_negative": np.nan,
            "n": 0,
        }
    q = np.quantile(valid, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "mean": float(np.mean(valid)),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q10": float(q[2]),
        "q25": float(q[3]),
        "q50": float(q[4]),
        "q75": float(q[5]),
        "q90": float(q[6]),
        "q95": float(q[7]),
        "q99": float(q[8]),
        "frac_negative": float(np.mean(valid < 0)),
        "n": int(valid.size),
    }


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return np.nan
    xx = x[m].astype("float64")
    yy = y[m].astype("float64")
    x0 = xx - xx.mean()
    y0 = yy - yy.mean()
    den = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if den == 0:
        return np.nan
    return float(np.sum(x0 * y0) / den)


def rankdata_average(a: np.ndarray) -> np.ndarray:
    # Equivalent to average-tie ranking (as in Spearman with average ranks).
    s = pd.Series(a)
    return s.rank(method="average").to_numpy(dtype="float64")


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return np.nan
    xx = x[m]
    yy = y[m]
    rx = rankdata_average(xx)
    ry = rankdata_average(yy)
    x0 = rx - rx.mean()
    y0 = ry - ry.mean()
    den = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if den == 0:
        return np.nan
    return float(np.sum(x0 * y0) / den)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def plot_corr_heatmap(corr: pd.DataFrame, title: str, out_path: Path, hist_index: int):
    labels = list(corr.index)
    mat = corr.values

    fig_w = max(8, 0.45 * len(labels) + 4)
    fig_h = max(7, 0.45 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("correlation", rotation=90)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)

    # Highlight historical row/column.
    ax.axhline(hist_index - 0.5, color="black", linewidth=2)
    ax.axhline(hist_index + 0.5, color="black", linewidth=2)
    ax.axvline(hist_index - 0.5, color="black", linewidth=2)
    ax.axvline(hist_index + 0.5, color="black", linewidth=2)

    ax.get_xticklabels()[hist_index].set_fontweight("bold")
    ax.get_yticklabels()[hist_index].set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_scaled(ds: xr.Dataset, var_name: str) -> np.ndarray:
    return (np.asarray(ds[var_name].values, dtype="float32") / 730.0).astype("float32")


def build_water_analysis(ds: xr.Dataset, entries, water: str):
    hp8100 = choose_observed(entries, period="HP8100", water=water)
    hp0120 = choose_observed(entries, period="HP0120", water=water)
    logging.info(
        f"{water}: observed baseline={hp8100['var_name']}, observed recent={hp0120['var_name']}"
    )

    baseline = load_scaled(ds, hp8100["var_name"])
    hp0120_arr = load_scaled(ds, hp0120["var_name"])
    mask = baseline > 0
    if not np.any(mask):
        raise RuntimeError(f"Baseline mask has no positive cells for water={water}")

    map_specs = [
        {
            "label": "observed|observed",
            "scenario": "observed",
            "model": "observed",
            "kind": "historical",
            "future_var": None,
        }
    ]

    for e in ordered_future_maps(entries, water=water):
        map_specs.append(
            {
                "label": f"{e['scenario']}|{e['model']}",
                "scenario": e["scenario"],
                "model": e["model"],
                "kind": "future",
                "future_var": e["var_name"],
            }
        )

    vectors = {}
    summary_rows = []

    for spec in map_specs:
        if spec["kind"] == "historical":
            delta = hp0120_arr - baseline
        else:
            fut = load_scaled(ds, spec["future_var"])
            delta = fut - baseline

        vec = delta[mask].astype("float32")
        vectors[spec["label"]] = vec

        s = summarize_vector(vec)
        summary_rows.append(
            {
                "scenario": spec["scenario"],
                "model": spec["model"],
                **s,
            }
        )

    summary = pd.DataFrame(summary_rows)
    for c in [
        "mean",
        "q01",
        "q05",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "q95",
        "q99",
    ]:
        summary[c] = summary[c].round(0).astype(int)
    summary["frac_negative"] = summary["frac_negative"].round(3)
    summary = summary.drop(columns=["n"])

    labels = [s["label"] for s in map_specs]
    n = len(labels)
    pear = np.full((n, n), np.nan, dtype="float64")
    spear = np.full((n, n), np.nan, dtype="float64")

    for i in range(n):
        pear[i, i] = 1.0
        spear[i, i] = 1.0
        for j in range(i + 1, n):
            a = vectors[labels[i]]
            b = vectors[labels[j]]
            p = pearson_corr(a, b)
            s = spearman_corr(a, b)
            pear[i, j] = pear[j, i] = p
            spear[i, j] = spear[j, i] = s

    pearson = pd.DataFrame(pear, index=labels, columns=labels)
    spearman = pd.DataFrame(spear, index=labels, columns=labels)

    overlap_rows = []
    hist_vec = vectors["observed|observed"]
    hist_valid = np.isfinite(hist_vec)
    hist_thr = np.quantile(hist_vec[hist_valid], 0.10)
    hist_bottom = hist_valid & (hist_vec <= hist_thr)

    for label in labels[1:]:
        fut = vectors[label]
        fut_valid = np.isfinite(fut)
        overlap = np.nan
        if np.any(fut_valid):
            fut_thr = np.quantile(fut[fut_valid], 0.10)
            fut_bottom = fut_valid & (fut <= fut_thr)
            common = hist_bottom & fut_valid
            denom = np.sum(common)
            if denom > 0:
                overlap = float(np.sum(common & fut_bottom) / denom)

        scen, model = label.split("|")
        overlap_rows.append(
            {
                "scenario": scen,
                "model": model,
                "bottom_decile_overlap": round(overlap, 2) if np.isfinite(overlap) else np.nan,
            }
        )

    overlap_df = pd.DataFrame(overlap_rows)

    return {
        "summary": summary,
        "pearson": pearson,
        "spearman": spearman,
        "overlap": overlap_df,
    }


def write_report(out_dir: Path, by_water: dict, input_nc: Path):
    report = []
    report.append("# Calorie Yield Delta Analysis")
    report.append("")
    report.append("## Setup")
    report.append("")
    report.append(f"- Input NetCDF: {input_nc}")
    report.append("- Units transformed to people-fed-yearly by dividing all cal_yld values by 730")
    report.append("- Baseline mask: HP8100 > 0")
    report.append("- Deltas: HP0120-HP8100 and FP2140-HP8100")
    report.append("")

    for water in ["HRLM", "HILM"]:
        if water not in by_water:
            continue
        label = WATER_LABELS.get(water, water)
        res = by_water[water]

        report.append(f"## {water} ({label})")
        report.append("")

        report.append("### Summary Table")
        report.append("")
        report.append(markdown_table(res["summary"]))
        report.append("")

        pearson_img = f"{water.lower()}_pearson_heatmap.png"
        spearman_img = f"{water.lower()}_spearman_heatmap.png"

        report.append("### Pearson Correlation Heatmap")
        report.append("")
        report.append(f"![{water} Pearson]({pearson_img})")
        report.append("")

        report.append("### Spearman Correlation Heatmap")
        report.append("")
        report.append(f"![{water} Spearman]({spearman_img})")
        report.append("")

        report.append("### Bottom-Decile Overlap (vs historical delta)")
        report.append("")
        report.append(markdown_table(res["overlap"]))
        report.append("")

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze calorie-yield delta maps from NetCDF.")
    parser.add_argument("--input", default="cal_yld_RES06-YCX.nc", help="Input NetCDF path")
    parser.add_argument("--outdir", default="outputs/analysis_cal_yld", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, LOGLEVEL), format="%(levelname)s: %(message)s")

    input_nc = pick_input_nc(Path(args.input))
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(input_nc.as_posix())
    entries = collect_entries(ds)
    if not entries:
        raise RuntimeError("No matching cal_yld variables found in input dataset")

    by_water = {}
    for water in ["HRLM", "HILM"]:
        water_entries = [
            e
            for e in entries
            if e["water"] == water and e["period"] in {"HP8100", "HP0120", "FP2140"}
        ]
        if not water_entries:
            logging.warning(f"No matching entries for water={water}; skipping")
            continue

        res = build_water_analysis(ds, water_entries, water=water)
        by_water[water] = res

        plot_corr_heatmap(
            res["pearson"],
            title=f"{water} Pearson Correlation (delta maps)",
            out_path=out_dir / f"{water.lower()}_pearson_heatmap.png",
            hist_index=0,
        )
        plot_corr_heatmap(
            res["spearman"],
            title=f"{water} Spearman Correlation (delta maps)",
            out_path=out_dir / f"{water.lower()}_spearman_heatmap.png",
            hist_index=0,
        )

        res["summary"].to_csv(out_dir / f"{water.lower()}_summary.csv", index=False)
        res["overlap"].to_csv(out_dir / f"{water.lower()}_bottom_decile_overlap.csv", index=False)
        res["pearson"].to_csv(out_dir / f"{water.lower()}_pearson_corr.csv")
        res["spearman"].to_csv(out_dir / f"{water.lower()}_spearman_corr.csv")

    write_report(out_dir, by_water, input_nc=input_nc)
    logging.info(f"Analysis written to: {out_dir}")


if __name__ == "__main__":
    main()
