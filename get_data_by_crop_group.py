from pathlib import Path
import logging
import xarray as xr

from src.utils import (
    group_kcal_average,
    get_crop_mapping,
    get_cal_mapper,
    gaezv5_path,
    preload_har_cache,
    check_urls_exist,
)

OUTDIR = Path("outputs/cal_yld_by_group")
YIELD_VARS = ["RES05-YCX"]
WATERS = ["HILM", "HRLM"]  # "LILM","LRLM" only exist for some crops

OVERWRITE = False
LOGLEVEL = "INFO"

HIST_PERIODS = ["HP8100", "HP0120"]
FUTURE_PERIODS = ["FP2140", "FP4160", "FP6180", "FP8100"]

HIST_MODELS = ["AGERA5"]
FUTURE_MODELS = ["ENSEMBLE"]

SCENARIOS = {
    "HIST": {"periods": HIST_PERIODS, "models": HIST_MODELS},
    "SSP126": {"periods": FUTURE_PERIODS, "models": FUTURE_MODELS},
    "SSP370": {"periods": FUTURE_PERIODS, "models": FUTURE_MODELS},
    "SSP585": {"periods": FUTURE_PERIODS, "models": FUTURE_MODELS},
}


def valid_combos():
    for scen, cfg in SCENARIOS.items():
        for period in cfg["periods"]:
            for model in cfg["models"]:
                yield (period, model, scen)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _build_yield_urls(yield_var, crop_mapping, groups):
    """Enumerate all yield URLs that will be requested during the run."""
    urls = set()
    for (period, model, scen) in valid_combos():
        for water in WATERS:
            for group in groups:
                for crop in crop_mapping[group]:
                    urls.add(
                        gaezv5_path(
                            variable_code=yield_var,
                            period=period,
                            climate_model=model,
                            scenario=scen,
                            crop=crop,
                            water_code=water,
                        )
                    )
    return urls


def run():
    logging.basicConfig(
        level=getattr(logging, LOGLEVEL),
        format="%(levelname)s: %(message)s",
    )
    ensure_dir(OUTDIR)

    # kcal per kg by Theme-6 code
    cal_map_t6 = get_cal_mapper()

    for yield_var in YIELD_VARS:
        outdir_var = OUTDIR / yield_var
        ensure_dir(outdir_var)
        crop_mapping = get_crop_mapping(yield_var)

        groups = [g for g in cal_map_t6 if g in crop_mapping]
        if not groups:
            logging.warning(f"No valid crop groups found for {yield_var}, skipping")
            continue

        # Pre-build caches once per yield_var
        har_cache = preload_har_cache(groups, WATERS)

        yield_urls = _build_yield_urls(yield_var, crop_mapping, groups)
        url_manifest = check_urls_exist(yield_urls)

        for (period, model, scen) in valid_combos():
            for water in WATERS:
                for group in groups:
                    fname = f"cal_yld_{yield_var}_{period}_{model}_{scen}_{water}_{group}"
                    out_path = outdir_var / f"{fname}.tif"
                    if out_path.exists() and not OVERWRITE:
                        logging.info(f"Exists, skipping: {out_path}")
                        continue

                    logging.info(f"Building {fname} ...")
                    try:
                        da: xr.DataArray = group_kcal_average(
                            group=group,
                            crops=crop_mapping[group],
                            kcal_per_kg=cal_map_t6[group],
                            water_code=water,
                            variable_code_yield=yield_var,
                            period=period,
                            climate_model=model,
                            scenario=scen,
                            har_cache=har_cache,
                            url_manifest=url_manifest,
                        )
                        da = da.astype("float32")
                        da.name = fname
                        da.attrs.setdefault("units", "kcal")
                        da.attrs.setdefault("source", "GAEZ v5")
                        da.attrs.setdefault("crop_group", group)

                        da.rio.to_raster(
                            out_path.as_posix(),
                            tiled=True,
                            BLOCKXSIZE=512,
                            BLOCKYSIZE=512,
                            NUM_THREADS="ALL_CPUS",
                            windowed=True,
                            BIGTIFF="IF_SAFER",
                            dtype="float32",
                            SPARSE_OK=True,
                            compress="ZSTD",
                            ZSTD_LEVEL=5,
                        )
                        logging.info(f"Saved {out_path}")
                    except Exception as exc:
                        logging.exception(f"Failed {fname}: {exc}")


if __name__ == "__main__":
    run()
