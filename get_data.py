from src.utils import sum_groups_kcal, get_crop_mapping, get_cal_mapper
from pathlib import Path
import logging
import xarray as xr

OUTDIR = Path("outputs/cal_yld")
YIELD_VARS = ["RES05-YCX"]
WATERS = ["HILM","HRLM"] #"LILM","LRLM" only exist for some crops

OVERWRITE = False
LOGLEVEL = "INFO"

# Periods split
HIST_PERIODS = ["HP8100","HP0120"]       #w need to split
FUTURE_PERIODS = ["FP2140","FP4160","FP6180","FP8100"]

# Models per scenario (adjust to your chosen set)
HIST_MODELS = ["AGERA5"]                       # <- the “model” token used for historical in GAEZ v5
FUTURE_MODELS = ["ENSEMBLE"]                  # extend as needed

SCENARIOS = {
    "HIST":   {"periods": HIST_PERIODS,   "models": HIST_MODELS},
    "SSP126": {"periods": FUTURE_PERIODS, "models": FUTURE_MODELS},
    "SSP370": {"periods": FUTURE_PERIODS, "models": FUTURE_MODELS},
    "SSP585": {"periods": FUTURE_PERIODS, "models": FUTURE_MODELS},
}
# ----------------------------------------------------


def valid_combos():
    for scen, cfg in SCENARIOS.items():
        for period in cfg["periods"]:
            for model in cfg["models"]:
                yield (period, model, scen)


def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)


def run():

    logging.basicConfig(level=getattr(logging, LOGLEVEL), format="%(levelname)s: %(message)s")
    ensure_dir(OUTDIR)

    # kcal per kg by Theme-6 code
    cal_map_t6 = get_cal_mapper()  # {"BAN": 394, "BRL": 391, ...}

    for yield_var in YIELD_VARS:
        outdir_var = OUTDIR / yield_var
        ensure_dir(outdir_var)
        crop_mapping = get_crop_mapping(yield_var)

        for (period, model, scen) in valid_combos():
            for water in WATERS:

                fname = f"cal_yld_{yield_var}_{period}_{model}_{scen}_{water}"
                out_path = outdir_var / f"{fname}.tif"
                if out_path.exists() and not OVERWRITE:
                    logging.info(f"Exists, skipping: {out_path}")
                    continue

                logging.info(f"Building {fname} …")
                try:
                    da: xr.DataArray = sum_groups_kcal(
                        crop_mapping=crop_mapping,
                        calorie_mapping=cal_map_t6,
                        water_code=water,
                        variable_code_yield=yield_var,
                        period=period,
                        climate_model=model,
                        scenario=scen,
                      )
                    da = da.astype("float32")
                    rname = da.name or fname
                    out_path = outdir_var /f"{rname}.tif"
                    da = da.astype("float32")
                    da.attrs.setdefault("units", "kcal")
                    da.attrs.setdefault("source", "GAEZ v5")

                    da.rio.to_raster(
                      out_path.as_posix(),
                      tiled=True, BLOCKXSIZE=512, BLOCKYSIZE=512,
                      NUM_THREADS="ALL_CPUS",
                      windowed=True, BIGTIFF="IF_SAFER",
                      dtype="float32", SPARSE_OK=True,      # <— skips all-zero tiles
                      compress="ZSTD",        # here had no compression before
                      ZSTD_LEVEL=5
                    )
                    logging.info(f"Saved {out_path}")
                except Exception as e:
                    logging.exception(f"Failed {fname}: {e}")


if __name__ == "__main__":
  run()