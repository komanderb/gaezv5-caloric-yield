from pathlib import Path
import logging
import xarray as xr

from src.utils import get_cal_mapper, preload_har_cache, YR_WATER_TO_AREA

OUTDIR = Path("outputs/har_area_by_group")
WATERS = ["HILM", "HRLM"]

OVERWRITE = False
LOGLEVEL = "INFO"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run():
    logging.basicConfig(
        level=getattr(logging, LOGLEVEL),
        format="%(levelname)s: %(message)s",
    )
    ensure_dir(OUTDIR)

    # Use the same Theme-6 group filter as calorie outputs.
    cal_map_t6 = get_cal_mapper()
    groups = list(cal_map_t6.keys())

    logging.info(f"Preparing harvested area for {len(groups)} crop groups")
    har_cache = preload_har_cache(groups, WATERS)

    for water in WATERS:
        area_water = YR_WATER_TO_AREA[water]

        for group in groups:
            fname = f"har_area_{water}_{group}"
            out_path = OUTDIR / f"{fname}.tif"

            if out_path.exists() and not OVERWRITE:
                logging.info(f"Exists, skipping: {out_path}")
                continue

            key = (group, area_water)
            if key not in har_cache:
                logging.warning(f"Missing HAR raster for group={group}, water={water}; skipping")
                continue

            logging.info(f"Building {fname} ...")
            try:
                da: xr.DataArray = har_cache[key].astype("float32")
                da.name = fname
                da.attrs.setdefault("units", "ha")
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
