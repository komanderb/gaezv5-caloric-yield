from src.utils import get_cal_mapper, gaezv5_path, open_raster
from pathlib import Path
import logging
import xarray as xr

OUTDIR = Path("outputs/har_area")
WATERS = ["HILM", "HRLM"]

# Map yield water code to harvested-area water supply
YR_WATER_TO_AREA = {
    "HILM": "WSI",  # irrigated/high-input  -> irrigated harvested area
    "HRLM": "WSR",  # rainfed/high-input    -> rainfed harvested area
}

OVERWRITE = False
LOGLEVEL = "INFO"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sum_groups_har_area(groups, water_code):
    """
    Sum harvested area across all crop groups for a given water condition.
    
    Parameters
    ----------
    groups : list
        List of Theme-6 crop group codes (e.g., ["BAN", "BRL", ...])
    water_code : str
        Water condition code ("HILM" or "HRLM")
    
    Returns
    -------
    xr.DataArray
        Total harvested area summed across all groups
    """
    area_water = YR_WATER_TO_AREA[water_code]
    
    area_layers = []
    ref = None
    
    for group in groups:
        a_path = gaezv5_path(
            variable_code="RES06-HAR",
            period=None,
            climate_model=None,
            scenario=None,
            crop=group,
            water_code=area_water,
        )
        try:
            areaG = open_raster(a_path)
        except Exception as e:
            logging.warning(f"Skipping group={group} ({water_code}) – failed to open: {a_path} -> {e}")
            continue
        
        # Reproject to match reference if needed
        if ref is None:
            ref = areaG
        else:
            if (areaG.rio.crs != ref.rio.crs) or (areaG.rio.transform() != ref.rio.transform()) or (areaG.shape != ref.shape):
                areaG = areaG.rio.reproject_match(ref)
        
        area_layers.append(areaG.astype("float64"))
    
    if not area_layers:
        raise ValueError(f"No valid area layers found for water_code={water_code}")
    
    # Sum across all groups
    total_area = xr.concat(area_layers, dim="group").sum("group", skipna=True)
    total_area.name = f"har_area_{water_code}"
    return total_area


def run():
    logging.basicConfig(level=getattr(logging, LOGLEVEL), format="%(levelname)s: %(message)s")
    ensure_dir(OUTDIR)

    # Get the crop groups from calorie mapping (Theme-6 codes)
    cal_map_t6 = get_cal_mapper()
    groups = list(cal_map_t6.keys())
    
    logging.info(f"Processing {len(groups)} crop groups: {groups}")

    for water in WATERS:
        fname = f"har_area_{water}"
        out_path = OUTDIR / f"{fname}.tif"
        
        if out_path.exists() and not OVERWRITE:
            logging.info(f"Exists, skipping: {out_path}")
            continue

        logging.info(f"Building {fname} …")
        try:
            da: xr.DataArray = sum_groups_har_area(
                groups=groups,
                water_code=water,
            )
            da = da.astype("float32")
            da.attrs.setdefault("units", "ha")
            da.attrs.setdefault("source", "GAEZ v5")

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
        except Exception as e:
            logging.exception(f"Failed {fname}: {e}")


if __name__ == "__main__":
    run()
