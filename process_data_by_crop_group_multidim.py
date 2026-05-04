from pathlib import Path
import re

import xarray as xr
import rioxarray as rxr


# ---------- CONFIG (edit) ----------
FOLDER = Path("outputs/cal_yld_by_group/RES05-YCX")
PREFIX = "cal_yld_RES05-YCX"  # loads FOLDER / f"{PREFIX}_*.tif"
OUT_NC = Path(f"{PREFIX}_multidim.nc")
DATA_VAR_NAME = "kcal"
CLAMP_NEGATIVES = True
MODEL_FILTER = None  # e.g. "ENSEMBLE" to keep one model only
# -----------------------------------

OPEN_CHUNKS = {"y": 256, "x": 256}

NAME_RE = re.compile(
    r"^cal_yld_(RES\d{2}-[A-Z]{3})_([A-Z0-9]+)_([^_]+)_([^_]+)_([A-Z0-9]+)_([A-Z0-9]+)$"
)

TIME_ORDER = ["HP8100", "HP0120", "FP2140", "FP4160", "FP6180", "FP8100"]
SCENARIO_ORDER = ["HIST", "SSP126", "SSP370", "SSP585"]
WATER_ORDER = ["HILM", "HRLM", "LILM", "LRLM"]


def open_tif_local(path: Path):
    da = rxr.open_rasterio(path.as_posix(), masked=True, chunks=OPEN_CHUNKS).squeeze()
    return da.astype("float32")


def _order_values(values, preferred_order):
    rank = {v: i for i, v in enumerate(preferred_order)}
    return sorted(values, key=lambda v: (rank.get(v, len(preferred_order)), v))


def parse_filename(path: Path):
    m = NAME_RE.match(path.stem)
    if not m:
        raise ValueError(f"Unexpected file name format: {path.name}")

    variable_code, time_code, model, scenario, water, crop = m.groups()
    return {
        "path": path,
        "name": path.stem,
        "variable_code": variable_code,
        "time": time_code,
        "model": model,
        "scenario": scenario,
        "water": water,
        "crop": crop,
    }


def collect_records(folder: Path, prefix: str):
    files = sorted(folder.glob(f"{prefix}_*.tif"))
    if not files:
        raise FileNotFoundError(f"No tifs matching {prefix}_*.tif in {folder}")

    records = []
    for file_path in files:
        rec = parse_filename(file_path)
        if MODEL_FILTER is not None and rec["model"] != MODEL_FILTER:
            continue
        records.append(rec)

    if not records:
        raise RuntimeError("No records left after applying MODEL_FILTER")
    return records


def ensure_unique_model_per_key(records):
    seen = {}
    for rec in records:
        key = (rec["scenario"], rec["water"], rec["time"], rec["crop"])
        seen.setdefault(key, set()).add(rec["model"])

    conflicts = {k: v for k, v in seen.items() if len(v) > 1}
    if conflicts:
        sample_key, models = next(iter(conflicts.items()))
        raise RuntimeError(
            "Multiple models found for the same scenario/water/time/crop key. "
            f"Example key={sample_key}, models={sorted(models)}. "
            "Set MODEL_FILTER to choose one model."
        )


def build_multidim_dataset(records, clamp_negatives=True) -> xr.Dataset:
    ref = open_tif_local(records[0]["path"])
    ref_crs = ref.rio.crs
    ref_tx = ref.rio.transform()
    ref_shape = ref.shape

    arrays = []
    for rec in records:
        da = open_tif_local(rec["path"])

        if (da.rio.crs != ref_crs) or (da.rio.transform() != ref_tx) or (da.shape != ref_shape):
            da = da.rio.reproject_match(ref)

        if clamp_negatives:
            da = da.where(da >= 0, 0).astype("float32")

        da = da.expand_dims(
            scenario=[rec["scenario"]],
            water=[rec["water"]],
            time=[rec["time"]],
            crop=[rec["crop"]],
        )
        arrays.append(da)

    data = xr.combine_by_coords(arrays, combine_attrs="drop_conflicts").astype("float32")

    scenario_vals = _order_values({r["scenario"] for r in records}, SCENARIO_ORDER)
    water_vals = _order_values({r["water"] for r in records}, WATER_ORDER)
    time_vals = _order_values({r["time"] for r in records}, TIME_ORDER)
    crop_vals = sorted({r["crop"] for r in records})

    data = data.reindex(
        scenario=scenario_vals,
        water=water_vals,
        time=time_vals,
        crop=crop_vals,
    )

    data = data.transpose("scenario", "water", "time", "crop", "y", "x")
    data.name = DATA_VAR_NAME

    ds = xr.Dataset({DATA_VAR_NAME: data})
    ds.rio.write_crs(ref_crs, inplace=True)

    ds["x"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    ds["y"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})

    variable_codes = sorted({r["variable_code"] for r in records})
    models = sorted({r["model"] for r in records})

    ds[DATA_VAR_NAME].attrs.update(
        {
            "units": "kcal",
            "source": "GAEZ v5 + pipeline",
            "note": "Values are per-cell calories for each crop group.",
        }
    )
    ds.attrs.update(
        {
            "title": f"{PREFIX} multidimensional dataset",
            "grid": "5 arc-min, WGS84",
            "schema": "dims=scenario,water,time,crop,y,x",
            "variable_codes": ",".join(variable_codes),
            "models_present": ",".join(models),
            "model_filter": MODEL_FILTER if MODEL_FILTER is not None else "None",
            "note": "Converted from GeoTIFF collection by filename tokens.",
        }
    )
    return ds


def main():
    records = collect_records(FOLDER, PREFIX)
    ensure_unique_model_per_key(records)

    ds = build_multidim_dataset(records, clamp_negatives=CLAMP_NEGATIVES)
    print(ds)

    enc = {
        DATA_VAR_NAME: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "chunksizes": (1, 1, 1, 1, 256, 256),
        }
    }
    ds.to_netcdf(OUT_NC.as_posix(), engine="netcdf4", encoding=enc)
    print(f"wrote: {OUT_NC}")


if __name__ == "__main__":
    main()
