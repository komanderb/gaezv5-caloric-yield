from pathlib import Path
import re

import xarray as xr
import rioxarray as rxr


# ---------- CONFIG (edit) ----------
FOLDER = Path("outputs/har_area_by_group")
PREFIX = "har_area"  # loads FOLDER / f"{PREFIX}_*.tif"
OUT_NC = Path("har_area_by_crop_group_multidim.nc")
DATA_VAR_NAME = "har_area"
CLAMP_NEGATIVES = True
# -----------------------------------

OPEN_CHUNKS = {"y": 256, "x": 256}

NAME_RE = re.compile(r"^har_area_([A-Z0-9]+)_([A-Z0-9]+)$")
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

    water, crop = m.groups()
    return {
        "path": path,
        "name": path.stem,
        "water": water,
        "crop": crop,
    }


def collect_records(folder: Path, prefix: str):
    files = sorted(folder.glob(f"{prefix}_*.tif"))
    if not files:
        raise FileNotFoundError(f"No tifs matching {prefix}_*.tif in {folder}")

    return [parse_filename(file_path) for file_path in files]


def ensure_unique_key(records):
    seen = set()
    duplicates = set()
    for rec in records:
        key = (rec["water"], rec["crop"])
        if key in seen:
            duplicates.add(key)
        seen.add(key)

    if duplicates:
        sample = sorted(duplicates)[0]
        raise RuntimeError(f"Duplicate water/crop key found: {sample}")


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
            water=[rec["water"]],
            crop=[rec["crop"]],
        )
        arrays.append(da)

    data = xr.combine_by_coords(arrays, combine_attrs="drop_conflicts").astype("float32")

    water_vals = _order_values({r["water"] for r in records}, WATER_ORDER)
    crop_vals = sorted({r["crop"] for r in records})

    data = data.reindex(
        water=water_vals,
        crop=crop_vals,
    )

    data = data.transpose("water", "crop", "y", "x")
    data.name = DATA_VAR_NAME

    ds = xr.Dataset({DATA_VAR_NAME: data})
    ds.rio.write_crs(ref_crs, inplace=True)

    ds["x"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    ds["y"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})

    ds[DATA_VAR_NAME].attrs.update(
        {
            "units": "ha",
            "source": "GAEZ v5 + pipeline",
            "note": "Harvested area per cell for each crop group.",
        }
    )
    ds.attrs.update(
        {
            "title": "har_area by crop group multidimensional dataset",
            "grid": "5 arc-min, WGS84",
            "schema": "dims=water,crop,y,x",
            "note": "Converted from GeoTIFF collection by filename tokens.",
        }
    )
    return ds


def main():
    records = collect_records(FOLDER, PREFIX)
    ensure_unique_key(records)

    ds = build_multidim_dataset(records, clamp_negatives=CLAMP_NEGATIVES)
    print(ds)

    enc = {
        DATA_VAR_NAME: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "chunksizes": (1, 1, 256, 256),
        }
    }
    ds.to_netcdf(OUT_NC.as_posix(), engine="netcdf4", encoding=enc)
    print(f"wrote: {OUT_NC}")


if __name__ == "__main__":
    main()
