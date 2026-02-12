# scripts/make_har_area_dataset_from_tifs.py
from pathlib import Path
import xarray as xr
import rioxarray as rxr

# ---------- CONFIG (edit) ----------
FOLDER = Path("outputs/har_area")
PREFIX = "har_area"                     # loads FOLDER / f"{PREFIX}_*.tif"
OUT_NC = Path(f"{PREFIX}.nc")           # NetCDF output
CLAMP_NEGATIVES = True                  # set <0 to 0
# -----------------------------------

OPEN_CHUNKS = {"y": 256, "x": 256}      # keep or remove if you want full in-RAM


def open_tif_local(path: Path):
    da = rxr.open_rasterio(path.as_posix(), masked=True, chunks=OPEN_CHUNKS).squeeze()
    return da.astype("float32")


def build_dataset_for_har_area(folder: Path, variable_prefix: str, clamp_negatives=True) -> xr.Dataset:
    """
    Build an xarray Dataset from harvested area TIF files.
    
    Parameters
    ----------
    folder : Path
        Directory containing the TIF files
    variable_prefix : str
        Prefix for TIF files to load (e.g., "har_area" matches "har_area_*.tif")
    clamp_negatives : bool
        If True, clamp negative values to 0
    
    Returns
    -------
    xr.Dataset
        Dataset with each TIF as a data variable
    """
    files = sorted(folder.glob(f"{variable_prefix}_*.tif"))
    if not files:
        raise FileNotFoundError(f"No tifs matching {variable_prefix}_*.tif in {folder}")

    # reference grid from first file
    ref = open_tif_local(files[0])
    ref_crs = ref.rio.crs
    ref_tx = ref.rio.transform()
    ref_shp = ref.shape

    data_vars = {}
    for f in files:
        da = open_tif_local(f)
        # align to ref if needed (should already match)
        if (da.rio.crs != ref_crs) or (da.rio.transform() != ref_tx) or (da.shape != ref_shp):
            da = da.rio.reproject_match(ref)
        if clamp_negatives:
            da = da.where(da >= 0, 0).astype("float32")   # clamp <0 to 0
        data_vars[f.stem] = da

    ds = xr.Dataset(data_vars).assign_coords(x=ref.x, y=ref.y)
    ds.rio.write_crs(ref_crs, inplace=True)

    # nicer CF-ish coord names for portability
    ds["x"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    ds["y"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})

    # Variable-specific attributes
    for var in ds.data_vars:
        ds[var].attrs.update({
            "units": "ha",
            "long_name": f"Harvested area ({var.replace('har_area_', '')})",
        })

    ds.attrs.update({
        "title": f"{variable_prefix} (harvested area per cell)",
        "grid": "5 arc-min, WGS84",
        "source": "GAEZ v5 + pipeline",
        "note": "Each variable = one water supply type (HILM=irrigated, HRLM=rainfed). Negatives clamped to 0.",
    })
    return ds


def main():
    ds = build_dataset_for_har_area(FOLDER, PREFIX, clamp_negatives=CLAMP_NEGATIVES)
    print(ds)

    # save NetCDF (portable single file)
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (256, 256)}
           for v in ds.data_vars}
    ds.to_netcdf(OUT_NC.as_posix(), engine="netcdf4", encoding=enc)
    print(f"wrote: {OUT_NC}")


if __name__ == "__main__":
    main()
