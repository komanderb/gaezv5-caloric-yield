from pathlib import Path
import dask
import xarray as xr
import rioxarray as rxr
import rasterio
import numpy as np
import pandas as pd
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

RASTER_CACHE_DIR = Path("cache/rasters")
OPEN_CHUNKS = {"y": 512, "x": 512}

# Water-code mapping (yield water -> harvested-area water)
YR_WATER_TO_AREA = {
    "HILM": "WSI",  # irrigated/high-input  -> irrigated harvested area
    "LILM": "WSI",  # irrigated/low-input   -> irrigated harvested area
    "HRLM": "WSR",  # rainfed/high-input    -> rainfed harvested area
    "LRLM": "WSR",  # rainfed/low-input     -> rainfed harvested area
}


def gaezv5_path(variable_code,       # e.g. "RES05-YCX" or "RES03-YLD" or "RES06-HAR"
                period,              # e.g. "HP0120","FP2140","HIST" years for TS
                climate_model,       # e.g. "GFDL-ESM4"
                scenario=None,       # e.g. "SSP126","SSP370","SSP585","HIST" (Theme 2/3/4)
                crop=None,           # e.g. "MAIZ","CSV" (Theme 3/4/5/6 depending)
                water_code=None,     # e.g. "HILM","HRLM","LILM","LRLM" (Theme 3/4)
                scheme="https"
               ):
    base = {
        "https": "https://storage.googleapis.com/fao-gismgr-gaez-v5-data/DATA/GAEZ-V5/MAPSET",
        "gs": "gs://fao-gismgr-gaez-v5-data/DATA/GAEZ-V5/MAPSET",
    }[scheme]

    if variable_code.startswith(("RES02","RES03","RES04")):  # Themes 3/4 examples
        fname = f"GAEZ-V5.{variable_code}.{period}.{climate_model}.{scenario}.{crop}.{water_code}.tif"
    elif variable_code.startswith(("RES05","RES04",)):       # Theme 4 examples also RES05-SIX/YCX/YXX
        fname = f"GAEZ-V5.{variable_code}.{period}.{climate_model}.{scenario}.{crop}.{water_code}.tif"
    elif variable_code.startswith(("RES06","RES05")):        # Themes 5/6 example (RES06-HAR etc.)
        # Theme 5/6 format: GAEZ-V5.<VARIABLE>.<CROP>.<WATER_SUPPLY>.tif
        fname = f"GAEZ-V5.{variable_code}.{crop}.{water_code}.tif"
    else:
        raise ValueError("Unrecognized variable code family.")
    return f"{base}/{variable_code}/{fname}"


# ── Local raster cache ────────────────────────────────────────────────────────

def _local_cache_path(url: str) -> Path:
    """Deterministic local path for a remote raster URL."""
    parts = url.rstrip("/").split("/")
    fname = parts[-1]
    var_code = parts[-2]
    return RASTER_CACHE_DIR / var_code / fname


def _download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to dest with atomic write. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=131072):  # 128 KB
                    f.write(chunk)
        tmp.rename(dest)
        return True
    except Exception as e:
        tmp.unlink(missing_ok=True)
        logging.debug(f"Download failed: {url} -> {e}")
        return False


def _ensure_local(url: str) -> str:
    """Return local path for a remote URL, downloading if needed."""
    local = _local_cache_path(url)
    if local.exists():
        return str(local)
    if _download_file(url, local):
        return str(local)
    raise RuntimeError(f"Failed to download: {url}")


def open_raster(url_or_path):
    """Open a raster. Remote URLs are downloaded to local cache first."""
    path = str(url_or_path)
    if path.startswith(("http://", "https://")):
        path = _ensure_local(path)
    da = rxr.open_rasterio(path, masked=True, chunks=OPEN_CHUNKS).squeeze()
    return da.astype("float32")


# ── URL existence manifest ────────────────────────────────────────────────────

def check_urls_exist(urls, max_workers=32):
    """Check which URLs exist via concurrent HEAD requests.
    URLs already in local cache are counted as existing without a network call.
    Returns the set of existing URLs."""
    existing = set()
    to_check = []
    for url in set(urls):
        if _local_cache_path(url).exists():
            existing.add(url)
        else:
            to_check.append(url)

    if not to_check:
        logging.info(f"URL manifest: all {len(existing)} URLs already cached locally")
        return existing

    logging.info(f"Checking {len(to_check)} URLs ({len(existing)} already cached) …")

    def _head(url):
        try:
            r = requests.head(url, timeout=10, allow_redirects=True)
            return url, r.status_code in (200, 206)
        except Exception:
            return url, False

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_head, u) for u in to_check]
        for fut in as_completed(futures):
            url, ok = fut.result()
            if ok:
                existing.add(url)

    logging.info(f"  {len(existing)} of {len(existing | set(to_check))} URLs available")
    return existing


# ── HAR raster preload cache ─────────────────────────────────────────────────

def preload_har_cache(groups, waters):
    """Download and open all HAR rasters for given groups and water codes.
    Returns dict keyed by (group, area_water) -> DataArray."""
    needed = {}
    for group in groups:
        for water in waters:
            area_water = YR_WATER_TO_AREA[water]
            key = (group, area_water)
            if key not in needed:
                needed[key] = gaezv5_path(
                    variable_code="RES06-HAR",
                    period=None, climate_model=None, scenario=None,
                    crop=group, water_code=area_water,
                )

    # Download all HAR files in parallel
    logging.info(f"Preloading {len(needed)} HAR rasters …")

    def _download_one(item):
        key, url = item
        try:
            local = _ensure_local(url)
            return key, local, None
        except Exception as e:
            return key, None, e

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_download_one, needed.items()))

    cache = {}
    for key, local_path, err in results:
        if err is not None:
            logging.warning(f"Failed to preload HAR {key}: {err}")
            continue
        try:
            cache[key] = open_raster(local_path)
        except Exception as e:
            logging.warning(f"Failed to open HAR {key}: {e}")

    logging.info(f"  Loaded {len(cache)}/{len(needed)} HAR rasters into cache")
    return cache


def get_crop_mapping(variable_code):

    path = "data/gaez_v5_crop_mapper.csv"

    crop_df = pd.read_csv(path).fillna("")

    crop_df = crop_df[crop_df["mapping_note"].str.lower() != "unmapped"]

    #crop_df = crop_df[crop_df["theme5_code"] != "MNG"] # for mango there is some weird stuff going on ..
    #TODO: drop fruits and nuts in total?
    for c in ["theme2_code", "theme5_code", "theme6_code"]:
        if c in crop_df.columns:
            crop_df[c] = crop_df[c].str.strip().str.upper()



    code_col = "theme5_code" if variable_code in ["RES05-YCX", "RES05-YXX"] else "theme2_code"


    out = (crop_df.loc[crop_df[code_col] != "", ["theme6_code", code_col]]
           .drop_duplicates()
           .groupby("theme6_code")[code_col]
           .apply(lambda s: sorted(s.unique()))
           .to_dict())
    return out


def get_cal_mapper(path="data/gaezv5_cal_mapping.csv"):


    df = pd.read_csv(path, dtype={"gaez_crop_code":"string"})

    df = df[df["crop_type"] == "grain"]
    df["gaez_crop_code"] = df["gaez_crop_code"].str.strip().str.upper()
    # drop FRT as too little variables and too large cal spread
    df = df[df["gaez_crop_code"] != "FRT"]
    # ensure numeric
    df["cal_yld"] = pd.to_numeric(df["cal_yld"], errors="coerce") * 10

    return dict(zip(df["gaez_crop_code"], df["cal_yld"]))


def group_kcal_average(group, crops, kcal_per_kg,
                       water_code, variable_code_yield, period, climate_model, scenario,
                       har_cache=None, url_manifest=None,
                       ):
    # transform water / input to area water
    area_water = YR_WATER_TO_AREA[water_code]

    # Use cached HAR raster if available, otherwise open fresh
    if har_cache is not None and (group, area_water) in har_cache:
        areaG = har_cache[(group, area_water)]
    else:
        a_path = gaezv5_path(
            variable_code="RES06-HAR",
            period=None, climate_model=None, scenario=None,
            crop=group, water_code=area_water
        )
        areaG = open_raster(a_path)


    yld_ha_layers = []
    ref = None
    for c in crops:
        y_path = gaezv5_path(
            variable_code=variable_code_yield,
            period=period,
            climate_model=climate_model,
            scenario=scenario,
            crop=c,
            water_code=water_code,
        )

        # Skip if URL manifest says it doesn't exist
        if url_manifest is not None and y_path not in url_manifest:
            logging.debug(f"[{group}] skipping crop={c} ({water_code}) – not in manifest")
            continue

        try:
            yld = open_raster(y_path)
        except Exception as e:
            logging.warning(f"[{group}] skipping crop={c} ({water_code}) – failed to open: {y_path} -> {e}")
            continue

        if ref is None:
            ref = yld
            areaG = areaG.rio.reproject_match(ref) if (
                (areaG.rio.crs != ref.rio.crs) or (areaG.rio.transform()!=ref.rio.transform()) or (areaG.shape!=ref.shape)
            ) else areaG
        # per-crop kcal/ha = yld(t/ha)*1000 * kcal/kg
        yld_ha_layers.append(yld.astype("float64"))

    if not yld_ha_layers:
        return areaG * 0.0

    # mean over crops present (uniform)
    yield_ha_mean = xr.concat(yld_ha_layers, dim="crop").mean("crop", skipna=True)  # mean yield /ha
    kcal_cell = yield_ha_mean * areaG * float(kcal_per_kg)               # kcal per cell
    kcal_cell.name = f"kcal_{group}"
    return kcal_cell


def sum_groups_kcal(crop_mapping, calorie_mapping,
                    water_code, variable_code_yield,
                    period, climate_model, scenario,
                    har_cache=None, url_manifest=None):

    total_kcal = None
    for group_crop, kcal in calorie_mapping.items():
        crops = crop_mapping.get(group_crop)

        layer = group_kcal_average(group=group_crop, crops=crops, kcal_per_kg=kcal,
                                   variable_code_yield=variable_code_yield,
                                   period=period, climate_model=climate_model,
                                   scenario=scenario, water_code=water_code,
                                   har_cache=har_cache, url_manifest=url_manifest)
        # Running sum avoids building a large concatenated array
        if total_kcal is None:
            total_kcal = layer.fillna(0)
        else:
            total_kcal = total_kcal + layer.fillna(0)

    if total_kcal is None:
        raise RuntimeError("No group layers produced")

    total_kcal.name = f"cal_yld_{variable_code_yield}_{period}_{climate_model}_{scenario}_{water_code}"
    return total_kcal

