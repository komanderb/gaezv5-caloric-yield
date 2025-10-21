from pathlib import Path
import dask
import xarray as xr
import rioxarray as rxr
import rasterio
import numpy as np
import pandas as pd
import logging


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


def open_raster(url):
    OPEN_CHUNKS = {"y": 512, "x": 512}
    # lean GDAL settings for faster HTTP range reads
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="TRUE",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        CPL_VSIL_CURL_CHUNK_SIZE="10485760",  # 16MB
        GDAL_HTTP_MAX_RETRY="3",
        GDAL_HTTP_RETRY_DELAY="1",
        #GDAL_CACHEMAX="1024",  # MB
        NUM_THREADS="ALL_CPUS"
    ):
        #try:

        da = rxr.open_rasterio(url, masked=True, chunks=OPEN_CHUNKS).squeeze()
        #except Exception as e:
            #raise RuntimeError(f"Failed to open raster: {url} -> {e}") from e
    return da.astype("float32")  # keeps I/O & memory lighter


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
                       water_code, variable_code_yield, period, climate_model, scenario
                       ):
    YR_WATER_TO_AREA = {
        "HILM": "WSI",  # irrigated/high-input  -> irrigated harvested area
        "LILM": "WSI",  # irrigated/low-input   -> irrigated harvested area
        "HRLM": "WSR",  # rainfed/high-input    -> rainfed harvested area
        "LRLM": "WSR",  # rainfed/low-input     -> rainfed harvested area
    }

    # transform water / input to area water
    area_water = YR_WATER_TO_AREA[water_code]
    a_path = gaezv5_path(
        variable_code="RES06-HAR",
        period=None, climate_model=None, scenario=None,
        crop=group, water_code=area_water
    )
    # load group harvested area
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
                    period, climate_model, scenario):

    group_layers = []
    for group_crop, kcal in calorie_mapping.items():
        crops = crop_mapping.get(group_crop)

        layer = group_kcal_average(group=group_crop, crops=crops, kcal_per_kg=kcal,
                                   variable_code_yield=variable_code_yield,
                                   period=period, climate_model=climate_model,
                                   scenario=scenario, water_code=water_code)
        group_layers.append(layer)

    total_kcal = xr.concat(group_layers, dim="group").sum("group", skipna=True)
    total_kcal.name = f"cal_yld_{variable_code_yield}_{period}_{climate_model}_{scenario}_{water_code}"
    return total_kcal

