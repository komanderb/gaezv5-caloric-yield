# GAEZv5 kcal pipeline

Compute **calories per grid cell** from GAEZ v5 Theme 2 potential yields and Theme 5 attainable yields, using Theme 6 harvested area.

## What it does
- Loads yields (Theme 2/5) and harvested area (Theme 6) from public GCS.
- Uniform-averages per-crop `(yield × kcal/kg)` within Theme-6 groups, then multiplies by the group harvested area.
- Handles historical (`HIST`/AGERA5) and future (SSP126/370/585, ENSEMBLE).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python get_data.py
