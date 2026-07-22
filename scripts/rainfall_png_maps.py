"""
rainfall_png_maps.py
=====================
GFS + ECMWF + ICON branded PNG map renderer. Fetches forecast data and renders
publication-quality PNG maps with title block, colorbar, and attribution.

This script produces PNG maps ONLY — it does not write JSON grids.
For the windy-style interactive viewer JSON data, use rainfall_viewer_data.py.

GFS + ECMWF + ICON running-total rainfall forecast with a DYNAMIC day range per
model AND per run cycle (00/06/12/18 UTC) — each combination supports a
different max lead time, verified against the live ECMWF/NOAA/DWD docs:

  GFS (NOAA, 0.25 deg):  384h (16 days) on every cycle (00/06/12/18Z).
                         Native cadence: hourly to 120h, 3-hourly to 240h,
                         12-hourly to 384h.
  ECMWF IFS (open-data): 360h (15 days) on 00Z/12Z, but only 144h (6 days)
                         on 06Z/18Z (current as of IFS Cycle 50r1, May 2026 —
                         06z/18z used to be capped at 90h under the
                         now-discontinued 'scda' stream; see
                         https://www.ecmwf.int/en/forecasts/datasets/open-data).
                         Native cadence: 3-hourly to 144h, 6-hourly beyond.
  ICON Global (DWD):     180h (7.5 days) on 00Z/12Z, 120h (5 days) on 06Z/18Z.
                         Native cadence: hourly to 78h, 3-hourly to 180h.
                         NOTE: DWD tot_prec is an HOURLY INCREMENT (rain in the
                         preceding 1 hour only), NOT a running total like GFS/ECMWF.
                         fetch_icon_all_hours() accumulates these increments into
                         running totals so the rest of the pipeline is identical
                         for all three models.

FINE-GRAINED (3-HOURLY) WINDOW: in addition to the daily 'Day N' running
totals, every step is also fetched at 3-hourly cadence (FINE_CADENCE_HOURS)
out to each model's own true native 3-hourly ceiling (FINE_CUTOFF_HOUR[model],
derived directly from gfs_native_steps()/ecmwf_native_steps()/icon_native_steps()
rather than a single shared number) — GFS genuinely supports 3-hourly out to
240h, ECMWF only to 144h, ICON only to 78h (then 3-hourly beyond that to 180h),
so each model gets its real ceiling rather than both being capped to the more
restrictive one.

GFS ACCUMULATION FIX: GFS pgrb2 files publish TWO precipitation records
under the same APCP name for any lead time beyond the first output
interval — one accumulated since forecast hour 0 (the genuine running
total) and one accumulated only since the last synoptic/output bucket
(e.g. the most recent 3-6h). This is documented NCEP behaviour (see
ecmwf/cfgrib issues #321, #344 and the NCEP/kerchunk write-up at
https://github.com/fsspec/kerchunk/issues/407). A naive "just grab the
tp/APCP variable" read can silently end up with the bucket record instead
of the true cumulative one. The two records are distinguished by their
GRIB stepRange: the cumulative-since-start record always starts at 0
(e.g. "0-72"), the bucket record doesn't (e.g. "66-72"). This script
explicitly filters for startStep == 0. See open_grib2_get_rainfall().

ICON ACCUMULATION: Unlike GFS (APCP since t=0) and ECMWF (tp since t=0),
DWD ICON publishes tot_prec as the rain that fell in the PRECEDING 1 hour only.
To produce a running total we sum all hourly increments from hour 1 up to each
target hour. This is handled entirely inside fetch_icon_all_hours() — the rest
of the pipeline (compute_delta_grids, render_png, save_meta, upload_to_r2)
sees identical running-total arrays for all three models.

Coverage (BBOX) spans 40-100°E, 0-40°N: this fully includes the Arabian Sea
(out to the Gulf of Aden/Somali coast near 43-50°E) and the western Indian
Ocean down to the equator, alongside the existing Bay of Bengal coverage.

Outputs (per model, in output/MODEL/):
  india_fxx{NNN}.png
  gujarat_plain_fxx{NNN}.png
  gujarat_district_fxx{NNN}.png
  output/meta.json

USAGE:
  pip install requests numpy matplotlib cartopy geopandas shapely scipy cfgrib eccodes ecmwf-opendata

  python rainfall_png_maps.py --model GFS
  python rainfall_png_maps.py --model ECMWF
  python rainfall_png_maps.py --model ICON
  python rainfall_png_maps.py --model both    # GFS + ECMWF (existing behaviour)
  python rainfall_png_maps.py --model all     # GFS + ECMWF + ICON
  python rainfall_png_maps.py --model GFS --no-png
"""

import argparse
import os
import sys
import json
import bz2
import tempfile
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import geopandas as gpd
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Expanded west to 40°E and south to 0° (equator) so the domain fully covers
# the Arabian Sea (incl. Gulf of Aden / Somali coast) and the western Indian
# Ocean, in addition to the existing Bay of Bengal / Himalayan coverage.
BBOX = dict(west=40, east=100, south=0, north=40)
OUT_DIR = "output"

# ── MODEL LOGOS (PNG export) ──────────────────────────────────────────────────
LOGO_DIR = "assets/logos"
LOGO_FILES = {
    'GFS':   'gfs_logo.png',
    'ECMWF': 'ecmwf_logo.png',
    'ICON':  'icon_logo.png',    # place DWD ICON logo here if you have it
}

# ── DYNAMIC FORECAST-HOUR SCHEDULE ────────────────────────────────────────────
# Max lead time (hours) depends on BOTH model and which of the four daily UTC
# cycles (00/06/12/18) actually produced the run. Verified against:
#   GFS:   https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
#   ECMWF: https://www.ecmwf.int/en/forecasts/datasets/open-data
#   ICON:  https://dwd-geoportal.de/products/G_EJM/ (July 2026)
MODEL_MAX_HOURS = {
    'GFS':   {0: 384, 6: 384, 12: 384, 18: 384},
    'ECMWF': {0: 360, 6: 144, 12: 360, 18: 144},
    'ICON':  {0: 180, 6: 120, 12: 180, 18: 120},
}

# ── ICON OPEN-DATA BASE URL ───────────────────────────────────────────────────
# DWD publishes regular-lat-lon remapped GRIB2 files (0.1 deg) alongside the
# native icosahedral files — we use the regular-lat-lon version so no CDO or
# regridding is needed. Files are bz2-compressed. Available ~5-6h after init.
# Files are deleted from server after ~24h — archive in R2 immediately.
ICON_BASE = "https://opendata.dwd.de/weather/nwp/icon/grib"

def gfs_native_steps(max_hour):
    """Legal GFS output hours: hourly 0-120, 3-hourly 120-240, 12-hourly 240-384."""
    steps = list(range(0, min(max_hour, 120) + 1, 1))
    if max_hour > 120:
        steps += list(range(123, min(max_hour, 240) + 1, 3))
    if max_hour > 240:
        steps += list(range(252, max_hour + 1, 12))
    return sorted(set(steps))

def ecmwf_native_steps(max_hour):
    """Legal ECMWF output hours: 3-hourly 0-144, 6-hourly 150+ (00/12Z only)."""
    steps = list(range(0, min(max_hour, 144) + 1, 3))
    if max_hour > 144:
        steps += list(range(150, max_hour + 1, 6))
    return sorted(set(steps))

def icon_native_steps(max_hour):
    """
    Legal ICON Global output hours for tot_prec.
    Hourly   : 0-78h  (every 1h)
    3-hourly : 81-180h (every 3h)

    IMPORTANT: ICON tot_prec is an HOURLY INCREMENT (rain in the preceding
    1 hour only), NOT a running total like GFS/ECMWF. fetch_icon_all_hours()
    accumulates these increments into running totals automatically so the
    rest of the pipeline is identical for all three models.
    """
    steps = list(range(0, min(max_hour, 78) + 1, 1))
    if max_hour > 78:
        steps += list(range(81, max_hour + 1, 3))
    return sorted(set(steps))

def day_milestones(max_hour, native_steps=None):
    """
    The 24h-multiple 'Day N' marks. Kept as a utility since "which hours
    are clean day boundaries" is still a useful concept on its own.
    If native_steps is given, only milestones that are legal native output
    steps are kept.
    """
    milestones = list(range(24, max_hour + 1, 24))
    if native_steps is not None:
        native_set = set(native_steps)
        milestones = [m for m in milestones if m in native_set]
    return milestones

# ── FINE-GRAINED (3-HOURLY) SCHEDULE ──────────────────────────────────────────
FINE_CADENCE_HOURS      = 3
FINE_CEILING_PROBE_HOUR = 500   # comfortably beyond any model's real max

def native_fine_ceiling(native_steps_fn, fine_cadence=FINE_CADENCE_HOURS,
                         probe_hour=FINE_CEILING_PROBE_HOUR):
    """
    The largest hour H such that every fine_cadence-hour step up to H is a
    genuine native output step for this model.
    """
    native_set = set(native_steps_fn(probe_hour))
    h = 0
    while (h + fine_cadence) in native_set:
        h += fine_cadence
    return h

FINE_CUTOFF_HOUR = {
    'GFS':   native_fine_ceiling(gfs_native_steps),    # 240h
    'ECMWF': native_fine_ceiling(ecmwf_native_steps),  # 144h
    'ICON':  native_fine_ceiling(icon_native_steps),   # 78h (hourly to 78, then 3h)
}

def native_medium_cadence(native_steps_fn, fine_ceiling, probe_hour=FINE_CEILING_PROBE_HOUR):
    """
    The model's native step spacing immediately beyond its fine-cadence zone.
    Derived by inspecting the native-step function itself so it can't drift
    out of sync if native cadences are ever updated.
    """
    native_set = sorted(set(native_steps_fn(probe_hour)))
    after = [h for h in native_set if h > fine_ceiling]
    if len(after) < 2:
        return None
    return after[1] - after[0]

MEDIUM_CADENCE_HOURS = {
    'GFS':   native_medium_cadence(gfs_native_steps,   FINE_CUTOFF_HOUR['GFS']),   # 12h
    'ECMWF': native_medium_cadence(ecmwf_native_steps, FINE_CUTOFF_HOUR['ECMWF']), # 6h
    'ICON':  native_medium_cadence(icon_native_steps,  FINE_CUTOFF_HOUR['ICON']),  # 3h
}

def fine_grained_steps(max_hour, native_steps, fine_cutoff_hour, fine_cadence=FINE_CADENCE_HOURS):
    """
    Full fetch ladder, using the model's TRUE native resolution throughout.
    fine_cadence-hourly from hour `fine_cadence` up to fine_cutoff_hour,
    then EVERY native step beyond that all the way to max_hour.
    Hour 0 is skipped — trivially all-zero precipitation.
    """
    native_set   = set(native_steps)
    fine_limit   = min(fine_cutoff_hour, max_hour)
    fine_steps   = [h for h in range(fine_cadence, fine_limit + 1, fine_cadence) if h in native_set]
    medium_steps = [h for h in native_steps if fine_limit < h <= max_hour]
    return sorted(set(fine_steps) | set(medium_steps))

def day_label(fxx):
    """
    'Day N' for a clean 24h multiple; 'Day N +Hh' for a 3-hourly sub-step.
    """
    if fxx % 24 == 0:
        return f"Day {fxx // 24}"
    day_num     = fxx // 24 + 1
    hour_in_day = fxx % 24
    return f"Day {day_num} +{hour_in_day}h"

# India boundary files — manually downloaded and placed alongside the script.
INDIA_COMPOSITE_DEFAULT = "india-composite.geojson"
INDIA_STATES_DEFAULT    = "india_state.geojson"

# ── COLOUR SCALE (Meteologix-style) — COMMENTED OUT, retained for reference ──
# Uncomment below and comment out the new scale to restore the old Meteologix scale.
# OLD_LEVELS = [0, 0.1, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90,
#               100, 125, 150, 175, 200, 250, 300, 400, 500]
# OLD_COLORS_FILL = [
#     '#ffffff00',  # <0.1      transparent
#     '#DEDEF2',    # 0.1-1
#     '#B4D7FF',    # 1-2
#     '#80BFFF',    # 2-3
#     '#359AFF',    # 3-5
#     '#0C86FF',    # 5-7
#     '#0069D2',    # 7-10
#     '#00367F',    # 10-15
#     '#148F1B',    # 15-20
#     '#19C404',    # 20-25
#     '#63ED07',    # 25-30
#     '#FFF42B',    # 30-40
#     '#E8DC00',    # 40-50
#     '#F06000',    # 50-60
#     '#FF7F27',    # 60-70
#     '#FF9F5F',    # 70-80
#     '#F84E78',    # 80-90
#     '#F71E54',    # 90-100
#     '#BF0000',    # 100-125
#     '#910000',    # 125-150
#     '#64007F',    # 150-175
#     '#C200FB',    # 175-200
#     '#DD66FF',    # 200-250
#     '#E99BFF',    # 250-300
#     '#F9E6FF',    # 300-400
#     '#D4D4D4',    # 400-500
#     '#969696',    # 500+
# ]
# ── END OLD SCALE ─────────────────────────────────────────────────────────────

# ── COLOUR SCALE — Spectral/PRISM-style (active) ─────────────────────────────
LEVELS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150,
          175, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
COLORS_FILL = [
    '#ffffff',   # 0-5       white (trace/none)
    '#e0f7fa',   # 5-10      very pale aqua
    '#80deea',   # 10-15     light teal
    '#26c6da',   # 15-20     medium cyan
    '#00acc1',   # 20-25     teal
    '#00c853',   # 25-30     green
    '#2db400',   # 30-40     bright green
    '#76d400',   # 40-50     lime green
    '#c6ef00',   # 50-60     yellow-green
    '#f0e800',   # 60-70     yellow
    '#f5c200',   # 70-80     golden yellow
    '#e88000',   # 80-90     orange
    '#c85a00',   # 90-100    dark orange
    '#8b3200',   # 100-125   brown
    '#6b0000',   # 125-150   dark red
    '#3b0000',   # 150-175   very dark maroon
    '#1a0030',   # 175-200   near-black purple
    '#4a0080',   # 200-250   dark purple
    '#6a00aa',   # 250-300   medium purple
    '#cc00cc',   # 300-400   magenta
    '#e8008a',   # 400-500   hot pink
    '#e060b0',   # 500-600   medium pink
    '#f0a0d0',   # 600-700   light pink
    '#ffd0e8',   # 700-800   very light pink
    '#cccccc',   # 800-900   light grey
    '#888888',   # 900-1000  medium grey
    '#444444',   # 1000+     dark grey (extend='max')
]
CMAP = mcolors.ListedColormap(COLORS_FILL)
NORM = BoundaryNorm(LEVELS, ncolors=len(COLORS_FILL))


# ── INDIA BORDERS ─────────────────────────────────────────────────────────────

def load_india_geodataframes(composite_path=INDIA_COMPOSITE_DEFAULT,
                              states_path=INDIA_STATES_DEFAULT):
    """
    Load the two manually-supplied India boundary GeoJSONs.
    composite_path : datameet india-composite.geojson — single country polygon.
    states_path    : geohacker india_state.geojson    — one polygon per state/UT.
    """
    for fpath, label in [(composite_path, 'india-composite'),
                         (states_path,    'india-states')]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"India boundary file not found: {fpath}\n"
                f"  Download '{label}' and place it alongside the script, or\n"
                f"  pass the correct path via the CLI argument.")
    composite_gdf = gpd.read_file(composite_path)
    states_gdf    = gpd.read_file(states_path)
    print(f"  India composite: {len(composite_gdf)} feature(s) <- {composite_path}")
    print(f"  India states:    {len(states_gdf)} feature(s)  <- {states_path}")
    return composite_gdf, states_gdf


# ── GRIB2 READER ──────────────────────────────────────────────────────────────

def open_grib2_get_rainfall(tmp_path):
    """
    Try multiple cfgrib strategies. Returns DataArray or None.
    Works identically for GFS APCP, ECMWF tp, and ICON tot_prec.

    GFS files can contain TWO precipitation records — one accumulated since
    forecast hour 0 (the genuine running total) and one accumulated only since
    the last output bucket. Filtering on startStep == 0 selects the correct
    cumulative-since-start record. See ecmwf/cfgrib issues #321, #344.
    ECMWF and ICON have no such second record so this filter is a no-op there.
    """
    import cfgrib

    try:
        ds = cfgrib.open_dataset(tmp_path,
            filter_by_keys={'stepType': 'accum', 'startStep': 0}, errors='ignore')
        for v in ['tp', 'acpcp', 'unknown']:
            if v in ds:
                return ds[v]
        if ds.data_vars:
            return ds[list(ds.data_vars)[0]]
    except Exception as e:
        print(f"    filter stepType=accum,startStep=0: {e} — falling back")

    try:
        datasets = cfgrib.open_datasets(tmp_path)
        for ds in datasets:
            for v in ['tp', 'acpcp', 'unknown']:
                if v in ds:
                    print("    WARNING: fell back to open_datasets() — "
                          "accumulation style not verified, may undercount")
                    return ds[v]
            if ds.data_vars:
                print("    WARNING: fell back to open_datasets() — "
                      "accumulation style not verified, may undercount")
                return ds[list(ds.data_vars)[0]]
    except Exception as e:
        print(f"    open_datasets: {e}")

    try:
        ds = cfgrib.open_dataset(tmp_path,
            filter_by_keys={'stepType': 'accum'}, errors='ignore')
        for v in ['tp', 'acpcp', 'unknown']:
            if v in ds:
                print("    WARNING: fell back to stepType=accum (no startStep filter)")
                return ds[v]
        if ds.data_vars:
            print("    WARNING: fell back to stepType=accum (no startStep filter)")
            return ds[list(ds.data_vars)[0]]
    except Exception as e:
        print(f"    filter stepType=accum: {e}")

    try:
        ds = cfgrib.open_dataset(tmp_path, errors='ignore')
        for v in ['tp', 'acpcp', 'unknown']:
            if v in ds:
                print("    WARNING: fell back to plain open_dataset()")
                return ds[v]
        if ds.data_vars:
            print("    WARNING: fell back to plain open_dataset()")
            return ds[list(ds.data_vars)[0]]
    except Exception as e:
        print(f"    plain open_dataset: {e}")

    return None


# ── GFS FETCH (single hour) ───────────────────────────────────────────────────

def fetch_gfs_single(run_dt, forecast_hour, res=0.25):
    """
    Download one GFS APCP file from NOMADS for a specific run_dt + forecast_hour.
    Returns (lats, lons, grid_mm) or raises on failure.
    """
    res_str  = '0p25' if res <= 0.25 else '0p50'
    product  = f'pgrb2.{res_str}'
    date_str = run_dt.strftime('%Y%m%d')
    run_str  = f"{run_dt.hour:02d}"
    fxx_str  = f"{forecast_hour:03d}"
    filename = f"gfs.t{run_str}z.{product}.f{fxx_str}"

    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{res_str}.pl"
        f"?dir=%2Fgfs.{date_str}%2F{run_str}%2Fatmos"
        f"&file={filename}"
        f"&var_APCP=on"
        f"&subregion=&leftlon={BBOX['west']}&rightlon={BBOX['east']}"
        f"&toplat={BBOX['north']}&bottomlat={BBOX['south']}"
    )

    r = requests.get(url, timeout=120, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")

    content = r.content
    if len(content) < 5000 or content[:4] != b'GRIB':
        raise RuntimeError(f"Not valid GRIB2 ({len(content)} bytes)")

    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"gfs_{date_str}_{run_str}_f{fxx_str}.grib2")
    with open(tmp_path, 'wb') as f:
        f.write(content)

    try:
        da = open_grib2_get_rainfall(tmp_path)
        if da is None:
            raise RuntimeError("Could not extract rainfall variable")

        lats = da.latitude.values
        lons = da.longitude.values
        vals = da.values
        if vals.ndim == 3:
            vals = vals[-1]
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            vals = vals[::-1, :]
        vals = np.where(np.isnan(vals), 0.0, vals)
        return lats, lons, vals

    finally:
        try: os.unlink(tmp_path)
        except: pass


def compute_delta_grids(results, lookback_hours, forecast_hours=None):
    """
    Generic trailing-delta computation: delta[fxx] = total[fxx] - total[fxx - lookback_hours].
    Works for ANY lookback window (3h, 24h, ...) as long as both endpoints were fetched.
    Early steps where fxx - lookback_hours <= 0 fall back to the running total itself.
    Returns dict {fxx: (lats, lons, delta_grid)}. Negative values clipped to 0.
    """
    if forecast_hours is None:
        forecast_hours = sorted(results.keys())

    deltas = {}
    for fxx in sorted(forecast_hours):
        if fxx not in results:
            continue
        lats, lons, total_now = results[fxx]
        prev_fxx = fxx - lookback_hours

        if prev_fxx <= 0 or prev_fxx not in results:
            delta_grid = total_now.copy()
        else:
            _, _, total_prev = results[prev_fxx]
            delta_grid = np.clip(total_now - total_prev, 0, None)

        deltas[fxx] = (lats, lons, delta_grid)

    return deltas

def compute_incremental_grids(results, forecast_hours=None):
    """Backwards-compatible alias: the original 24h-trailing delta."""
    return compute_delta_grids(results, lookback_hours=24, forecast_hours=forecast_hours)


# ── GFS FETCH (all hours) ────────────────────────────────────────────────────

def fetch_gfs_all_hours(res=0.25, max_hour=None, cycle='auto'):
    """
    Find the latest available GFS run, then download forecast hours at every
    step in the fine-grained schedule up to that cycle's max lead time.
    Returns (run_dt, results_dict) where results_dict = {fxx: (lats, lons, grid)}
    """
    now = datetime.now(timezone.utc)

    candidates = []
    if cycle != 'auto':
        target_hour = int(cycle)
        for days_back in range(3):
            dt = (now - timedelta(days=days_back)).replace(
                hour=target_hour, minute=0, second=0, microsecond=0)
            if dt <= now:
                candidates.append(dt)
    else:
        for days_back in range(2):
            for run_hour in [12, 0]:
                dt = (now - timedelta(days=days_back)).replace(
                    hour=run_hour, minute=0, second=0, microsecond=0)
                if dt <= now:
                    candidates.append(dt)

    run_dt = None
    print("  Finding latest available GFS run...")
    for candidate in candidates:
        try:
            print(f"    Trying {candidate.strftime('%Y-%m-%d %H UTC')} F024...")
            lats, lons, vals = fetch_gfs_single(candidate, 24, res)
            run_dt = candidate
            print(f"    Found! Run: {run_dt.strftime('%Y-%m-%d %H UTC')}")
            results = {24: (lats, lons, vals)}
            break
        except Exception as e:
            print(f"    {e} - skipping")
            continue

    if run_dt is None:
        raise RuntimeError("Could not find any available GFS run.")

    run_max        = max_hour or MODEL_MAX_HOURS['GFS'].get(run_dt.hour, 384)
    forecast_hours = fine_grained_steps(run_max, gfs_native_steps(run_max), FINE_CUTOFF_HOUR['GFS'])
    print(f"  [GFS] Run {run_dt.hour:02d}Z -> max {run_max}h -> "
          f"{len(forecast_hours)} steps ({FINE_CADENCE_HOURS}-hourly to "
          f"{min(FINE_CUTOFF_HOUR['GFS'], run_max)}h, then daily to Day {run_max // 24})")

    res_str  = '0p25' if res <= 0.25 else '0p50'
    date_str = run_dt.strftime('%Y%m%d')
    run_str  = f"{run_dt.hour:02d}"
    check_url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{res_str}.pl"
        f"?dir=%2Fgfs.{date_str}%2F{run_str}%2Fatmos"
        f"&file=gfs.t{run_str}z.pgrb2.{res_str}.f{run_max:03d}&var_APCP=on"
        f"&subregion=&leftlon=75&rightlon=80&toplat=25&bottomlat=20"
    )
    print(f"  [GFS] Checking if F{run_max:03d} (final step) is available...")
    try:
        cr = requests.get(check_url, timeout=30, stream=True)
        chunk = next(cr.iter_content(256), b'')
        cr.close()
        available = cr.status_code == 200 and chunk[:4] == b'GRIB'
    except Exception:
        available = False
    if not available:
        print(f"  [GFS] F{run_max:03d} not yet on server — run still uploading.")
        print(f"  [GFS] Exiting cleanly. Re-run this script once the run is complete.")
        return None, None
    print(f"  [GFS] F{run_max:03d} confirmed available — proceeding with full fetch.")

    prev_max = float(np.nanmax(vals))
    for fxx in forecast_hours:
        if fxx == 24:
            continue
        print(f"  Fetching F{fxx:03d} ({day_label(fxx)})...")
        try:
            lats, lons, vals = fetch_gfs_single(run_dt, fxx, res)
            results[fxx] = (lats, lons, vals)
            this_max = float(np.nanmax(vals))
            if this_max < prev_max - 0.5:
                print(f"    WARNING: F{fxx:03d} max ({this_max:.1f}mm) is LOWER "
                      f"than an earlier milestone's max ({prev_max:.1f}mm). A "
                      f"true running total can't decrease — may have picked up "
                      f"a bucket record. Treat these numbers with suspicion.")
            prev_max = max(prev_max, this_max)
            print(f"    OK — max: {this_max:.1f} mm")
        except Exception as e:
            print(f"    ERROR F{fxx:03d}: {e} — skipping")

    return run_dt, results


# ── ECMWF FETCH (all hours) ──────────────────────────────────────────────────

def fetch_ecmwf_all_hours(res=0.25, max_hour=None, cycle='auto'):
    """
    Download ECMWF IFS tp at every step the ACTUAL latest run supports.
    00Z/12Z cycles extend to 360h (15 days); 06Z/18Z stop at 144h (6 days).
    Pins every subsequent request to the exact run discovered via probe so
    a rolling-archive rotation mid-fetch can't shift later steps onto a
    different run.
    Returns (run_dt, results_dict)
    """
    try:
        from ecmwf.opendata import Client
    except ImportError:
        raise ImportError("Run: pip install ecmwf-opendata")

    client  = Client(source="ecmwf")
    results = {}

    print("  Finding latest available ECMWF run...")
    probe_path = os.path.join(tempfile.gettempdir(), "ecmwf_probe.grib2")
    try:
        if cycle != 'auto':
            target_hour = int(cycle)
            now = datetime.now(timezone.utc)
            run_dt = None
            for days_back in range(3):
                candidate_date = (now - timedelta(days=days_back)).strftime('%Y-%m-%d')
                try:
                    print(f"    Trying ECMWF {candidate_date} {target_hour:02d}Z F024...")
                    probe = client.retrieve(
                        date=candidate_date,
                        time=target_hour,
                        step=24, type="fc", param="tp",
                        area=[BBOX['north'], BBOX['west'], BBOX['south'], BBOX['east']],
                        target=probe_path)
                    run_dt = probe.datetime
                    print(f"    Found! Run: {run_dt.strftime('%Y-%m-%d %H UTC')}")
                    break
                except Exception as e:
                    print(f"    {candidate_date} {target_hour:02d}Z not available: {e}")
                    continue
            if run_dt is None:
                print("  [ECMWF] Could not find requested cycle — check date/availability.")
                return None, None
        else:
            probe = client.retrieve(
                step=24, type="fc", param="tp",
                area=[BBOX['north'], BBOX['west'], BBOX['south'], BBOX['east']],
                target=probe_path)
            run_dt = probe.datetime
            print(f"    Found! Run: {run_dt.strftime('%Y-%m-%d %H UTC')}")

        da = open_grib2_get_rainfall(probe_path)
        if da is None:
            raise RuntimeError("Could not extract tp from probe request")
        lats = da.latitude.values
        lons = da.longitude.values
        vals = da.values * 1000.0
        if vals.ndim == 3:
            vals = vals[-1]
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            vals = vals[::-1, :]
        vals = np.where(np.isnan(vals), 0.0, vals)
        results[24] = (lats, lons, vals)
    finally:
        try: os.unlink(probe_path)
        except: pass

    run_max        = max_hour or MODEL_MAX_HOURS['ECMWF'].get(run_dt.hour, 144)
    forecast_hours = fine_grained_steps(run_max, ecmwf_native_steps(run_max), FINE_CUTOFF_HOUR['ECMWF'])
    print(f"  [ECMWF] Latest run: {run_dt.strftime('%Y-%m-%d %H UTC')} -> "
          f"max {run_max}h -> {len(forecast_hours)} steps ({FINE_CADENCE_HOURS}-hourly "
          f"to {min(FINE_CUTOFF_HOUR['ECMWF'], run_max)}h, then daily to Day {run_max // 24})")

    print(f"  [ECMWF] Checking if F{run_max:03d} (final step) is available...")
    check_path = os.path.join(tempfile.gettempdir(), "ecmwf_avail_check.grib2")
    available  = False
    try:
        client.retrieve(
            date=run_dt.strftime('%Y-%m-%d'), time=run_dt.hour,
            step=run_max, type="fc", param="tp",
            area=[25, 75, 20, 80],
            target=check_path)
        available = os.path.exists(check_path) and os.path.getsize(check_path) > 100
    except Exception:
        available = False
    finally:
        try: os.unlink(check_path)
        except: pass
    if not available:
        print(f"  [ECMWF] F{run_max:03d} not yet on server — run still uploading.")
        print(f"  [ECMWF] Exiting cleanly. Re-run this script once the run is complete.")
        return None, None
    print(f"  [ECMWF] F{run_max:03d} confirmed available — proceeding with full fetch.")

    for fxx in forecast_hours:
        if fxx == 24:
            continue
        print(f"  Fetching ECMWF F{fxx:03d} ({day_label(fxx)})...")
        tmp_path = os.path.join(tempfile.gettempdir(), f"ecmwf_tp_f{fxx:03d}.grib2")
        try:
            client.retrieve(
                date=run_dt.strftime('%Y-%m-%d'), time=run_dt.hour,
                step=fxx, type="fc", param="tp",
                area=[BBOX['north'], BBOX['west'], BBOX['south'], BBOX['east']],
                target=tmp_path)

            da = open_grib2_get_rainfall(tmp_path)
            if da is None:
                raise RuntimeError("Could not extract tp")

            lats = da.latitude.values
            lons = da.longitude.values
            vals = da.values * 1000.0   # metres -> mm
            if vals.ndim == 3:
                vals = vals[-1]
            if lats[0] > lats[-1]:
                lats = lats[::-1]
                vals = vals[::-1, :]
            vals = np.where(np.isnan(vals), 0.0, vals)
            results[fxx] = (lats, lons, vals)
            print(f"    OK — max: {np.nanmax(vals):.1f} mm")

        except Exception as e:
            print(f"    ERROR F{fxx:03d}: {e} — skipping")
        finally:
            try: os.unlink(tmp_path)
            except: pass

    return run_dt, results


# ── ICON FETCH ────────────────────────────────────────────────────────────────

def fetch_icon_single(run_dt, forecast_hour):
    """
    Download ONE ICON Global tot_prec file from opendata.dwd.de.

    CONFIRMED filename pattern (from live server directory listing, July 2026):
      icon_global_icosahedral_single-level_{YYYYMMDDHH}_{FFF}_TOT_PREC.grib2.bz2

    IMPORTANT — ICOSAHEDRAL GRID:
      DWD ICON Global tot_prec is on the native icosahedral (triangular)
      unstructured grid, NOT regular-lat-lon. cfgrib reads it as 1D arrays
      of (lat, lon, value) scatter points rather than a 2D grid. We remap
      these scatter points onto our target regular 0.25-deg lat/lon grid
      using scipy's NearestNDInterpolator, which is fast and robust for
      this kind of unstructured→regular mapping.

    Files are bz2-compressed — decompressed in-memory before cfgrib reads them.
    Returns (lats_1d, lons_1d, grid_2d) on a regular 0.25-deg grid over BBOX.
    The value is the rain in THIS hour only (not cumulative).
    fetch_icon_all_hours() handles accumulation into running totals.
    """
    from scipy.interpolate import NearestNDInterpolator

    run_str  = f"{run_dt.hour:02d}"
    date_str = run_dt.strftime('%Y%m%d')
    fxx_str  = f"{forecast_hour:03d}"

    # Correct filename — icosahedral, not regular-lat-lon
    filename = (
        f"icon_global_icosahedral_single-level_"
        f"{date_str}{run_str}_{fxx_str}_TOT_PREC.grib2.bz2"
    )
    url = f"{ICON_BASE}/{run_str}/tot_prec/{filename}"

    r = requests.get(url, timeout=120, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {url}")

    compressed = r.content
    if len(compressed) < 200:
        raise RuntimeError(
            f"Response too small ({len(compressed)} bytes) — "
            f"probably an HTML error page, not a GRIB2 file")

    # Decompress bz2
    try:
        raw = bz2.decompress(compressed)
    except OSError as e:
        raise RuntimeError(f"bz2 decompress failed: {e}")

    if raw[:4] != b'GRIB':
        raise RuntimeError(
            f"Decompressed content is not GRIB2 (first 4 bytes: {raw[:4]!r})")

    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"icon_{date_str}{run_str}_f{fxx_str}.grib2"
    )
    try:
        with open(tmp_path, 'wb') as f:
            f.write(raw)

        da = open_grib2_get_rainfall(tmp_path)
        if da is None:
            raise RuntimeError("Could not extract tot_prec from GRIB2")

        # Icosahedral GRIB2 uses 'lat'/'lon' coordinate names, not
        # 'latitude'/'longitude' like regular-lat-lon grids. Try both.
        coords = list(da.coords)
        if 'latitude' in coords:
            src_lats = da.latitude.values.ravel()
            src_lons = da.longitude.values.ravel()
        elif 'lat' in coords:
            src_lats = da.lat.values.ravel()
            src_lons = da.lon.values.ravel()
        else:
            lat_coord = next((c for c in coords if 'lat' in c.lower()), None)
            lon_coord = next((c for c in coords if 'lon' in c.lower()), None)
            if lat_coord is None or lon_coord is None:
                raise RuntimeError(
                    f"Could not find lat/lon coordinates in GRIB2. "
                    f"Available coords: {coords}")
            src_lats = da[lat_coord].values.ravel()
            src_lons = da[lon_coord].values.ravel()
        src_vals = da.values.ravel()   # already in mm

        src_vals = np.where(np.isnan(src_vals), 0.0, src_vals)
        src_vals = np.clip(src_vals, 0.0, None)

        # Remap to regular 0.25-deg grid over BBOX using nearest-neighbour
        # NearestNDInterpolator is the right tool for unstructured→regular:
        # fast, no triangulation artifacts, handles sparse/dense regions well.
        out_res  = 0.25
        out_lats = np.arange(BBOX['south'], BBOX['north'] + out_res, out_res)
        out_lons = np.arange(BBOX['west'],  BBOX['east']  + out_res, out_res)
        out_lon2d, out_lat2d = np.meshgrid(out_lons, out_lats)

        interp   = NearestNDInterpolator(
            list(zip(src_lats, src_lons)), src_vals)
        out_grid = interp(out_lat2d, out_lon2d)
        out_grid = np.clip(out_grid, 0.0, None)

        return out_lats, out_lons, out_grid

    finally:
        try: os.unlink(tmp_path)
        except: pass


def fetch_icon_all_hours(max_hour=None, cycle='auto'):
    """
    Download all ICON tot_prec hourly increments and accumulate them into
    running totals — the same format GFS and ECMWF produce — so the rest
    of the pipeline works identically for all three models.

    WHY WE FETCH EVERY HOUR:
      ICON tot_prec[fxx] = rain in hour (fxx-1)→fxx only.
      To get total rain from run start to hour N we must sum every
      increment from hour 1 to hour N. Skipping any middle hour would
      make all subsequent totals wrong.

      Example for Day 1 (F024):
        total = inc[F001] + inc[F002] + ... + inc[F024]

      We fetch all 78 hourly steps + 3-hourly steps beyond that.
      Only steps in forecast_hours get stored in results — intermediate
      steps are used for accumulation then discarded.

    Returns (run_dt, results_dict)
      results_dict = {fxx: (lats, lons, running_total_grid)}
      — identical structure to fetch_gfs_all_hours() / fetch_ecmwf_all_hours().
    Returns (None, None) if no run is available yet.
    """
    now = datetime.now(timezone.utc)

    candidates = []
    if cycle != 'auto':
        target_hour = int(cycle)
        for days_back in range(3):
            dt = (now - timedelta(days=days_back)).replace(
                hour=target_hour, minute=0, second=0, microsecond=0)
            if dt <= now:
                candidates.append(dt)
    else:
        for days_back in range(2):
            for run_hour in [12, 0, 18, 6]:
                dt = (now - timedelta(days=days_back)).replace(
                    hour=run_hour, minute=0, second=0, microsecond=0)
                if dt <= now:
                    candidates.append(dt)

    # Probe with F001 — ICON files appear ~5-6h after init
    run_dt = None
    print("  Finding latest available ICON Global run...")
    for candidate in candidates:
        try:
            print(f"    Trying {candidate.strftime('%Y-%m-%d %H UTC')} F001...")
            lats, lons, inc = fetch_icon_single(candidate, 1)
            run_dt = candidate
            print(f"    Found! Run: {run_dt.strftime('%Y-%m-%d %H UTC')}")
            break
        except Exception as e:
            print(f"    {e} — skipping")
            continue

    if run_dt is None:
        print("  [ICON] Could not find any available ICON run.")
        return None, None

    run_max        = max_hour or MODEL_MAX_HOURS['ICON'].get(run_dt.hour, 120)
    native_steps   = icon_native_steps(run_max)
    forecast_hours = fine_grained_steps(
        run_max, native_steps, FINE_CUTOFF_HOUR['ICON'])

    total_native = len([h for h in native_steps if h > 0])
    print(f"  [ICON] Run {run_dt.hour:02d}Z → max {run_max}h → "
          f"{len(forecast_hours)} steps to store "
          f"(fetching all {total_native} native steps for accumulation)")

    # Availability check — probe final step before committing to full fetch
    print(f"  [ICON] Checking if F{run_max:03d} (final step) is available...")
    try:
        fetch_icon_single(run_dt, run_max)
        print(f"  [ICON] F{run_max:03d} confirmed — proceeding with full fetch.")
    except Exception as e:
        print(f"  [ICON] F{run_max:03d} not yet on server ({e})")
        print(f"  [ICON] Run still uploading — exiting cleanly, retry later.")
        return None, None

    # Accumulation loop — walk every native step in order
    running_total = None
    results       = {}
    forecast_set  = set(forecast_hours)
    all_steps     = sorted(h for h in native_steps if h > 0)

    for fxx in all_steps:
        if fxx > run_max:
            break

        is_store = fxx in forecast_set
        print(f"  Fetching ICON F{fxx:03d} ({day_label(fxx)})..."
              + (" [store]" if is_store else ""))

        try:
            lats, lons, increment = fetch_icon_single(run_dt, fxx)

            if running_total is None:
                running_total = increment.copy()
            else:
                running_total = running_total + increment

            if is_store:
                results[fxx] = (lats, lons, running_total.copy())
                print(f"    running total max: {np.nanmax(running_total):.1f} mm")

        except Exception as e:
            print(f"    ERROR F{fxx:03d}: {e} — skipping increment")
            if running_total is not None:
                print(f"    WARNING: running total may be underestimated "
                      f"beyond F{fxx:03d} due to missing increment")

    print(f"  [ICON] Done — {len(results)} steps stored.")
    return run_dt, results


# ── JSON GRID OUTPUT ──────────────────────────────────────────────────────────

def save_grid_json(lats, lons, grid, run_dt, model_key, fxx, out_path, kind='accum'):
    """
    Save one forecast hour as JSON grid for Leaflet.
    kind: 'accum' = running total from run start (0-fxx h)
          '24h'   = trailing 24h rainfall ending at this step
          '3h'    = trailing 3h rainfall ending at this step
    """
    valid_dt = run_dt + timedelta(hours=fxx)
    parameter = {
        'accum': "accumulated_rainfall_running_total",
        '24h':   "rainfall_24h_trailing",
        '3h':    "rainfall_3h_trailing",
    }.get(kind, kind)
    data = {
        "model":          model_key,
        "kind":           kind,
        "forecast_hour":  fxx,
        "day_label":      day_label(fxx),
        "run_time_utc":   run_dt.strftime('%Y-%m-%d %H:%M UTC'),
        "valid_time_utc": valid_dt.strftime('%Y-%m-%d %H:%M UTC'),
        "valid_date":     valid_dt.strftime('%d %b %Y'),
        "parameter":      parameter,
        "units":          "mm",
        "bbox":           BBOX,
        "lats":           [round(float(x), 2) for x in lats],
        "lons":           [round(float(x), 2) for x in lons],
        "max_mm":         round(float(np.nanmax(grid)), 1),
        "values":         [[round(float(v), 1) for v in row] for row in grid],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print(f"  JSON -> {out_path}  ({os.path.getsize(out_path)//1024} KB)")


# ── META JSON ─────────────────────────────────────────────────────────────────

def save_meta(model_results, model_incremental, model_fine, model_medium, out_dir):
    """
    Save meta.json listing all available models and forecast steps.
    Frontend reads this to build the day slider/timeline dynamically.
    """
    meta = {
        "generated_at_utc":  datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        "units":             "mm",
        "bbox":              BBOX,
        "fine_cadence_hours": FINE_CADENCE_HOURS,
        "models":            {},
    }

    for mk, (run_dt, results) in model_results.items():
        incremental = model_incremental.get(mk, {})
        fine        = model_fine.get(mk, {})
        medium      = model_medium.get(mk, {})
        fine_cutoff = FINE_CUTOFF_HOUR[mk]
        steps = []
        for fxx in sorted(results.keys()):
            lats, lons, accum_grid = results[fxx]
            valid_dt = run_dt + timedelta(hours=fxx)

            inc_max = None
            if fxx in incremental:
                _, _, inc_grid = incremental[fxx]
                inc_max = round(float(np.nanmax(inc_grid)), 1)

            fine_max = None
            if fxx in fine:
                _, _, fine_grid = fine[fxx]
                fine_max = round(float(np.nanmax(fine_grid)), 1)

            medium_max = None
            if fxx in medium:
                _, _, medium_grid = medium[fxx]
                medium_max = round(float(np.nanmax(medium_grid)), 1)

            step = {
                "forecast_hour":   fxx,
                "day_label":       day_label(fxx),
                "day_number":      fxx // 24 + (1 if fxx % 24 else 0),
                "is_day_milestone": fxx % 24 == 0,
                "resolution":      "fine" if fxx <= fine_cutoff else "medium",
                "valid_time_utc":  valid_dt.strftime('%Y-%m-%d %H:%M UTC'),
                "valid_date":      valid_dt.strftime('%d %b %Y'),
                "max_mm_accum":    round(float(np.nanmax(accum_grid)), 1),
                "max_mm_24h":      inc_max,
                "grid_json_accum": f"{mk.lower()}_f{fxx:03d}_accum_grid.json",
                "grid_json_24h":   f"{mk.lower()}_f{fxx:03d}_24h_grid.json",
            }
            if fxx in fine:
                step["max_mm_3h"]    = fine_max
                step["grid_json_3h"] = f"{mk.lower()}_f{fxx:03d}_3h_grid.json"
            if fxx in medium:
                step["max_mm_medium"]    = medium_max
                step["grid_json_medium"] = f"{mk.lower()}_f{fxx:03d}_medium_grid.json"
            steps.append(step)

        max_fxx = max(results.keys()) if results else 0
        meta["models"][mk] = {
            "run_time_utc":        run_dt.strftime('%Y-%m-%d %H:%M UTC'),
            "run_hour_utc":        run_dt.hour,
            "max_forecast_hour":   max_fxx,
            "num_days":            max_fxx // 24,
            "fine_cadence_hours":  FINE_CADENCE_HOURS,
            "medium_cadence_hours": MEDIUM_CADENCE_HOURS.get(mk),
            "fine_cutoff_hour":  min(FINE_CUTOFF_HOUR[mk], max_fxx),
            "steps":             steps,
        }

    path = os.path.join(out_dir, 'meta.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  meta -> {path}")


# ── INTERPOLATION ─────────────────────────────────────────────────────────────

def interpolate_to_hires(lats, lons, grid, hi_res=0.05):
    interp   = RegularGridInterpolator(
        (lats, lons), grid, method='linear',
        bounds_error=False, fill_value=0.0)
    hi_lats  = np.arange(BBOX['south'], BBOX['north'] + hi_res, hi_res)
    hi_lons  = np.arange(BBOX['west'],  BBOX['east']  + hi_res, hi_res)
    hi_lon2d, hi_lat2d = np.meshgrid(hi_lons, hi_lats)
    pts      = np.column_stack([hi_lat2d.ravel(), hi_lon2d.ravel()])
    hi_grid  = interp(pts).reshape(hi_lat2d.shape)
    return hi_lats, hi_lons, hi_lon2d, hi_lat2d, np.clip(hi_grid, 0, 500)


# ── PNG RENDERER ──────────────────────────────────────────────────────────────

IST_OFFSET = timedelta(hours=5, minutes=30)

def fmt_ist(dt_utc):
    ist = dt_utc + IST_OFFSET
    return f"{ist.strftime('%a %m/%d/%Y')}, {ist.strftime('%I:%M%p').lower()} IST"

def draw_model_logo(ax, model_key):
    """
    Draws the model's official logo top-left of the map panel if and only if
    a real logo file has been supplied at LOGO_DIR/LOGO_FILES[model_key].
    Never fabricates a logo — silently skips if file is missing.
    """
    filename = LOGO_FILES.get(model_key)
    if not filename:
        return
    logo_path = os.path.join(LOGO_DIR, filename)
    if not os.path.isfile(logo_path):
        print(f"  (no logo file at {logo_path} -- skipping logo badge, "
              f"text attribution only)")
        return
    try:
        logo_img = plt.imread(logo_path)
    except Exception as e:
        print(f"  (failed to read logo {logo_path}: {e} -- skipping)")
        return
    logo_ax = ax.inset_axes([0.015, 0.885, 0.16, 0.09])
    logo_ax.add_patch(mpatches.Rectangle(
        (0, 0), 1, 1, transform=logo_ax.transAxes,
        facecolor='#0d1b2a', edgecolor='#3b5b8c', linewidth=0.8,
        alpha=0.85, zorder=1))
    logo_ax.imshow(logo_img, zorder=2)
    logo_ax.axis('off')

_GUJ_BBOX = None

def _guj_bbox(gujarat_gdf, buffer=0.35):
    global _GUJ_BBOX
    if _GUJ_BBOX is None:
        b = gujarat_gdf.total_bounds  # [minx, miny, maxx, maxy]
        _GUJ_BBOX = (b[0]-buffer, b[2]+buffer, b[1]-buffer, b[3]+buffer)
    return _GUJ_BBOX


def _draw_map_base(ax, lats, lons, grid, india_composite_gdf, proj,
                   render_extent=None):
    """
    Shared base: ocean bg, neighbour land fill, India land fill, rainfall imshow.

    Layer order (bottom to top):
      z=0  ocean/sea background  (#f5f5f5 white)
      z=1  neighbour land fill   (white, no borders)
      z=2  India land fill       (india-composite.geojson, white)
      z=3  rainfall imshow       (on top of land fill)
    """
    import matplotlib.colors as mcolors
    import cartopy.feature as cfeature
    from scipy.interpolate import RegularGridInterpolator

    ax.set_facecolor('#f5f5f5')

    try:
        land_geoms = list(cfeature.LAND.with_scale('50m').geometries())
        if land_geoms:
            ax.add_geometries(land_geoms, crs=proj,
                              facecolor='#f5f5f5', edgecolor='none', zorder=1)
    except Exception as e:
        print(f"  WARNING: could not draw land feature (Natural Earth not cached yet?): {e}")

    ax.add_geometries(india_composite_gdf.geometry, crs=proj,
                      facecolor='#f5f5f5', edgecolor='none', zorder=2)

    if render_extent is None:
        render_extent = (BBOX['west'], BBOX['east'], BBOX['south'], BBOX['north'])
    west, east, south, north = render_extent

    span_lon = east - west
    span_lat = north - south
    fine_res = min(span_lon, span_lat) / 1000.0
    fine_res = max(fine_res, 0.008)

    fine_lats = np.arange(south, north + fine_res, fine_res)
    fine_lons = np.arange(west,  east  + fine_res, fine_res)
    interp    = RegularGridInterpolator(
        (lats, lons), grid, method='linear',
        bounds_error=False, fill_value=0.0)
    flo, fla  = np.meshgrid(fine_lons, fine_lats)
    fine_grid = np.clip(
        interp(np.column_stack([fla.ravel(), flo.ravel()])).reshape(fla.shape),
        0, LEVELS[-1])

    band_idx = np.clip(np.digitize(fine_grid, LEVELS[1:], right=False),
                       0, len(COLORS_FILL) - 1)
    rgba_lut  = np.array([mcolors.to_rgba(c) for c in COLORS_FILL])
    img_rgba  = rgba_lut[band_idx].copy()
    img_rgba[..., 3] *= 0.90

    ax.imshow(img_rgba,
              extent=[west, east, south, north],
              origin='lower', transform=proj, zorder=3,
              interpolation='bilinear', aspect='auto')
    return fine_grid


def render_png(lats, lons, grid, india_composite_gdf, india_states_gdf,
               gujarat_gdf, district_gdf,
               model_key, model_name, run_dt, fxx, max_forecast_hour,
               outdir):
    """
    Renders clean map-only PNGs per forecast step.
    All chrome (title, colorbar, metadata) lives in the frontend.

    Outputs in outdir/MODEL/:
      india_fxx{N:03d}.png             India full domain
      gujarat_plain_fxx{N:03d}.png     Gujarat tight, no sub-divisions
      gujarat_district_fxx{N:03d}.png  Gujarat + district borders + labels
    """
    from matplotlib import patheffects as pe
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import matplotlib.ticker as mticker

    proj   = ccrs.PlateCarree()
    mk_dir = os.path.join(outdir, model_key)
    os.makedirs(mk_dir, exist_ok=True)
    fstr   = f'{fxx:03d}'

    if gujarat_gdf is None and district_gdf is None:
        print('    (skipping Gujarat PNGs — no Gujarat GeoJSON supplied)')
        guj_west, guj_east, guj_south, guj_north = 68.0, 74.5, 20.0, 24.5
        gujarat_outline = None
    else:
        guj_west, guj_east, guj_south, guj_north = _guj_bbox(
            gujarat_gdf if gujarat_gdf is not None else district_gdf)
        try:
            gujarat_outline = gujarat_gdf.geometry.union_all() if gujarat_gdf is not None else None
        except Exception:
            gujarat_outline = None

    try:
        india_outline_geom = india_composite_gdf.geometry.union_all()
    except Exception:
        india_outline_geom = None

    def save_fig(fig, fname):
        path = os.path.join(mk_dir, fname)
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(),
                    bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        print(f'    PNG -> {path}')
        return path

    STROKE_LABEL = [pe.withStroke(linewidth=1.8, foreground='#0d1825')]

    def find_name_col(gdf, candidates):
        return next((c for c in candidates if c in gdf.columns), None)

    def draw_labels(a, gdf, col, fontsize=5.5):
        if col is None:
            return
        for _, row in gdf.iterrows():
            try:
                cen = row.geometry.centroid
                a.text(cen.x, cen.y, str(row[col]),
                       transform=proj, fontsize=fontsize, fontweight='bold',
                       color='#1a1a1a', ha='center', va='center',
                       zorder=12)
            except Exception:
                pass

    INDIA_EXTENT = (60, 100, 5, 40)

    IST       = timedelta(hours=5, minutes=30)
    valid_utc = run_dt + timedelta(hours=fxx)
    valid_ist = valid_utc + IST
    run_ist   = run_dt + IST

    def fmt_utc_range_short(dt):
        return f"{dt.strftime('%Hz')} {dt.day} {dt.strftime('%B')}"

    def fmt_ist_range_short(dt):
        return f"{dt.day} {dt.strftime('%B %I:%M')} {dt.strftime('%p').lower()}"

    def fmt_ist_init(dt):
        return f"{ordinal(dt.day)} {dt.strftime('%B %I:%M %p')}"

    def ordinal(n):
        s = {1:'st', 2:'nd', 3:'rd'}
        return str(n) + s.get(n if n < 20 else n % 10, 'th')

    # model_label shown in title
    model_label = {
        'GFS':   'GFS',
        'ECMWF': 'ECMWF IFS',
        'ICON':  'ICON Global (DWD)',
    }.get(model_key, model_key)

    line1 = (f"{model_label} Total Accumulated Precipitation (mm):  "
             f"Init:- {fmt_utc_range_short(run_dt)} {run_dt.year} (UTC)")
    line2 = (f"Forecast Hour: F{fxx:03d}  |  "
             f"From {fmt_utc_range_short(run_dt)} to "
             f"{fmt_utc_range_short(valid_utc)} {valid_utc.year} (UTC)")

    GU_MONTHS = {1:'જાન્યુઆરી',2:'ફેબ્રુઆરી',3:'માર્ચ',4:'એપ્રિલ',
                 5:'મે',6:'જૂન',7:'જુલાઈ',8:'ઓગસ્ટ',
                 9:'સપ્ટેમ્બર',10:'ઓક્ટોબર',11:'નવેમ્બર',12:'ડિસેમ્બર'}

    def tod_gu(dt):
        h = dt.hour
        if   5  <= h < 12: return 'સવારે'
        elif 12 <= h < 17: return 'બપોરે'
        elif 17 <= h < 20: return 'સાંજે'
        else:              return 'રાત્રે'

    def fmt_ist_gu(dt):
        h = dt.hour % 12 or 12
        m = dt.minute
        t = f"{h}:{m:02d}" if m else str(h)
        return f"{dt.day} {GU_MONTHS[dt.month]} {tod_gu(dt)} {t}"

    gu_init = fmt_ist_gu(run_ist)
    gu_from = fmt_ist_gu(run_ist)
    gu_to   = fmt_ist_gu(valid_ist)

    gu_line1_segs = [
        ('હવામાન મોડલ :-  ', False),
        (model_label,         True),
        ('  |  મોડલ આધાર સમય :-  ', False),
        (gu_init,             False),
    ]
    gu_line2 = f'કુલ સંચિત વરસાદ (મિલિમીટર) :-  {gu_from}  થી  {gu_to}  સુધી'

    def make_gujarati_strip(width_px, height_px, dpi_val):
        from PIL import Image, ImageDraw, ImageFont
        import urllib.request
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gu_path  = os.path.join(script_dir, 'NotoSansGujarati-Bold.ttf')
        lat_path = os.path.join(script_dir, 'NotoSans-Bold.ttf')

        FONT_URLS = {
            gu_path:  ('https://github.com/googlefonts/noto-fonts/raw/main/'
                       'hinted/ttf/NotoSansGujarati/NotoSansGujarati-Bold.ttf'),
            lat_path: ('https://github.com/googlefonts/noto-fonts/raw/main/'
                       'hinted/ttf/NotoSans/NotoSans-Bold.ttf'),
        }
        for path, url in FONT_URLS.items():
            if not os.path.exists(path):
                print(f"  Downloading font: {os.path.basename(path)} ...")
                try:
                    urllib.request.urlretrieve(url, path)
                    print(f"  Font saved: {path}")
                except Exception as e:
                    print(f"  WARNING: Could not download {os.path.basename(path)}: {e}")

        fs = int(22 * dpi_val / 96)
        try:
            try:
                font_gu  = ImageFont.truetype(gu_path,  fs, layout_engine=ImageFont.Layout.RAQM)
                font_lat = ImageFont.truetype(lat_path, fs, layout_engine=ImageFont.Layout.RAQM)
            except Exception:
                font_gu  = ImageFont.truetype(gu_path,  fs)
                font_lat = ImageFont.truetype(lat_path, fs)
        except Exception as e:
            print(f"  WARNING: Gujarati font unavailable ({e}) -- skipping bottom strip")
            return None

        bg = '#e8eef4'
        fg = '#111111'
        mx = int(28 * dpi_val / 96)
        my = int(height_px * 0.10)
        lg = int(height_px * 0.50)
        img  = Image.new('RGB', (width_px, height_px), color=bg)
        draw = ImageDraw.Draw(img)
        draw.line([(0, 0), (width_px, 0)], fill='#b8c8d8', width=3)
        x = mx
        for text, is_latin in gu_line1_segs:
            font = font_lat if is_latin else font_gu
            bbox = font.getbbox(text)
            draw.text((x, my), text, font=font, fill=fg)
            x += bbox[2] - bbox[0]
        draw.text((mx, my + lg), gu_line2, font=font_gu, fill=fg)
        return img

    import matplotlib.colors as mcolors2

    def add_title_and_colorbar(f, map_ax, cbar_left=0.895):
        DPI_VAL = 150

        cbar_ax = f.add_axes([cbar_left, 0.145, 0.025, 0.755])
        cmap_cb = mcolors2.ListedColormap(COLORS_FILL)
        norm_cb = mcolors2.BoundaryNorm(LEVELS, ncolors=len(COLORS_FILL))
        sm = plt.cm.ScalarMappable(cmap=cmap_cb, norm=norm_cb)
        sm.set_array([])
        cb = f.colorbar(sm, cax=cbar_ax, extend='max')
        cb.set_ticks([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
                      125, 150, 175, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000])
        cb.ax.set_yticklabels(['0','5','10','15','20','25','30','40','50','60',
                                '70','80','90','100','125','150','175','200',
                                '250','300','400','500','600','700','800','900','1000+'],
                               fontsize=8.5, color='#ffffff', fontweight='bold')
        cb.ax.tick_params(colors='#ffffff', length=2)
        cb.ax.set_ylabel('Rainfall (mm)', fontsize=9.5, labelpad=7,
                         color='#ffffff', fontweight='bold')
        cb.outline.set_linewidth(0.5)

        title_bg = f.add_axes([0.0, 0.905, 1.0, 0.095])
        title_bg.set_facecolor('#ffffff')
        title_bg.set_xticks([]); title_bg.set_yticks([])
        for sp in title_bg.spines.values(): sp.set_visible(False)

        wm = f.add_axes([0.0, 0.960, 1.0, 0.040])
        wm.set_facecolor('#e8eef4')
        wm.set_xticks([]); wm.set_yticks([])
        for sp in wm.spines.values(): sp.set_visible(False)
        wm.text(0.008, 0.45, 'Map Generated by Ankit Patel',
                transform=wm.transAxes,
                fontsize=15, fontweight='bold', color='#111111', va='center')
        wm.text(0.992, 0.45, 'gujaratweatherman.com',
                transform=wm.transAxes,
                fontsize=15, fontweight='bold', color='#1a6a9a',
                va='center', ha='right')

        f.text(0.008, 0.952, line1,
               fontsize=12, fontweight='bold', color='#111111', va='top')
        f.text(0.008, 0.924, line2,
               fontsize=12, fontweight='bold', color='#333333', va='top')

        fig_w_px     = int(f.get_figwidth()  * DPI_VAL)
        fig_h_px     = int(f.get_figheight() * DPI_VAL)
        strip_h_frac = 0.10
        strip_px_h   = int(fig_h_px * strip_h_frac)
        strip_img    = make_gujarati_strip(fig_w_px, strip_px_h, DPI_VAL)
        if strip_img is not None:
            bot_ax = f.add_axes([0.0, 0.0, 1.0, strip_h_frac])
            bot_ax.imshow(np.array(strip_img), aspect='auto', origin='upper',
                          extent=[0,1,0,1], transform=bot_ax.transAxes, zorder=1)
            bot_ax.set_xlim(0,1); bot_ax.set_ylim(0,1)
            bot_ax.set_xticks([]); bot_ax.set_yticks([])
            for sp in bot_ax.spines.values(): sp.set_visible(False)

    # ── 1. INDIA — full extent ─────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 12))
    fig.patch.set_facecolor('#0d1825')
    ax  = fig.add_axes([0.05, 0.145, 0.83, 0.755], projection=proj)
    ax.set_extent(list(INDIA_EXTENT), crs=proj)

    _draw_map_base(ax, lats, lons, grid, india_composite_gdf, proj,
                   render_extent=INDIA_EXTENT)

    for feat, lw in [
        (cfeature.BORDERS.with_scale('50m'),   0.9),
        (cfeature.COASTLINE.with_scale('50m'), 0.9),
    ]:
        try:
            geoms = list(feat.geometries())
            if geoms:
                ax.add_geometries(geoms, crs=proj,
                                  facecolor='none', edgecolor='#555555',
                                  linewidth=lw, zorder=5)
        except Exception as e:
            print(f"  WARNING: {feat} unavailable: {e}")

    ax.add_geometries(india_states_gdf.geometry, crs=proj,
                      facecolor='none', edgecolor='#2a2a2a',
                      linewidth=0.7, zorder=7)

    if india_outline_geom:
        ax.add_geometries([india_outline_geom], crs=proj,
                          facecolor='none', edgecolor='#000000',
                          linewidth=1.8, zorder=9)
    else:
        ax.add_geometries(india_composite_gdf.geometry, crs=proj,
                          facecolor='none', edgecolor='#000000',
                          linewidth=1.8, zorder=9)

    gl = ax.gridlines(crs=proj, draw_labels=True,
                      linewidth=0.7, color='#ffffff',
                      alpha=0.65, linestyle='--', zorder=10)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([60, 65, 70, 75, 80, 85, 90, 95, 100])
    gl.ylocator = mticker.FixedLocator([5, 10, 15, 20, 25, 30, 35, 40])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8, 'color': '#ffffff', 'fontweight': 'bold'}
    gl.ylabel_style = {'size': 8, 'color': '#ffffff', 'fontweight': 'bold'}
    for sp in ax.spines.values(): sp.set_visible(False)

    add_title_and_colorbar(fig, ax)
    save_fig(fig, f'india_fxx{fstr}.png')

    # ── Gujarat base figure helper ─────────────────────────────────────────
    def guj_base_fig():
        _ref_gdf = gujarat_gdf if gujarat_gdf is not None else district_gdf
        _w, _e, _s, _n = (_guj_bbox(_ref_gdf) if _ref_gdf is not None
                           else (guj_west, guj_east, guj_south, guj_north))
        f = plt.figure(figsize=(12, 11))
        f.patch.set_facecolor('#0d1825')
        a = f.add_axes([0.06, 0.145, 0.82, 0.755], projection=proj)
        a.set_extent([_w, _e, _s, _n], crs=proj)
        _draw_map_base(a, lats, lons, grid, india_composite_gdf, proj,
                       render_extent=(_w, _e, _s, _n))
        a.add_geometries(india_states_gdf.geometry, crs=proj,
                         facecolor='none', edgecolor='#aaaaaa',
                         linewidth=0.5, zorder=5)
        if india_outline_geom:
            a.add_geometries([india_outline_geom], crs=proj,
                             facecolor='none', edgecolor='#000000',
                             linewidth=1.0, zorder=6)
        import math
        lon_ticks = list(range(int(math.floor(_w)), int(math.ceil(_e)) + 1))
        lat_ticks = list(range(int(math.floor(_s)), int(math.ceil(_n)) + 1))
        gl_g = a.gridlines(crs=proj, draw_labels=True,
                            linewidth=0.5, color='#ffffff',
                            alpha=0.6, linestyle='--', zorder=10)
        gl_g.top_labels   = False
        gl_g.right_labels = False
        gl_g.xlocator  = mticker.FixedLocator(lon_ticks)
        gl_g.ylocator  = mticker.FixedLocator(lat_ticks)
        gl_g.xformatter = LONGITUDE_FORMATTER
        gl_g.yformatter = LATITUDE_FORMATTER
        gl_g.xlabel_style = {'size': 7, 'color': '#ffffff', 'fontweight': 'bold'}
        gl_g.ylabel_style = {'size': 7, 'color': '#ffffff', 'fontweight': 'bold'}
        for sp in a.spines.values(): sp.set_visible(False)
        add_title_and_colorbar(f, a)
        return f, a

    # ── 2. GUJARAT PLAIN ──────────────────────────────────────────────────
    fig, ax = guj_base_fig()
    save_fig(fig, f'gujarat_plain_fxx{fstr}.png')

    if district_gdf is None:
        print('    (skipping district/taluka PNGs — no district GeoJSON supplied)')
        return

    dist_col = find_name_col(district_gdf,
        ('DISTRICT','NAME','district','name','DIST_NAME','DT_CEN_CD'))

    # ── 3. GUJARAT + DISTRICTS ────────────────────────────────────────────
    fig, ax = guj_base_fig()
    ax.add_geometries(district_gdf.geometry, crs=proj,
                      facecolor='none', edgecolor='#1a1a1a',
                      linewidth=0.85, zorder=7)
    draw_labels(ax, district_gdf, dist_col, fontsize=5.8)
    save_fig(fig, f'gujarat_district_fxx{fstr}.png')

    # ── 4. GUJARAT + TALUKAS ─────────────────────────────────────────────
    # NOTE: Taluka (sub-district) PNG rendering is currently disabled.
    # The taluka GeoJSON and rendering code are preserved here so this can
    # be re-enabled in future by uncommenting the block below.
    #
    # taluka_col = find_name_col(gujarat_gdf,
    #     ('TALUKA','taluka','NAME','name','SUB_DIST','SUBDIST','TAL_NAME'))
    # fig, ax = guj_base_fig()
    # ax.add_geometries(gujarat_gdf.geometry, crs=proj,
    #                   facecolor='none', edgecolor='#1a1a1a',
    #                   linewidth=0.60, zorder=6)
    # draw_labels(ax, gujarat_gdf, taluka_col, fontsize=4.5)
    # save_fig(fig, f'gujarat_taluka_fxx{fstr}.png')


# ── PNG META JSON ─────────────────────────────────────────────────────────────

def save_png_meta(model_results, out_dir, existing_meta=None):
    """
    Write meta.json for the frontend viewer.
    Preserves models from existing meta.json that were NOT updated this run
    so a single model failure doesn't wipe other models from the frontend.
    """
    meta = {
        "generated_at_utc":   datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        "units":              "mm",
        "bbox":               BBOX,
        "fine_cadence_hours": FINE_CADENCE_HOURS,
        "models":             {},
    }

    if existing_meta and 'models' in existing_meta:
        for mk, model_data in existing_meta['models'].items():
            if mk not in model_results:
                meta['models'][mk] = model_data
                print(f"  [META] Preserved existing {mk} entry (not updated this run)")

    for mk, (run_dt, results) in model_results.items():
        daily_fxx = sorted(fxx for fxx in results if fxx % 24 == 0 and fxx > 0)
        steps = []
        for fxx in daily_fxx:
            valid_dt = run_dt + timedelta(hours=fxx)
            steps.append({
                "forecast_hour":    fxx,
                "day_label":        day_label(fxx),
                "day_number":       fxx // 24,
                "is_day_milestone": True,
                "resolution":       "medium" if fxx > FINE_CUTOFF_HOUR.get(mk, 240) else "fine",
                "valid_time_utc":   valid_dt.strftime('%Y-%m-%d %H:%M UTC'),
                "valid_date":       valid_dt.strftime('%d %b %Y'),
                "max_mm_accum":     round(float(np.nanmax(results[fxx][2])), 1),
                "max_mm_24h":       None,
                "png_india":            f"{mk}/india_fxx{fxx:03d}.png",
                "png_gujarat_plain":    f"{mk}/gujarat_plain_fxx{fxx:03d}.png",
                "png_gujarat_district": f"{mk}/gujarat_district_fxx{fxx:03d}.png",
            })

        max_fxx = max(daily_fxx) if daily_fxx else 0
        meta["models"][mk] = {
            "run_time_utc":       run_dt.strftime('%Y-%m-%d %H:%M UTC'),
            "run_hour_utc":       run_dt.hour,
            "max_forecast_hour":  max_fxx,
            "num_days":           max_fxx // 24,
            "fine_cadence_hours": FINE_CADENCE_HOURS,
            "medium_cadence_hours": MEDIUM_CADENCE_HOURS.get(mk),
            "fine_cutoff_hour":   min(FINE_CUTOFF_HOUR.get(mk, 240), max_fxx),
            "steps":              steps,
        }

    path = os.path.join(out_dir, 'meta.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json -> {path}  ({len(meta['models'])} model(s): {list(meta['models'].keys())})")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Dynamic-range running total rainfall forecast (GFS + ECMWF + ICON)')
    parser.add_argument('--geojson', default=None,
                        help='Path to gujarat_taluka_clean.geojson (taluka boundaries, '
                             'currently unused — taluka rendering is disabled)')
    parser.add_argument('--district-geojson', default=None,
                        help='Path to gujarat_district_clean.geojson (district boundaries)')
    parser.add_argument('--india-geojson',        default=INDIA_COMPOSITE_DEFAULT,
                        help='Path to india-composite.geojson')
    parser.add_argument('--india-states-geojson', default=INDIA_STATES_DEFAULT,
                        help='Path to india_state.geojson')
    parser.add_argument('--model',
                        choices=['GFS', 'ECMWF', 'ICON', 'both', 'all'],
                        default='both',
                        help=(
                            "Which model(s) to fetch.\n"
                            "  both = GFS + ECMWF (existing behaviour, default)\n"
                            "  all  = GFS + ECMWF + ICON\n"
                            "  ICON = ICON only (separate workflow job recommended)"
                        ))
    parser.add_argument('--cycle', default='auto',
                        help='Model cycle UTC hour: auto, 00, 06, 12, 18. '
                             'auto = detect latest.')
    parser.add_argument('--res',    type=float, default=0.25,
                        help='Grid resolution for GFS/ECMWF fetch (ICON is always 0.1 deg)')
    parser.add_argument('--outdir', default=OUT_DIR)
    parser.add_argument('--no-png', action='store_true',
                        help='Skip PNG rendering (JSON only, much faster)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Expand model list
    if args.model == 'both':
        models_to_fetch = ['GFS', 'ECMWF']
    elif args.model == 'all':
        models_to_fetch = ['GFS', 'ECMWF', 'ICON']
    else:
        models_to_fetch = [args.model]

    print("\n[SETUP] Loading geographic data...")
    gujarat_gdf = None
    if args.geojson and os.path.exists(args.geojson):
        gujarat_gdf = gpd.read_file(args.geojson)
        print(f"  Gujarat talukas: {len(gujarat_gdf)} features (disabled, loaded for future use)")
    district_gdf = None
    if args.district_geojson:
        if not os.path.exists(args.district_geojson):
            print(f"  WARNING: district GeoJSON not found: {args.district_geojson} — "
                  f"district/taluka PNGs will be skipped")
        else:
            district_gdf = gpd.read_file(args.district_geojson)
            print(f"  Gujarat districts: {len(district_gdf)} features")
    india_composite_gdf = None
    india_states_gdf    = None
    if not args.no_png:
        india_composite_gdf, india_states_gdf = load_india_geodataframes(
            args.india_geojson, args.india_states_geojson)

    model_results = {}

    for mk in models_to_fetch:
        print(f"\n[{mk}] Fetching forecast for PNG rendering...")
        try:
            if mk == 'GFS':
                run_dt, results = fetch_gfs_all_hours(res=args.res, cycle=args.cycle)
            elif mk == 'ECMWF':
                run_dt, results = fetch_ecmwf_all_hours(res=args.res, cycle=args.cycle)
            elif mk == 'ICON':
                run_dt, results = fetch_icon_all_hours(cycle=args.cycle)

            if run_dt is None or results is None:
                print(f"  [{mk}] Run not yet complete — skipping.")
                continue

            written_fxx = set(results.keys())

            if not args.no_png:
                print(f"\n[{mk}] Rendering PNGs (accumulated view, daily milestones only)...")
                label = {
                    'GFS':   'GFS 0.25 deg (NOAA)',
                    'ECMWF': 'ECMWF IFS HRES 0.25 deg',
                    'ICON':  'ICON Global 0.1 deg (DWD)',
                }.get(mk, mk)
                run_max_hour = max(written_fxx) if written_fxx else 0
                for fxx in sorted(written_fxx):
                    if fxx % 24 != 0:
                        continue
                    try:
                        lats, lons, grid = results[fxx]
                        render_png(lats, lons, grid,
                                   india_composite_gdf, india_states_gdf,
                                   gujarat_gdf, district_gdf,
                                   mk, label, run_dt, fxx, run_max_hour,
                                   args.outdir)
                    except Exception as e:
                        print(f"  WARNING [{mk} F{fxx:03d}]: PNG render failed, skipping: {e}")

            if written_fxx:
                model_results[mk] = (run_dt, {fxx: results[fxx] for fxx in written_fxx})
            else:
                print(f"  [{mk}] No steps written — model absent from output.")

        except Exception as e:
            print(f"  ERROR [{mk}]: {e} — other model(s) are unaffected.")
            import traceback; traceback.print_exc()

    if model_results:
        print(f"\n[META] Writing meta.json for frontend...")
        existing_meta_path = os.path.join(args.outdir, 'existing_meta.json')
        existing_meta = None
        if os.path.exists(existing_meta_path):
            try:
                with open(existing_meta_path) as f:
                    existing_meta = json.load(f)
                print(f"  [META] Found existing meta.json — preserving models not updated this run")
            except Exception as e:
                print(f"  [META] Could not read existing meta.json: {e}")
        save_png_meta(model_results, args.outdir, existing_meta=existing_meta)
        print(f"\n[DONE] PNG rendering complete for: {list(model_results.keys())}")
    else:
        print("\n[DONE] No model produced PNG output.")

    print(f"\nDone! Output in: {args.outdir}/")
    total_kb = 0
    for fn in sorted(os.listdir(args.outdir)):
        sz = os.path.getsize(os.path.join(args.outdir, fn))
        total_kb += sz // 1024
        print(f"   {fn:<48} {sz//1024:>5} KB")
    print(f"   {'TOTAL':<48} {total_kb:>5} KB")

if __name__ == '__main__':
    main()
