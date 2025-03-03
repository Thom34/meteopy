#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script optimisé pour générer des cartes météorologiques interpolées 
avec une meilleure colorisation des départements et des performances améliorées.
"""

import os
import sys
import glob
import argparse
import math
import logging
import concurrent.futures
import functools
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Optional

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from tqdm import tqdm

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constantes globales
VAR_LABELS_FR = {
    'TN': "Température minimale (°C)", 
    'TX': "Température maximale (°C)", 
    'RR': "Précipitations (mm)", 
    'INST': "Indice d'ensoleillement (h)"
}
VALID_VARIABLES = set(VAR_LABELS_FR.keys())
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHAPEFILE_PATH = os.path.join(SCRIPT_DIR, "shapefiles", "departements-2019.shp")
CSV_DIR = os.path.join(SCRIPT_DIR, "csv_meteo")
DEFAULT_MIN_COVERAGE = 95.0
DEFAULT_GRID_RESOLUTION = (150, 110)

# Cache global pour éviter les rechargements
_gdf_cache = None
_dept_adjacency_cache = None

# --- Fonctions de vérification et chargement des données géographiques ---
def check_critical_paths() -> bool:
    """Vérifie l'existence des chemins critiques pour le fonctionnement du script."""
    paths_to_check = {"Répertoire CSV": CSV_DIR, "Shapefile des départements": SHAPEFILE_PATH}
    all_ok = True
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            logging.error(f"{name} introuvable: {path}")
            all_ok = False
    if not all_ok:
        logging.error("Chemins critiques manquants. Vérifiez votre installation.")
    return all_ok

def get_gdf_cached():
    """Charge et met en cache le GeoDataFrame à partir du shapefile."""
    global _gdf_cache
    if _gdf_cache is None:
        logging.info(f"Chargement du shapefile: {SHAPEFILE_PATH}")
        _gdf_cache = gpd.read_file(SHAPEFILE_PATH).to_crs("EPSG:4326")
        # Correction des géométries invalides
        _gdf_cache["geometry"] = _gdf_cache["geometry"].buffer(0)
    return _gdf_cache

def build_dept_adjacency(shapefile_path=SHAPEFILE_PATH):
    """
    Construit un dictionnaire d'adjacence des départements.
    """
    global _dept_adjacency_cache
    if _dept_adjacency_cache is not None:
        return _dept_adjacency_cache
    if not os.path.isfile(shapefile_path):
        logging.error(f"Shapefile introuvable : {shapefile_path}")
        return None
    try:
        gdf = get_gdf_cached()
        dept_polygons = {row["INSEE_DEP"]: row["geometry"] for _, row in gdf.iterrows()}
        neighbors = {code: set() for code in dept_polygons}
        dept_codes = list(dept_polygons.keys())
        with tqdm(total=len(dept_codes), desc="Calcul des adjacences départementales", unit="dept") as pbar:
            for i, dept1 in enumerate(dept_codes):
                for dept2 in dept_codes[i + 1:]:
                    if dept_polygons[dept1].touches(dept_polygons[dept2]):
                        neighbors[dept1].add(dept2)
                        neighbors[dept2].add(dept1)
                pbar.update(1)
        logging.info(f"Adjacence des départements construite pour {len(neighbors)} départements")
        _dept_adjacency_cache = neighbors
        return neighbors
    except Exception as e:
        logging.error(f"Erreur lors de la construction de l'adjacence : {e}")
        return None

def find_neighbors_n_steps(neighbors_dict, start_depts, n_steps=1):
    """
    Recherche BFS des départements à <= n bonds.
    """
    if not neighbors_dict or n_steps < 0:
        return set(start_depts)
    visited = set()
    queue = deque((dept, 0) for dept in start_depts if dept in neighbors_dict)
    while queue:
        current_dept, distance = queue.popleft()
        if current_dept in visited:
            continue
        visited.add(current_dept)
        if distance < n_steps:
            for neighbor in neighbors_dict.get(current_dept, set()):
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
    return visited

# --- Fonctions de parsing et chargement des CSV ---
def parse_date_range(date_str: str):
    """
    Analyse une chaîne de dates pour retourner un intervalle (start, end).
    """
    try:
        if not date_str:
            raise ValueError("Chaîne de date vide")
        if '-' in date_str:
            start_str, end_str = date_str.split('-')
            if len(start_str) == 8 and len(end_str) == 8 and start_str.isdigit() and end_str.isdigit():
                start = datetime.strptime(start_str, "%Y%m%d")
                end = datetime.strptime(end_str, "%Y%m%d")
            elif len(start_str) == 6 and len(end_str) == 6 and start_str.isdigit() and end_str.isdigit():
                year1, month1 = int(start_str[:4]), int(start_str[4:6])
                year2, month2 = int(end_str[:4]), int(end_str[4:6])
                if not (1 <= month1 <= 12 and 1 <= month2 <= 12):
                    raise ValueError(f"Mois invalide : {month1} ou {month2}")
                start = datetime(year1, month1, 1)
                end = datetime(year2, month2 + 1, 1) - timedelta(days=1) if month2 < 12 else datetime(year2 + 1, 1, 1) - timedelta(days=1)
            else:
                raise ValueError(f"Format de plage invalide : {date_str}")
        elif len(date_str) == 8 and date_str.isdigit():
            start = end = datetime.strptime(date_str, "%Y%m%d")
        elif len(date_str) == 6 and date_str.isdigit():
            year, month = int(date_str[:4]), int(date_str[4:6])
            if not (1 <= month <= 12):
                raise ValueError(f"Mois invalide : {month}")
            start = datetime(year, month, 1)
            end = datetime(year, month + 1, 1) - timedelta(days=1) if month < 12 else datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            raise ValueError(f"Format de date invalide : {date_str}")
        if start > end:
            start, end = end, start
            logging.warning("Dates inversées dans la plage, elles ont été réorganisées")
        return start, end
    except ValueError as e:
        raise ValueError(f"Erreur dans le parsing de la date '{date_str}' : {e}")

def get_csv_files(dept_codes, var_code, start_year, end_year, csv_dir=CSV_DIR):
    """
    Obtient la liste des fichiers CSV correspondant aux critères.
    """
    var_code = var_code.upper()
    suffix = "autres-parametres.csv" if var_code == 'INST' else "RR-T-Vent.csv"
    file_patterns = []
    read_previous = start_year <= 2023
    read_latest = end_year >= 2024
    if "FR" in dept_codes:
        if read_previous:
            file_patterns.append(os.path.join(csv_dir, f"Q_*_previous-1950-2023_{suffix}"))
        if read_latest:
            file_patterns.append(os.path.join(csv_dir, f"Q_*_latest-2024-2025_{suffix}"))
    else:
        for dept in dept_codes:
            dept = dept.zfill(2)
            if read_previous:
                file_patterns.append(os.path.join(csv_dir, f"Q_{dept}_previous-1950-2023_{suffix}"))
            if read_latest:
                file_patterns.append(os.path.join(csv_dir, f"Q_{dept}_latest-2024-2025_{suffix}"))
    all_files = []
    for pattern in file_patterns:
        files = glob.glob(pattern)
        all_files.extend(files)
        logging.info(f"Pattern {pattern} : {len(files)} fichiers trouvés")
    return all_files

def read_csv_file(filepath, var_code):
    """
    Lit un fichier CSV et effectue les transformations nécessaires.
    """
    usecols = ['AAAAMMJJ', 'LON', 'LAT', 'NOM_USUEL', 'NUM_POSTE', var_code]
    try:
        dtype = {'NUM_POSTE': str}
        df = pd.read_csv(filepath, sep=';', usecols=usecols, dtype=dtype)
        if var_code not in df.columns or 'AAAAMMJJ' not in df.columns:
            logging.warning(f"Colonnes manquantes dans {filepath}")
            return pd.DataFrame()
        df['AAAAMMJJ'] = df['AAAAMMJJ'].astype(str).str.zfill(8)
        df[var_code] = pd.to_numeric(df[var_code], errors='coerce')
        if var_code in ['FXI', 'FXY', 'FFM']:
            df[var_code] *= 3.6  # Conversion en km/h
        elif var_code == 'INST':
            df[var_code] /= 60.0  # Conversion en heures
        return df
    except Exception as e:
        logging.warning(f"Impossible de lire {filepath} : {e}")
        return pd.DataFrame()

def load_csv_for_var(dept_codes, var_code, start_date, end_date, csv_dir=CSV_DIR, max_workers=4):
    """
    Charge les fichiers CSV pertinents pour une variable et une période donnée.
    """
    var_code = var_code.upper()
    all_files = get_csv_files(dept_codes, var_code, start_date.year, end_date.year, csv_dir)
    if not all_files:
        logging.warning(f"Aucun fichier CSV trouvé pour {var_code}")
        return pd.DataFrame()
    dataframes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        read_func = functools.partial(read_csv_file, var_code=var_code)
        future_to_file = {executor.submit(read_func, filepath): filepath for filepath in all_files}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                           total=len(all_files), 
                           desc=f"Lecture des fichiers CSV pour {var_code}",
                           unit="fichier"):
            filepath = future_to_file[future]
            try:
                df = future.result()
                if not df.empty:
                    dataframes.append(df)
            except Exception as e:
                logging.error(f"Échec lors du traitement de {filepath}: {e}")
    if not dataframes:
        logging.warning(f"Aucune donnée valide trouvée pour {var_code}")
        return pd.DataFrame()
    combined_df = pd.concat(dataframes, ignore_index=True)
    start_str, end_str = start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')
    mask = (combined_df['AAAAMMJJ'] >= start_str) & (combined_df['AAAAMMJJ'] <= end_str)
    filtered_df = combined_df[mask]
    logging.info(f"Données chargées pour {var_code}: {len(filtered_df)} lignes")
    return filtered_df

# --- Fonctions de traitement des données ---
def extract_dept(station_id) -> str:
    """
    Extrait le code du département à partir d'un code de station météo.
    """
    station_id = str(station_id)
    if len(station_id) == 8:
        return station_id[:2]
    elif len(station_id) == 7:
        if station_id.startswith(('7', '8', '9')) and '1' <= station_id[1] <= '9':
            return '0' + station_id[1]
    if station_id.startswith('01'): return '01'
    elif station_id.startswith('02'): return '02'
    elif station_id.startswith('03'): return '03'
    elif station_id.startswith('04'): return '04'
    elif station_id.startswith('05'): return '05'
    elif station_id.startswith('06'): return '06'
    elif station_id.startswith('07'): return '07'
    elif station_id.startswith('08'): return '08'
    elif station_id.startswith('09'): return '09'
    if len(station_id) >= 2 and station_id[:2].isdigit():
        return station_id[:2]
    return '??'

def compute_var_stats(var_code, daily_values):
    """
    Calcule les statistiques pour une variable.
    """
    values = [v[1] for v in daily_values]
    if not values:
        return 0.0, 0.0, "", "max", "moy"
    def find_date_of_value(target_value):
        best_date = ""
        for date, val in daily_values:
            if abs(val - target_value) < 1e-9 and date.isdigit():
                if not best_date or date < best_date:
                    best_date = date
        return best_date
    var_code = var_code.upper()
    if var_code in ['RR', 'INST']:
        total = sum(values)
        extreme = max(values)
        return total, extreme, find_date_of_value(extreme), "max daily", "cumul"
    elif var_code == 'TN':
        extreme = min(values)
        mean = sum(values) / len(values)
        return mean, extreme, find_date_of_value(extreme), "min", "moy"
    else:
        extreme = max(values)
        mean = sum(values) / len(values)
        return mean, extreme, find_date_of_value(extreme), "max", "moy"

def collect_station_data(subset, var_code, total_days):
    """
    Extrait les données par station dans le DataFrame filtré.
    """
    station_data = defaultdict(list)
    station_depts = {}
    for _, row in subset.iterrows():
        if pd.isna(row[var_code]):
            continue
        key = (row['LON'], row['LAT'], row['NOM_USUEL'])
        date = str(row['AAAAMMJJ']).zfill(8)
        value = float(row[var_code])
        station_data[key].append((date, value))
        station_depts[key] = extract_dept(row.get('NUM_POSTE', '??'))
    return station_data, station_depts

def aggregate_period(df, start_date, end_date, var_code):
    """
    Agrège les données sur une période pour chaque station.
    """
    if df.empty:
        return []
    start_str, end_str = start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')
    subset = df[(df['AAAAMMJJ'] >= start_str) & (df['AAAAMMJJ'] <= end_str)].copy()
    if subset.empty:
        logging.warning(f"Aucune donnée pour la période {start_str} à {end_str}")
        return []
    total_days = (end_date - start_date).days + 1
    required_cols = {'LON', 'LAT', 'NOM_USUEL'}
    if not required_cols.issubset(subset.columns):
        logging.error(f"Colonnes requises manquantes : {required_cols - set(subset.columns)}")
        return []
    station_data, station_depts = collect_station_data(subset, var_code, total_days)
    results = []
    for station, values in station_data.items():
        coverage = (len(set(v[0] for v in values)) / total_days) * 100
        val_main, val_ext, date_ext, label_ext, aggregator = compute_var_stats(var_code, values)
        results.append((
            station[0],      # longitude
            station[1],      # latitude
            station[2],      # nom de la station
            val_main,        # valeur principale
            val_ext,         # valeur extrême
            date_ext,        # date de l'extrême
            coverage,        # couverture
            station_depts[station],
            label_ext,
            aggregator
        ))
    return results

# --- Fonctions d'interpolation et de visualisation ---
def create_interpolated_grid(longitudes, latitudes, values, method='linear', resolution=DEFAULT_GRID_RESOLUTION):
    """
    Crée une grille interpolée à partir des points de données.
    """
    valid_data = [(lon, lat, val) for lon, lat, val in zip(longitudes, latitudes, values) if not np.isnan(val)]
    if not valid_data:
        logging.warning("Pas de données valides pour l'interpolation")
        return np.linspace(0, 1, 2), np.linspace(0, 1, 2), np.zeros((2, 2))
    lons, lats, vals = zip(*valid_data)
    if len(lons) == 1:
        lon, lat, val = lons[0], lats[0], vals[0]
        grid_x = np.linspace(lon - 0.3, lon + 0.3, 20)
        grid_y = np.linspace(lat - 0.3, lat + 0.3, 16)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        return grid_x, grid_y, np.full_like(grid_x, val)
    nx, ny = resolution
    grid_x = np.linspace(-5, 10, nx)
    grid_y = np.linspace(41, 52, ny)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata((lons, lats), vals, (grid_x, grid_y), method=method)
    if np.isnan(grid_z).any():
        logging.info(f"Interpolation secondaire (nearest) pour combler {np.isnan(grid_z).sum()} valeurs manquantes")
        fill_z = griddata((lons, lats), vals, (grid_x, grid_y), method='nearest')
        grid_z[np.isnan(grid_z)] = fill_z[np.isnan(grid_z)]
    return grid_x, grid_y, grid_z

def idw_interpolation(x_known, y_known, z_known, x_grid, y_grid, power=2.0, smoothing=1e-5):
    """
    Interpolation par pondération inverse à la distance (IDW).
    """
    x_known = np.array(x_known, dtype=np.float64)
    y_known = np.array(y_known, dtype=np.float64)
    z_known = np.array(z_known, dtype=np.float64)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = np.zeros_like(x_flat)
    batch_size = 5000
    num_batches = (len(x_flat) + batch_size - 1) // batch_size
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(x_flat))
        for i in range(start_idx, end_idx):
            distances = np.sqrt((x_known - x_flat[i])**2 + (y_known - y_flat[i])**2) + smoothing
            weights = 1.0 / (distances**power)
            if (wsum := np.sum(weights)) > 0:
                z_flat[i] = np.sum(weights * z_known) / wsum
    z_grid = z_flat.reshape(x_grid.shape)
    return z_grid

def get_custom_colormap(var_code):
    """
    Retourne une colormap adaptée à la variable.
    """
    var_code = var_code.upper()
    if var_code == 'RR':
        values = [0, 20, 50, 80, 120, 160, 200, 300, 400, 650]
        colors = [
            (1.0, 1.0, 1.0),
            (0.0, 0.4, 1.0),
            (0.0, 0.6, 0.8),
            (0.0, 0.7, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.65, 0.0),
            (1.0, 0.0, 0.0),
            (0.55, 0.27, 0.07),
            (0.5, 0.0, 0.5),
            (1.0, 0.75, 0.8)
        ]
    elif var_code == 'INST':
        values = [0, 20, 50, 100, 150, 200, 300, 400, 500]
        colors = [
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.65, 0.0),
            (1.0, 0.0, 0.0),
            (0.55, 0.27, 0.07),
            (0.5, 0.0, 0.5),
            (1.0, 0.75, 0.8)
        ]
    else:
        return plt.cm.rainbow
    normed_vals = [(v - min(values)) / (max(values) - min(values)) for v in values]
    return mcolors.LinearSegmentedColormap.from_list(
        f'{var_code.lower()}_cmap', list(zip(normed_vals, colors)), N=256
    )

def mask_grid_to_dept(grid_x, grid_y, grid_z, dept_code):
    """
    Masque (met à NaN) les points en dehors du département.
    """
    try:
        from shapely.geometry import Point
        gdf = get_gdf_cached()
        dept_geom = gdf.loc[gdf['INSEE_DEP'] == dept_code.zfill(2), 'geometry']
        if dept_geom.empty:
            logging.warning(f"Impossible de trouver la géométrie du dept {dept_code}")
            return grid_z
        from shapely.ops import unary_union
        dept_polygon = unary_union(dept_geom)
        rows, cols = grid_x.shape
        for i in range(rows):
            for j in range(cols):
                if not dept_polygon.contains(Point(grid_x[i, j], grid_y[i, j])):
                    grid_z[i, j] = np.nan
    except Exception as e:
        logging.error(f"Erreur dans le masquage du dept {dept_code}: {e}")
    return grid_z

def add_base_map_features(ax, dept_main=None):
    """
    Ajoute les éléments de base (départements, pays, océans) sur la carte.
    """
    gdf = get_gdf_cached()
    if dept_main and "FR" not in dept_main:
        dept_main_normalized = [d.zfill(2) for d in dept_main]
        sel = gdf[gdf['INSEE_DEP'].isin(dept_main_normalized)]
        other = gdf[~gdf['INSEE_DEP'].isin(dept_main_normalized)]
        ax.add_geometries(other['geometry'], ccrs.PlateCarree(), facecolor='white', edgecolor='none', zorder=2)
        ax.add_geometries(sel['geometry'], ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=1.2, zorder=3)
        if not sel.empty:
            minx, miny, maxx, maxy = sel.total_bounds
            ax.set_extent([minx-0.5, maxx+0.5, miny-0.5, maxy+0.5], crs=ccrs.PlateCarree())
    else:
        ax.add_geometries(gdf['geometry'], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, zorder=2)
    countries = shapereader.natural_earth('10m', 'cultural', 'admin_0_countries')
    for country in shapereader.Reader(countries).records():
        if country.attributes['NAME'] != 'France':
            ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor='white', edgecolor='gray', linewidth=0.5, zorder=4)
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '10m')
    ax.add_geometries(ocean.geometries(), ccrs.PlateCarree(), facecolor='lightblue', edgecolor='none', zorder=10)

def add_station_annotations(ax, stations_data, var_code, dept_main=None, is_zoomed=False, 
                            min_coverage=DEFAULT_MIN_COVERAGE, allvaleurs=False, allname=False):
    """
    Ajoute les annotations de stations sur la carte.
    """
    var_code = var_code.upper()
    dept_main_normalized = [d.zfill(2) for d in dept_main] if dept_main else []
    if allvaleurs:
        valid_stations = []
        for lon, lat, name, val_main, _, _, coverage, dept, _, _ in stations_data:
            if coverage < min_coverage:
                continue
            if var_code == 'RR' and val_main < 0.2:
                continue
            if dept_main_normalized and "FR" not in dept_main_normalized and dept not in dept_main_normalized:
                continue
            valid_stations.append((lon, lat, name, val_main))
        for lon, lat, name, value in valid_stations:
            ax.plot(lon, lat, marker='o', color='black', markersize=2, transform=ccrs.PlateCarree(), zorder=100)
            ax.text(lon, lat+0.01, f"{value:.1f}", transform=ccrs.PlateCarree(),
                    fontsize=7, color='red', ha='center', va='bottom', zorder=100,
                    bbox=dict(facecolor='white', alpha=0.7, pad=0.1, boxstyle='round'))
            if allname:
                ax.text(lon, lat-0.01, name, transform=ccrs.PlateCarree(),
                        fontsize=5, color='black', ha='center', va='top', zorder=100)
        return

    if var_code == 'RR':
        dept_max = {}
        for lon, lat, _, val_main, _, _, coverage, dept, _, agg in stations_data:
            if coverage < min_coverage:
                continue
            if val_main < 0.2:
                continue
            if agg != "cumul":
                continue
            if dept_main_normalized and "FR" not in dept_main_normalized and dept not in dept_main_normalized:
                continue
            if dept not in dept_max or val_main > dept_max[dept]['val']:
                dept_max[dept] = {'lon': lon, 'lat': lat, 'val': val_main}
        annotations = sorted([(data['lon'], data['lat'], data['val']) for data in dept_max.values()], key=lambda x: x[2], reverse=True)
        threshold = 0.1 if is_zoomed else 0.2
        displayed = []
        min_labels_to_show = 3 if is_zoomed else 1
        guaranteed_labels = annotations[:min_labels_to_show]
        other_labels = annotations[min_labels_to_show:]
        for lon, lat, val in guaranteed_labels:
            displayed.append((lon, lat, val))
            ax.text(lon, lat, f"{val:.1f}", transform=ccrs.PlateCarree(), 
                    fontsize=7 if is_zoomed else 6, fontweight='bold', color='red',
                    ha='center', va='center', zorder=100, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.1'))
        for lon, lat, val in other_labels:
            if any(math.hypot(lon - d_lon, lat - d_lat) < threshold for d_lon, d_lat, _ in displayed):
                continue
            displayed.append((lon, lat, val))
            ax.text(lon, lat, f"{val:.1f}", transform=ccrs.PlateCarree(), 
                    fontsize=7 if is_zoomed else 6, fontweight='bold', color='red',
                    ha='center', va='center', zorder=100, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.1'))
    else:
        filtered_stations = []
        for lon, lat, name, val_main, _, _, coverage, dept, _, _ in stations_data:
            if coverage < min_coverage:
                continue
            if dept_main_normalized and "FR" not in dept_main_normalized and dept not in dept_main_normalized:
                continue
            filtered_stations.append((lon, lat, name, val_main, dept))
        filtered_stations.sort(key=lambda x: x[3], reverse=True)
        threshold = 0.1 if is_zoomed else 0.2
        max_labels = 50 if is_zoomed else 20
        displayed = []
        min_labels_to_show = 5 if is_zoomed else 2
        guaranteed_labels = filtered_stations[:min_labels_to_show]
        other_labels = filtered_stations[min_labels_to_show:]
        for lon, lat, name, val, dept in guaranteed_labels:
            displayed.append((lon, lat))
            ax.text(lon, lat, f"{val:.1f}", transform=ccrs.PlateCarree(), 
                    fontsize=8, color='black', ha='center', va='center', zorder=11)
        for lon, lat, name, val, dept in other_labels:
            if len(displayed) >= max_labels:
                break
            if any(math.hypot(lon - d_lon, lat - d_lat) < threshold for d_lon, d_lat in displayed):
                continue
            displayed.append((lon, lat))
            ax.text(lon, lat, f"{val:.1f}", transform=ccrs.PlateCarree(), 
                    fontsize=8, color='black', ha='center', va='center', zorder=11)

def add_map_attribution(ax):
    """Ajoute une attribution en bas de la carte."""
    ax.text(0.5, 0.01, 
            "Carte générée par https://www.meteociel.fr/ (données Meteo-France)",
            transform=ax.transAxes, ha='center', va='bottom', 
            color='white', fontsize=8,
            bbox=dict(facecolor='black', alpha=0.7), 
            zorder=9999, clip_on=False)

# --- Fonctions pour le reporting détaillé ---
def log_complete_station_stats(stations_data, total_days, var_code, depts_to_analyze):
    logging.info("\n===== DONNÉES COMPLÈTES (SANS FILTRES) =====")
    stations_group = defaultdict(list)
    for key in stations_data:
        stations_group[key].extend(stations_data[key])
    for dept in depts_to_analyze:
        dept_stations = [(sid, name, dept_code) for (sid, name, dept_code) in stations_group if dept_code == dept]
        if not dept_stations:
            logging.info(f"Département {dept}: Aucune station trouvée")
            continue
        logging.info(f"\nDépartement {dept}: {len(dept_stations)} stations")
        for i, (station_id, name, _) in enumerate(sorted(dept_stations)):
            daily_values = stations_group[(station_id, name, dept)]
            coverage = (len(set(date for date, _ in daily_values)) / total_days) * 100
            if var_code == 'RR':
                total_value = sum(val for _, val in daily_values)
                logging.info(f"{i+1}. {station_id} - {name}: {total_value:.1f}mm (cumul), Couverture: {coverage:.1f}%")
            elif var_code == 'TN':
                min_value = min(val for _, val in daily_values)
                mean_value = sum(val for _, val in daily_values) / len(daily_values)
                logging.info(f"{i+1}. {station_id} - {name}: {mean_value:.1f}°C (moy), {min_value:.1f}°C (min), Couverture: {coverage:.1f}%")
            elif var_code == 'TX':
                max_value = max(val for _, val in daily_values)
                mean_value = sum(val for _, val in daily_values) / len(daily_values)
                logging.info(f"{i+1}. {station_id} - {name}: {mean_value:.1f}°C (moy), {max_value:.1f}°C (max), Couverture: {coverage:.1f}%")
            else:
                mean_value = sum(val for _, val in daily_values) / len(daily_values)
                logging.info(f"{i+1}. {station_id} - {name}: {mean_value:.1f} (moy), Couverture: {coverage:.1f}%")

def log_filtered_station_stats(stations_data, total_days, var_code, depts_to_analyze):
    logging.info("\n===== DONNÉES FILTRÉES (COUVERTURE ≥ 95%, RR ≥ 0.2mm) =====")
    stations_group = defaultdict(list)
    for key in stations_data:
        stations_group[key].extend(stations_data[key])
    for dept in depts_to_analyze:
        dept_stations = [(sid, name, dept_code) for (sid, name, dept_code) in stations_group if dept_code == dept]
        if not dept_stations:
            continue
        filtered_stations = []
        for station_id, name, _ in dept_stations:
            daily_values = stations_group[(station_id, name, dept)]
            coverage = (len(set(date for date, _ in daily_values)) / total_days) * 100
            if var_code == 'RR':
                value = sum(val for _, val in daily_values)
                if coverage >= 95.0 and value >= 0.2:
                    filtered_stations.append((station_id, name, value, coverage))
            else:
                if var_code == 'TN':
                    value = min(val for _, val in daily_values)
                elif var_code == 'TX':
                    value = max(val for _, val in daily_values)
                else:
                    value = sum(val for _, val in daily_values) / len(daily_values)
                if coverage >= 95.0:
                    filtered_stations.append((station_id, name, value, coverage))
        filtered_stations.sort(key=lambda x: x[2], reverse=True)
        logging.info(f"\nDépartement {dept}: {len(filtered_stations)} stations après filtrage")
        for i, (station_id, name, value, coverage) in enumerate(filtered_stations):
            if var_code == 'RR':
                logging.info(f"{i+1}. {station_id} - {name}: {value:.1f}mm (cumul), Couverture: {coverage:.1f}%")
            elif var_code == 'TN':
                logging.info(f"{i+1}. {station_id} - {name}: {value:.1f}°C (min), Couverture: {coverage:.1f}%")
            elif var_code == 'TX':
                logging.info(f"{i+1}. {station_id} - {name}: {value:.1f}°C (max), Couverture: {coverage:.1f}%")
            else:
                logging.info(f"{i+1}. {station_id} - {name}: {value:.1f} (moy), Couverture: {coverage:.1f}%")

def analyze_data_details(df, var_code, start_date, end_date, dept_main=None):
    """
    Analyse détaillée des données pour l'option --check-data.
    """
    if df.empty:
        logging.warning("Aucune donnée à analyser")
        return
    start_str, end_str = start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')
    mask = (df['AAAAMMJJ'] >= start_str) & (df['AAAAMMJJ'] <= end_str)
    df_period = df[mask].copy()
    if df_period.empty:
        logging.warning(f"Aucune donnée pour la période du {start_date} au {end_date}")
        return
    if 'DEPT' not in df_period.columns:
        df_period['DEPT'] = df_period['NUM_POSTE'].apply(extract_dept)
    total_days = (end_date - start_date).days + 1
    if dept_main and "FR" not in dept_main:
        depts_to_analyze = [d.zfill(2) for d in dept_main]
    else:
        depts_to_analyze = sorted(df_period['DEPT'].unique())
    total_stations = df_period['NUM_POSTE'].nunique()
    logging.info(f"Analyse pour {var_code} du {start_date} au {end_date}: {total_stations} stations au total")
    station_data, _ = collect_station_data(df_period, var_code, total_days)
    log_complete_station_stats(station_data, total_days, var_code, depts_to_analyze)
    log_filtered_station_stats(station_data, total_days, var_code, depts_to_analyze)

# --- Fonctions pour l'affichage des cartes ---
def _filter_station_data(stations_data, var_code, min_coverage):
    """Filtre les stations selon la couverture minimale et (pour RR) la valeur minimale."""
    filtered = []
    for s in stations_data:
        if s[6] < min_coverage:
            continue
        if var_code == 'RR' and s[3] < 0.2:
            continue
        filtered.append((s[0], s[1], s[3]))
    return filtered

def _create_grid_and_levels(lons, lats, vals, interpolation_method, grid_resolution, var_code):
    """Crée la grille interpolée et calcule les niveaux pour contourf."""
    if interpolation_method.lower() == 'idw':
        grid_x = np.linspace(-5, 10, grid_resolution[0])
        grid_y = np.linspace(41, 52, grid_resolution[1])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        grid_z = idw_interpolation(lons, lats, vals, grid_x, grid_y)
    else:
        grid_x, grid_y, grid_z = create_interpolated_grid(lons, lats, vals,
                                                           method=interpolation_method,
                                                           resolution=grid_resolution)
    valid_vals = [v for v in vals if not np.isnan(v)]
    if not valid_vals:
        levels = np.linspace(0, 1, 50)
    else:
        if var_code == 'RR':
            vmin, vmax = 0, 650
        elif var_code == 'INST':
            vmin, vmax = 0, 500
        else:
            vmin, vmax = min(valid_vals), max(valid_vals)
        vmax = max(vmax, vmin + 1)
        levels = np.linspace(vmin, vmax, 50)
    return grid_x, grid_y, grid_z, levels

def plot_map(ax, stations_data, var_code, start_date=None, end_date=None, 
             dept_main=None, interpolation_method='linear',
             grid_resolution=DEFAULT_GRID_RESOLUTION, min_coverage=DEFAULT_MIN_COVERAGE, 
             allvaleurs=False, allname=False, mask_outside=False):
    """
    Affiche une carte interpolée avec annotations.
    """
    if not stations_data:
        ax.set_title("Aucune donnée")
        return None
    var_code = var_code.upper()
    title = VAR_LABELS_FR.get(var_code, var_code)
    if start_date and end_date:
        date_str = start_date.strftime('%d/%m/%Y') if start_date == end_date else f"Du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}"
        ax.set_title(f"{title}\n{date_str}")
    filtered_data = _filter_station_data(stations_data, var_code, min_coverage)
    if not filtered_data:
        logging.warning(f"Aucune station avec une couverture >= {min_coverage}% trouvée")
        ax.text(0.5, 0.5, "Données insuffisantes\nCouverture < 95%", 
                transform=ax.transAxes, ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
        return None
    lons, lats, vals = zip(*filtered_data)
    grid_x, grid_y, grid_z, levels = _create_grid_and_levels(lons, lats, vals, interpolation_method, grid_resolution, var_code)
    if mask_outside and dept_main and len(dept_main) == 1 and "FR" not in dept_main:
        grid_z = mask_grid_to_dept(grid_x, grid_y, grid_z, dept_main[0])
    cmap = get_custom_colormap(var_code)
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
    is_zoomed = bool(dept_main and "FR" not in dept_main)
    add_base_map_features(ax, dept_main)
    add_station_annotations(ax, stations_data, var_code, dept_main, is_zoomed, min_coverage, allvaleurs, allname)
    return contour

# --- Fonctions de génération des cartes ---
def process_variable_map(var, dept_for_data, start_date, end_date, dept_main,
                         interpolation, grid_resolution, min_coverage, check_data,
                         allvaleurs, allname, mask_outside, ax):
    df = load_csv_for_var(dept_for_data, var, start_date, end_date)
    if df.empty:
        logging.warning(f"Pas de données pour {var}")
        return None
    if check_data:
        analyze_data_details(df, var, start_date, end_date, dept_main)
    data = aggregate_period(df, start_date, end_date, var)
    if not data:
        logging.info(f"Aucune donnée agrégée pour {var} sur {start_date}..{end_date}")
        return None
    cset = plot_map(ax, data, var, start_date, end_date, dept_main,
                    interpolation_method=interpolation,
                    grid_resolution=grid_resolution,
                    min_coverage=min_coverage,
                    allvaleurs=allvaleurs,
                    allname=allname,
                    mask_outside=mask_outside)
    return cset

def determine_output_filename(var_list, start_date, end_date, dept_main, output_dir):
    if "FR" not in dept_main:
        dept_str = "-".join(sorted(dept_main))
        filename = f"{var_list[0]}_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}_dept-{dept_str}.png"
    else:
        filename = f"{var_list[0]}_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.png"
    return os.path.join(output_dir, filename)

def generate_maps(date_range, variables, departements="FR", adj_level=1, dpi=150,
                  output_dir=".", interpolation="linear", min_coverage=DEFAULT_MIN_COVERAGE,
                  grid_resolution=DEFAULT_GRID_RESOLUTION, check_data=False, allvaleurs=False, 
                  allname=False, mask_outside=False):
    os.makedirs(output_dir, exist_ok=True)
    try:
        start_date, end_date = parse_date_range(date_range)
        var_list = [v.strip().upper() for v in variables.split(',')]
        var_list = [v for v in var_list if v in VALID_VARIABLES]
        if not var_list:
            logging.error("Aucune variable valide fournie.")
            return []
        dept_main = [d.strip().zfill(2) for d in departements.split(',')]
        neighbors_dict = build_dept_adjacency()
        if "FR" in dept_main:
            dept_for_data = ["FR"]
        else:
            dept_for_data = sorted(find_neighbors_n_steps(neighbors_dict, dept_main, adj_level))
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = dpi
        fig, axs = plt.subplots(
            len(var_list), 1, 
            figsize=(18, 8 * len(var_list)),
            subplot_kw={'projection': ccrs.AlbersEqualArea(2, 46.5)},
            gridspec_kw={'wspace': 0.05, 'hspace': 0.1}
        )
        axs = np.atleast_1d(axs)
        if len(axs.shape) == 1:
            axs = axs.reshape(1, 1)
        output_files = []
        for i, var in enumerate(var_list):
            logging.info(f"Traitement de la variable {var}")
            ax = axs[i, 0]
            cset = process_variable_map(var, dept_for_data, start_date, end_date, dept_main,
                                        interpolation, grid_resolution, min_coverage,
                                        check_data, allvaleurs, allname, mask_outside, ax)
            if cset:
                ticks = [0, 20, 50, 80, 120, 160, 200, 300, 400, 650] if var == 'RR' else None
                cb = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.03, shrink=0.2, ticks=ticks)
                cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                cb.ax.tick_params(labelsize=8)
            add_map_attribution(ax)
        output_file = determine_output_filename(var_list, start_date, end_date, dept_main, output_dir)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        logging.info(f"Carte exportée : {output_file}")
        plt.close(fig)
        output_files.append(output_file)
        return output_files
    except Exception as e:
        logging.error(f"Erreur lors de la génération des cartes : {e}")
        import traceback
        logging.error(traceback.format_exc())
        return []

def main():
    """Exécute le programme principal à partir des arguments de ligne de commande."""
    start_time_total = datetime.now()
    parser = argparse.ArgumentParser(
        description="Génère des cartes météorologiques interpolées à partir de données Météo France.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('date_range', 
                        help="Période au format YYYYMMDD-YYYYMMDD, YYYYMMDD ou YYYYMM (ex: 20250101-20250131 ou 202501)")
    parser.add_argument('variables', 
                        help="Variables météo à tracer, séparées par des virgules (ex: TN,TX,RR,INST)")
    parser.add_argument('--departements', 
                        help="Codes des départements à mettre en évidence, séparés par des virgules (ex: 41,75 ou FR pour France entière)", 
                        default="FR")
    parser.add_argument('--adj-level', type=int, default=1, help="Niveau BFS pour inclure les départements voisins")
    parser.add_argument('--dpi', type=int, default=150, help="Résolution DPI des cartes générées")
    parser.add_argument('--output-dir', default=".", help="Répertoire de sortie pour les cartes générées")
    parser.add_argument('--interpolation', choices=['linear', 'cubic', 'nearest', 'idw'], 
                        default='linear', help="Méthode d'interpolation spatiale")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help="Niveau de détail des logs")
    parser.add_argument('--min-coverage', type=float, default=DEFAULT_MIN_COVERAGE,
                        help="Couverture minimale en pourcentage pour l'affichage des stations")
    parser.add_argument('--grid-resolution', type=str, default='150,110',
                        help="Résolution de la grille d'interpolation (nx,ny)")
    parser.add_argument('--check-data', action='store_true',
                        help="Analyse et affiche les détails des valeurs par département")
    parser.add_argument('--allvaleurs', action='store_true',
                        help="Affiche toutes les valeurs des stations")
    parser.add_argument('--allname', action='store_true',
                        help="Affiche également le nom des stations (à utiliser avec --allvaleurs)")
    parser.add_argument('--mask-outside', action='store_true',
                        help="Masque la colorisation en dehors du département sélectionné")
    parser.add_argument('--departement', dest='departements',
                        help="Alias pour --departements")
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    if not check_critical_paths():
        sys.exit(1)
    try:
        nx, ny = map(int, args.grid_resolution.split(','))
        grid_resolution = (nx, ny)
    except Exception:
        logging.warning(f"Format de résolution invalide ({args.grid_resolution}), utilisation des valeurs par défaut")
        grid_resolution = DEFAULT_GRID_RESOLUTION
    output_files = generate_maps(
        args.date_range,
        args.variables,
        args.departements,
        args.adj_level,
        args.dpi,
        args.output_dir,
        args.interpolation,
        min_coverage=args.min_coverage,
        grid_resolution=grid_resolution,
        check_data=args.check_data,
        allvaleurs=args.allvaleurs,
        allname=args.allname,
        mask_outside=args.mask_outside
    )
    if output_files:
        time_elapsed = (datetime.now() - start_time_total).total_seconds()
        logging.info(f"{len(output_files)} carte(s) générée(s) en {time_elapsed:.2f} secondes.")
        for f in output_files:
            print(f"Carte générée: {f}")
    else:
        logging.error("Aucune carte n'a pu être générée.")
        sys.exit(1)

if __name__ == "__main__":
    main()
