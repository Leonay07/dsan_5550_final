# This script loads multiple .tif files (LST data), clips the DC region,
# splits the region into patches, generates binary labels (heat island or not),
# and saves the patches and labels for each day.

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from tqdm import tqdm

# Config
DATA_DIR = "../data/raw"
OUTPUT_DIR = "../data/processed"
DC_BOUNDS = {  # lon/lat bounds
    'west': -77.12,
    'east': -76.90,
    'south': 38.80,
    'north': 39.00
}
PATCH_SIZE = 128
TEMP_THRESHOLD = 305  # Kelvin
DAYS = [196, 197, 198, 200, 205]  # You can add more days here

os.makedirs(OUTPUT_DIR, exist_ok=True)

def latlon_to_pixel(src, lon, lat):
    row, col = rowcol(src.transform, lon, lat)
    return col, row

def extract_patch_data(tif_path, day):
    with rasterio.open(tif_path) as src:
        # Convert DC lat/lon to pixel coordinates
        x_min, y_max = latlon_to_pixel(src, DC_BOUNDS['west'], DC_BOUNDS['north'])
        x_max, y_min = latlon_to_pixel(src, DC_BOUNDS['east'], DC_BOUNDS['south'])

        width, height = x_max - x_min, y_max - y_min
        data = src.read(1, window=Window(x_min, y_min, width, height)).astype(np.float32)

        # Replace invalid values (0 or negative) with NaN
        data[data <= 0] = np.nan

        # Normalize data
        data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

        # Split into patches
        patches = []
        labels = []
        for i in range(0, data.shape[0] - PATCH_SIZE, PATCH_SIZE):
            for j in range(0, data.shape[1] - PATCH_SIZE, PATCH_SIZE):
                patch = data[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                if np.isnan(patch).mean() > 0.3:
                    continue  # Skip mostly invalid patches

                label = int(np.nanmean(patch) * (np.nanmax(data) - np.nanmin(data)) + np.nanmin(data) > TEMP_THRESHOLD)
                patches.append(patch)
                labels.append(label)

        # Save
        patches = np.array(patches)
        labels = np.array(labels)
        np.save(os.path.join(OUTPUT_DIR, f"patches_day{day}.npy"), patches)
        np.save(os.path.join(OUTPUT_DIR, f"labels_day{day}.npy"), labels)
        print(f"Saved {len(patches)} patches for day {day}")

if __name__ == '__main__':
    for day in DAYS:
        tif_file = os.path.join(DATA_DIR, f"gf_Day2020_{day}.tif")
        if not os.path.exists(tif_file):
            print(f"File not found: {tif_file}")
            continue
        extract_patch_data(tif_file, day)
