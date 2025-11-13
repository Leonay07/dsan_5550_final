# Urban Heat Island Detection 

This project demonstrates an end-to-end workflow for detecting Urban Heat Islands (UHIs) using MODIS land surface temperature (LST) data. 

## Pipeline Summary
1. Load and crop LST `.tif` files for the Washington DC/DMV region.
2. Clean invalid values and split into 16×16 patches.
3. Label patches as UHI (1) or Non-UHI (0) based on a 303 K threshold.
4. Batch process multiple days and save results as `.npy` files.
5. (To be done)Train a CNN model for classification.

## Files
- `notebooks/batch_preprocess.ipynb` – full preprocessing pipeline
- `data/raw/` – original LST `.tif` files (too large --- gitignore)
- `data/processed/` – generated patch and label arrays

## Dataset Description
The dataset includes six processed days (Day 196–215) for the Washington DC region.

| Day | Total | UHI (1) | Non-UHI (0) |
|-----|--------|---------|-------------|
| 196 | 767 | 275 | 492 |
| 197 | 767 | 715 | 52 |
| 198 | 767 | 17  | 750 |
| 200 | 767 | 710 | 57 |
| 205 | 767 | 404 | 363 |
| 215 | 767 | 343 | 424 |

Day 196： Jul 15, 2020

Day 198 shows minimal UHI may due to a cooler or cloudy day, which reflects realistic weather variability.

Overall, the combined dataset currently includes six days (196–215) covering various temperature conditions.  
This dataset captures both extreme and balanced UHI distributions.  
We may refine or remove certain days in future iterations based on model training results.

## Training & Validation
- Training days: 196, 197, 198, 200, 205  
- Validation day: 215  


