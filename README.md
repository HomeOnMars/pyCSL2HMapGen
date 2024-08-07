# pyCSL2HMapGen
A python-based height map interpolation script for Cities: Skylines 2 (CSL2) Map Editor.

See the `Example*.ipynb` jupyter notebooks for a quick guide!


## What does it do
Generating and handling world & playable area's height maps for CSL2,
written by and for the python-knowing folks.

#### Functionalities
- Import `.tiff` data (require `gdal`) (See `Example1_*.ipynb`)
- Load / save height maps from `.png` files
- Resample & rescale height maps
- Extract playable area from world map for CSL2 height maps
- Re-insert playable area into world map for CSL2 height maps
- \[Not yet completed\] Erosion ~~(optionally require `cudatoolkit` and cuda-compatible GPU for acceleration)~~


## Dependencies
- `python` (>=3.10)
- Packages: `numba, numpy, scipy, matplotlib, pypng`
- Optional: `gdal, cudatoolkit`
