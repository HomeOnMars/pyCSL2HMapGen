# pyCSL2HMapGen

A python-based height map interpolation script for Cities: Skylines 2 (CSL2) Map Editor.

See the `Example*.ipynb` jupyter notebooks for a quick guide!


## What does it do
Generating and handling world & playable area's height maps for CSL2,
written by and for the python-knowing folks.

#### Functionalities

- Import `.tiff` data (require `gdal`)    (See `Example1_*.ipynb`)
- Load / save height maps from `.png` files    (See `Example2_*.ipynb`)
- Resample & rescale height maps
- Extract playable area from world map for CSL2 height maps    (See `Example2_*.ipynb`)
- Re-insert playable area into world map for CSL2 height maps    (See `Example2_*.ipynb`)
- (Experimental / Work In Progress) Erosion (require CUDA, i.e., require `cuda-nvcc cuda-nvrtc` and cuda-compatible GPU (most modern NVIDIA GPUs are))


## Dependencies

- `python` (>=3.10)
- Packages: `numba numpy scipy matplotlib pypng`
- Optional: `gdal cuda-nvcc cuda-nvrtc jupyter jupyterlab`
- You may also need to update your Nvidia GPU driver if using the CUDA GPU-acceleration feature.
