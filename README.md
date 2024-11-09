# pyCSL2HMapGen

Height map interpolation and erosion (Work In Progress).
For Cities: Skylines 2 (CSL2) Map Editor or other generic gameplay scenarios.

Python-based.
Designed for non-serious situations like games.
**No guarantees whatsoever** on the physical accuracy of the erosion process.

See the `Example*.ipynb` jupyter notebooks for a quick guide.


## Functionalities

- Import `.tiff` data (require `gdal`)    (See `Example1_*.ipynb`)
- Load / save height maps from `.png` files    (See `Example2_*.ipynb`)
- Generate and handle world & playable area's height maps for CSL2    (See `Example2_*.ipynb`)
- Resample & rescale height maps
- Extract & Re-insert playable area from world map for CSL2 height maps    (See `Example2_*.ipynb`)
- ~~(Experimental / Work In Progress) Erosion (require CUDA, i.e., require `cuda-nvcc cuda-nvrtc` and cuda-compatible GPU (most modern NVIDIA GPUs are))~~
  Warning: `Erosion` module has been postponed indefinitely. *Do not use*.


## Dependencies

- `python` (>=3.10)
- Packages: `numba numpy scipy matplotlib pypng`
- Optional: `gdal cuda-nvcc cuda-nvrtc jupyter jupyterlab`
- You may also need to update your Nvidia GPU driver if using the CUDA GPU-acceleration feature.
