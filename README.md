# pyCSL2HMapGen
A python-based height map interpolation script for Cities: Skylines 2 (CSL2) Map Editor.


See the Example section of pyCSL2HMapGen.ipynb for a quick guide!


## Purpose
Generating world & playable area's height maps for CSL2.


## Motivation
To get the terrain map without using mapbox or supplying anyone information about my credit card number...


## How to: import real-world height map for Cities: Skylines 2 Map Editor (beta)
This script uses python, so you will need a environment that can run python scripts.

Use pip or Anaconda (or whatever) to install the required dependencies:
    python (>=3.10), numpy, scipy, gdal, pyppng
before you continue.

Step 1: Find a real world location you want to make your CSL2 map out of.
    You can use many existing online tools, such as https://heightmap.skydark.pl/beta
    Make sure you grab 3 pieces of info:
        longtitude (of the center of the map),
        latitude (same),
        and the scale (1.0 means to scale 57kmx57km, while 2.0 means 1:2, i.e. mapping 115kmx115km real-world to 57kmx57km in-game.)

Step 2: Download the height map data.
    I personally download data from JAXA's ALOS Global Digital Surface Model ( https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm )
    Data from other sources should work too, but no guarantees.
    You only need to grab the data that covers the area you are interested in.
    If you download data from JAXA, you will need to register for their website,
    **and supply them your name, company, email address** etc. personal information.
    But hey, at least they don't ask for your credit card number!
    Make sure to read their *terms of service* too.
    Their data are free-to-use but **you need to acknowledge your use of their data** in the final product by adding a line states that:
    "The original data used for this product have been supplied by JAXA's ALOS Global Digital Surface Model "ALOS World 3D - 30m" (AW3D30)"

Step 3: Download & set the parameters of the script here.
    Use git clone or whatever method to download this (there should also be a download button in the up-right corner.)
    Make sure python & the dependencies are installed.
    Put the map data into the same folder as the script.
    Now open the jupyter notebook file (pyCSL2HMapGen.ipynb) if you have jupyter notebook,
    or the python script (pyCSL2HMapGen.py) in a text editor if you have not.
    Edit the last few lines under the section "# Example":
    Change the list `tiffilenames` to be a list of the filenames of *your* datafiles. The script will only use the data listed there.
    Change the parameters (from Step 1) for the function call get_CSL_height_maps():
        `long` for center longtitude,
        `lati` for center latitude,
        `scale` for the scale.
    Feel free to explore other parameters of the function.

Step 4: Run the script
    This should produce 2 files: worldmap_{cityname}.png and playable_{cityname}.png
    (cityname by default is the longtitude and the latitude.)
    These are the heightmaps.


Step 5: Import them in-game
    Put those 2 files in your CSL2 saves folder `...\Cities Skylines II\Heightmaps\`
    Boot up the game, enter the editor (beta), in the terrain section of the editor,
    select `import height map` and import the `playable` png file;
    select `import world map`  and import the `worldmap` png file.
    Now you are good to go!


Step 6: Enjoy map-building and celebrate!

