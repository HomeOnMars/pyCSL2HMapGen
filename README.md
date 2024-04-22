# pyCSL2HMapGen
A python-based height map interpolation script for Cities: Skylines 2 (CSL2) Map Editor.


See the Example section of pyCSL2HMapGen.ipynb for a quick guide!


## What does it do
Generating world & playable area's height maps for CSL2, for the tech-savvy folks who know how to run python scripts.

Codes in the script are written from scratch.


## Motivation
An alternative solution to get the terrain map without having to signing up for mapbox and giving them my credit card number...


## How to: import real-world height map for Cities: Skylines 2 Map Editor (beta)
You can just visit https://terraining.ateliernonta.com/ and use their website's tool to download the height map.

However, if you 1) prefer not to signing up for mapbox (required for the above website's height-map download function), and 2) know how to run pythong script,
then you can use this script to do more-or-less the same by following the following steps:

#### Step 0: Install python
Use pip or Anaconda (or whatever) to install the required dependencies:
    `python (>=3.10), numpy, scipy, gdal, pypng`
before you continue.

#### Step 1: Find a real world location you want to make your CSL2 map out of.
You can use many existing online tools, such as https://heightmap.skydark.pl/beta or https://terraining.ateliernonta.com/
Make sure you grab 3 pieces of info:
    longtitude (of the center of the map),
    latitude (same),
    and the scale (1.0 means to scale 57kmx57km, while 2.0 means 1:2, i.e. mapping 115kmx115km real-world to 57kmx57km in-game.)

#### Step 2: Download the height map data.
I personally download data from JAXA's ALOS Global Digital Surface Model ( https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm )
Data from other sources should work too, but no guarantees.
You only need to grab the data that covers the area you are interested in.
If you download data from JAXA, you will need to register for their website,
**and supply them your name, company, email address** etc. personal information.
But hey, at least they don't ask for your credit card number!
Make sure to read their *terms of service* too.
Their data are free-to-use but **you need to acknowledge your use of their data** in the final product by adding a line states that:
"The original data used for this product have been supplied by JAXA's ALOS Global Digital Surface Model "ALOS World 3D - 30m" (AW3D30)"

#### Step 3: Download & set the parameters of the script here.
Use `git clone` or whatever method to download this
(there should also be a green `code` button in the up-right corner- click that and you will see the download options.)
Make sure python & the dependencies are installed.
Put the map data into the raw folder inside the script folder.
Now open the jupyter notebook file (`pyCSL2HMapGen.ipynb`) if you have jupyter notebook,
or the python script (`pyCSL2HMapGen.py`) in a text editor if you have not.
Edit the last few lines under the section "# Example":
Change the list `tiffilenames` to be a list of the filenames of *your* datafiles. The script will only use the data listed there.
Change the parameters (from Step 1) for the function call get_CSL_height_maps():
    `long` for center longtitude,
    `lati` for center latitude,
    `scale` for the scale.
Feel free to explore other parameters of the function.

#### Step 4: Run the script
This should produce 2 files: `worldmap_{cityname}.png` and `playable_{cityname}.png` in the script folder.
(cityname by default is the longtitude, latitude, rotational angle, and scale.)
These are the heightmaps.

#### Step 5: Import them in-game
Put those 2 files in your CSL2 saves folder `...\Cities Skylines II\Heightmaps\`
Boot up the game, enter the editor (beta), in the terrain section of the editor,
select `import height map` and import the `playable` png file;
select `import world map`  and import the `worldmap` png file.
Now you are good to go!

#### Step 6: Enjoy map-building and celebrate!
(don't forget to add the acknowledgement for JAXA if you do publish anything!
Would be kind to acknowledge me too- but please don't feel pressured.)
