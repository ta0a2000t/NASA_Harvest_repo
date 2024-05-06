# NASA_Harvest_repo
## Machine learning for pre and post-harvest crop stages
Developing a machine learning method to discriminate between pre-harvesting and post-harvesting crop stages in cases where a high spectral similarity is present between them

In 2022, a country-level harvest model was created for Ukraine using unsupervised machine-learning techniques trained on spatiotemporal sampled data. This analysis, conducted independently of threshold assumptions and crop-specific characteristics, marked the first in-season harvest detection analysis performed at the country level.

The initial model used four-band planet data and demonstrated satisfactory performance for the majority of harvest events. However, challenges arose in distinguishing between the pre-harvest and post-harvest states of some areas with similar Normalized Difference Vegetation Index (NDVI) values. Although the Sentinel-2 satellite, equipped with an additional Short-Wave Infrared (SWIR) band, performed better in such scenarios, it was not employed due to coverage limitations and lower spatial resolution.

This project aims to leverage Sentinel-2 data to enhance the discrimination between pre-harvesting and post-harvesting crop stages where the original model struggled, ultimately improving the overall performance of the harvest detection model. 

# Satellite Image Processing for Agriculture with Earth Engine

This repository contains the Jupyter notebook `get_images_supSpec.ipynb` which is structured to facilitate the processing and analysis of satellite imagery for agricultural purposes using Google Earth Engine. Below is an outline of the notebook's content and functionalities.

## Overview

The notebook is designed to manipulate and analyze satellite data to monitor agricultural fields over time. It includes importing libraries, defining classes and constants, overlaying geographic points, and exporting data for further analysis.

## Sections

### Section 1: Importing Libraries and Initializing Earth Engine

- Import `ee` and other necessary libraries.
- Authenticate and initialize the Google Earth Engine.

### Section 2: Define Constants and Classes

- Define constants such as bands used for prediction, folder names, and file names.
- Define the `Dataset` class to manage different datasets.
- Set the geographical boundaries and center coordinates for Ukraine.

### Section 3: Reading Labeled File

- Load the labeled data from any of the three datasets into a pandas DataFrame.

### Section 4: Overlaying Points on an Image

- Define `overlay_points` function to overlay points on an image and create a FeatureCollection.
- Define `export_to_drive` function to export the FeatureCollection as a GeoJSON file to Google Drive.

### Section 5: Reading GeoJSON File

- Define `read_geojson` function to read a GeoJSON file into a pandas DataFrame and convert the date column to the appropriate format.

### Section 7: Create 3-week Image Collection from Sentinel-2

- Define `get_image` to retrieve a cloud-masked Sentinel-2 image.
- Define `extract_subset` to extract a subset of the image collection based on the start week.
- Create a new image collection by mapping the `extract_subset` function over a list of start weeks.

### Visualization and Mapping

- Visualize and map the image using folium and Earth Engine.
- Define `pinned_points` function to add pinned points on the map based on a DataFrame with coordinates.

## Example Scenario

Assuming there are 500 fields on the map of Ukraine, overlay these locations and then divide the map into 3-week images. Over one year, there are approximately 16 three-week periods, resulting in a data frame with 8,000 rows (500 fields × 16 periods). If 250 out of 500 fields are harvested at some point in the year, after dividing the year into 16 three-week periods, the sum of "harvested" labels remains 250. The remaining labels show how unbalanced the data can be, which is further processed in `nn_model method1.ipynb` & `nn_model method2.ipynb`.


* Presentation at the Global Sustainability Congress 2023:
https://www.timeshighered-events.com/gsd-congress-2023/agenda/speakers/3050569

* Project Slides: https://drive.google.com/file/d/1_UpxdQwB-oYagwFteEx1uzvnmmvRXtT7/view
