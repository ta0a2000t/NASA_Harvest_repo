import pandas as pd
import numpy as np
import math



def add_veg_indices(df:pd.DataFrame) -> list([str]):
    addedColNames = []

    NIR = df['B8']
    Red = df['B4']
    Blue = df['B2']
    Green = df['B3']
    RB = Blue - Red

    Red_Edge_1 = df['B5']
    Red_Edge_2 = df['B6']
    Red_Edge_3 = df['B7']

    B2 =  df['B2']
    B3 =  df['B3']
    B4 =  df['B4']
    B5 =  df['B5']
    B6 =  df['B6']
    B7 =  df['B7']
    B8 =  df['B8']

    B11 =  df['B11']
    B12 = df['B12']



    # https://www.hindawi.com/journals/js/2017/1353691/
    name = 'DVI'
    df.loc[:, name] = NIR - Red
    addedColNames.append(name)

    name = 'RVI'
    df.loc[:, name] = Red / NIR
    addedColNames.append(name)

    name = 'ARVI'
    df.loc[:, name] = (NIR - RB)/(NIR + RB)
    addedColNames.append(name)


    #####
    #http://www.eo4geo.eu/training/sentinel-2-data-and-vegetation-indices/
    # where: B7 = 783 nm, B6 = 740 nm, B5 = 705 nm, B4 = 665 nm

    """
    name = 'PSSRa'
    df.loc[:, name] = (Red_Edge_3) / (Red)
    addedColNames.append(name)


    """


    #df['TNDVI'] = math.sqrt(df['NDVI'] + 0.5)

    name = 'NDI45'
    df.loc[:, name] = (B5 - B4) / (B5 + B4)
    addedColNames.append(name)

    # same source^. indices with green wavelength
    name = 'GNDVI'
    df.loc[:, name] = (NIR - Green) / (NIR + Green)
    addedColNames.append(name)

    name = 'MCARI'
    df.loc[:, name] = ((B5 - B4) - 0.2 * (B5 - B3)) * (B5 - B4)
    addedColNames.append(name)


    # same source^. indices with red-edge wavelengths
    #name = 'S2REP'
    #df.loc[:, name] = 705 + 35 * ((B4 + B7)/2) - B5 / (B6 - B5)
    #addedColNames.append(name)
    """
    name = 'IRECI'
    df.loc[:, name] = (B7 - B4) / (B5 / B6)
    addedColNames.append(name)

    #####
    # https://www.researchgate.net/figure/Vegetation-indices-used-in-the-study-B2-B4-B5-B6-B7-B8-are-the-Sentinel-2-TOA_tbl1_362826335
    
    name = 'CIr'
    df.loc[:, name] = (B7/B5) - 1
    addedColNames.append(name)

    """
    name = 'MTCI'
    df.loc[:, name] = (B6 - B5) / (B5 - B4)
    addedColNames.append(name)
    """
    
    name = 'NDVIre'
    df.loc[:, name] = (B8 - B5) / (B8 + B5)
    addedColNames.append(name)

    name = 'NIRv'
    df.loc[:, name] = B4 * (B8 - B4) / (B8 + B4)
    addedColNames.append(name)

    name = 'EVI'
    df.loc[:, name] = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))
    addedColNames.append(name)
    """

    #####
    # unused source: https://www.indexdatabase.de/db/is.php?sensor_id=96



    # NDTI=(R1610−R2200)/(R1610+R2200)
    name = 'NDTI'
    df.loc[:, name] = (B11 - B12)/(B11 + B12)
    addedColNames.append(name)

    #https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
    # NDMI (Sentinel 2) = (B8 – B11) / (B8 + B11)
    name = 'NDMI'
    df.loc[:, name] =(B8 - B11) / (B8 + B11)
    addedColNames.append(name)

    #MSI (Sentinel 2) = B11 / B08
    name = 'MSI'
    df.loc[:, name] = B11 / B8
    addedColNames.append(name)

    # GCI = (NIR) / (Green) – 1
    name = 'GCI'
    df.loc[:, name] = (B8 / B3) -1
    addedColNames.append(name)

    #NBRI (Sentinel 2) = (B8 – B12) / (B8 + B12)
    name = 'NBRI'
    df.loc[:, name] = (B8 - B12) / (B8 + B12)
    addedColNames.append(name)

    #BSI = ((Red+SWIR) – (NIR+Blue)) / ((Red+SWIR) + (NIR+Blue))
    # BSI (Sentinel 2) = ((B11 + B4) – (B8 + B2)) / ((B11 + B4) + (B8 + B2))
    name = 'BSI'
    df.loc[:, name] =  ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))
    addedColNames.append(name)


    #NDWI (Sentinel 2) = (B3 – B8) / (B3 + B8)
    name = 'NDWI'
    df.loc[:, name] =  (B3 - B8) / (B3 + B8)
    addedColNames.append(name)

    # NDSI (Sentinel 2) = (B3 – B11) / (B3 + B11)
    name = 'NDSI'
    df.loc[:, name] = (B3 - B11) / (B3 + B11)
    addedColNames.append(name)



    

    print('Added: ', addedColNames)

    # TODO add indices with B11 & B12
    # TODO read about indices
    # TODO google "metric for seperation", then label each index with a metric for separability between harv_evnt
    # vegetation 
    return addedColNames

    





