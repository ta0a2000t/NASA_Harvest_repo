import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from distinctipy import distinctipy # get n distinct colors

import numpy as np
import pandas as pd
import math
import scipy
from scipy.stats import norm
import geopandas

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

import datetime

def rmved_outliers_iqr(arrayIn: np.array, retBools:bool=False):
    """returns a copy of arrayIn where outliers(shown in box plots) are set to nan

    Args:
        arrayIn (np.array): its shape is nx1

    Returns:
        array & bools where bools[i] is True and array[i] is nan
    """
    if(len(arrayIn.shape) != 1): # one dimension
        Exception("Wrong Input^^")
    array = arrayIn.copy()
    array = array.astype(float)

    Q1 = np.quantile(array, 0.25) # Q1
    Q3 = np.quantile(array, 0.75) # Q2
    
    IQR = Q3 - Q1

    bools = array < Q1 - 1.5*IQR
    bools = bools | (array > Q3 + 1.5*IQR)

    array[bools] = np.nan

    if(retBools):
        return array, bools
    return array



def saveFigsAsPDF(figs:list([matplotlib.figure.Figure]), filename:str):
    pp = PdfPages(filename)
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    print(filename,"  saved!")

def sort_by_points(df: pd.DataFrame) -> None:
    df['temp'] = df.point_idx.apply(lambda x: int(x[1:]))
    df.sort_values(by=['temp'], ascending=True, inplace=True)
    df.drop(["temp"], axis=1, inplace=True)

    
def sort_by_images(df: pd.DataFrame) -> None:
    df['temp'] = df.point_idx.apply(lambda x: int(x[1:]))
    df.sort_values(by=['temp'],ascending=True, inplace=True)
    df.drop(["temp"], axis=1, inplace=True)

def sort_by_points_images(df:pd.DataFrame) -> None:
    sort_by_points(df)
    sort_by_images(df)

def get_unique_sorted_image_idx(df: pd.DataFrame) -> list([str]):
    return sorted(np.unique(df.image_idx), key=lambda x: int(x[1:]))

def get_unique_sorted_point_idx(df: pd.DataFrame) -> list([str]):
    return sorted(np.unique(df.point_idx), key=lambda x: int(x[1:]))

# TODO add differences between other metrics too.
# TODO normalize
# TODO add back non-harvested samples
# TODO quantify saparation for each index/and diff
# TODO look up class balancing techniques(bc we would have a lot of non-harvested and only ~366 harvested labels)


# TODO possibly do CNN
# TODO ANN network
# TODO one-d CNN

def nan_arr(length:int) -> np.array:
    arr = np.zeros(length)
    arr[:] = np.nan
    return arr

def get_standarized_arr(arr: np.array) -> np.array:
    mean = arr.mean()
    sd = arr.std()
    return (arr - mean)/sd

def stretch_cols(df:pd.DataFrame, NUMERIC_COLS: list[str])->pd.DataFrame:
    stretch_numeric_df = pd.DataFrame({'value':np.array([]), 'class':np.array([]), 'har_evnt': np.array([]), 'image_idx':np.array([])})
    for col_name in NUMERIC_COLS:
        curr_df = pd.DataFrame({'value': df[col_name], 'class':np.tile(np.array([col_name]), df.shape[0]), 'har_evnt':df['har_evnt'], 'image_idx':df['image_idx'], 'start_date':df['start_date']})
        stretch_numeric_df = pd.concat([stretch_numeric_df, curr_df])
    return stretch_numeric_df

def get_rm_outlier_standarize(df:pd.DataFrame, NUMERIC_COLS: list[str], standarize:bool=True, rm_outliers:bool=True)->pd.DataFrame:
    """
        For each 3-week image, standarize each column, after removing outliers.
    """
    res_df = None
    for image_idx in df.image_idx.unique():
        curr_df = df[df.image_idx == image_idx]
        for numeric_col_name in NUMERIC_COLS:
            if(rm_outliers):
                # setting outliers to NaN
                curr_df[numeric_col_name] = rmved_outliers_iqr(curr_df[numeric_col_name])
            if(standarize):
                curr_df[numeric_col_name] = get_standarized_arr(curr_df[numeric_col_name])
            
        if(type(res_df) == type(None)):
            res_df = curr_df
        else:
            res_df = pd.concat([res_df, curr_df])
    return res_df

def _get_hists(arr1:np.array, arr2:np.array, num_buckets=50):
    minn = min(np.min(arr1), np.min(arr2))
    maxx = max(np.max(arr1), np.max(arr2))
    if(minn == np.nan):
        print(arr1, arr2, "______________________")
    hist1, _ = np.histogram(arr1, bins = num_buckets, range=(minn, maxx))
    hist2, _ = np.histogram(arr2, bins = num_buckets, range=(minn, maxx))

    return norm.pdf(hist1), norm.pdf(hist2)

def downsample(arr, length) -> np.array:
    indices = np.random.choice(len(arr), length, replace=False)
    return np.take(arr, indices)

def get_Bhattacharyya_coef(arr1:np.array, arr2:np.array) -> float:
    def bhattacharyya(h1, h2):
        '''Calculates the Byattacharyya distance of two histograms.'''

        def normalize(h):
            if(np.sum(h) == 0):
                return h
            return h / np.sum(h)

        return -np.log(np.sum(np.sqrt(np.multiply(normalize(h1), normalize(h2)))))

    
    hist1, hist2 = _get_hists(arr1, arr2)

    return bhattacharyya(hist1, hist2)
    

def get_MAE_coef(arr1:np.array, arr2:np.array)-> float:
    length = min(len(arr1), len(arr2))
    if(length == 0):
        return None
    return mean_absolute_error(downsample(arr1, length), downsample(arr2, length))

def get_RMSE_coef(arr1:np.array, arr2:np.array) -> float:
    length = min(len(arr1), len(arr2))
    if(length == 0):
        return None
    return np.sqrt(mean_squared_error(downsample(arr1, length), downsample(arr2, length)))
def get_cosine_similarity_coef(arr1:np.array, arr2:np.array) -> float:
    length = min(len(arr1), len(arr2))
    if(length == 0):
        return None
    arr1, arr2, = downsample(arr1, length), downsample(arr2, length)
    
    return cosine_similarity(np.reshape(arr1, (1, -1)), np.reshape(arr2, (1, -1)))[0][0]

def get_euclidean_dist(arr1:np.array, arr2:np.array) -> float:
    length = min(len(arr1), len(arr2))
    if(length == 0):
        return None
    return euclidean(downsample(arr1, length), downsample(arr2, length))

    
def get_Hellinger_distance(arr1:np.array, arr2:np.array, num_buckets=50):
    def Hellinger_distance(h1, h2):
        '''Calculates the Hellinger distance of two histograms.'''

        def normalize(h):
            return h / np.sum(h)

        return 1 - np.sum(np.sqrt(np.multiply(normalize(h1), normalize(h2))))
    
    hist1, hist2 = _get_hists(arr1, arr2)
    
    return Hellinger_distance(hist1, hist2)

def get_JSD_distance(arr1:np.array, arr2:np.array) -> float:
    """
        Jensen-Shannon Divergence (JSD): 
        This is a distance metric that measures 
        the similarity between two probability distributions. 
        The JSD is symmetric and bounded between 0 and 1, where 0 indicates 
        that the two distributions are identical, and 1 indicates that 
        they have no overlap. 
        
    """
    def KL(A: np.array, M: np.array):
        """         
        Kullback-Leibler divergence 
        A represents the data, the observations, or a measured probability distribution. 

        Distribution M represents instead a theory, a model.
        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        """

        return np.sum(np.where(A != 0, np.multiply(A, np.log(np.divide(A, M))), 0))

    def JSD_distance(hist_P: np.array, hist_Q: np.array)-> float:
        '''Calculates the JSD distance of two distributions.

            https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        '''
        
        def normalize(h):
            return h / np.sum(h)
        hist_P, hist_Q = normalize(hist_P), normalize(hist_Q)

        #M is the average distribution of P and Q.
        M = 0.5 * (hist_P + hist_Q)
        #print(M, "<----",  np.divide(hist_P, M))

        return 0.5 * KL(hist_P, M) + 0.5 * KL(hist_Q, M)


    
    hist1, hist2 = _get_hists(arr1, arr2)


    return JSD_distance(hist1, hist2)

def get_K_S_Test(arr1:np.array, arr2:np.array) -> float:
    """
        Two-sample Kolmogorov–Smirnov test

        https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    """
    hist1, hist2 = _get_hists(arr1, arr2)
    return scipy.stats.kstest(hist1, hist2).pvalue



    
def get_classes_colors(NUMERIC_COLS:list([str]))-> dict:
    names = sorted(NUMERIC_COLS)
    colors = distinctipy.get_colors(len(names))

    dic = {}
    for i in range(len(names)):
        name = names[i]
        dic[name] = colors[i]
    return dic


def get_drop_after_harvest(df: pd.DataFrame) -> pd.DataFrame:
    
    df_list = []
    for point_idx in get_unique_sorted_point_idx(df):
        curr_point_df = df[df["point_idx"] == point_idx]
        row_df_list = []
        for image_idx in get_unique_sorted_image_idx(curr_point_df):
            row_df = curr_point_df[curr_point_df.image_idx == image_idx]
            row_df_list.append(row_df)
            if(row_df.har_evnt.to_numpy()[0]):
                break
        df_list.append(pd.concat(row_df_list))
    return pd.concat(df_list)
            
def get_drop_after_finHarDat(df:pd.DataFrame) -> pd.DataFrame:
    df_list = []
    for point_idx in get_unique_sorted_point_idx(df):
        curr_point_df = df[df["point_idx"] == point_idx]
        row_df_list = []
        for image_idx in get_unique_sorted_image_idx(curr_point_df):
            row_df = curr_point_df[curr_point_df.image_idx == image_idx]
            row_df_list.append(row_df)
            if(row_df.finHarvDat.to_numpy()[0] < row_df.end_date.to_numpy()[0]):
                break
        df_list.append(pd.concat(row_df_list))
    return pd.concat(df_list)

def get_within_dates(df:pd.DataFrame):
    df = df[(df["start_date"] > datetime.datetime(year=2022, month=5, day=1))]
    df = df[(df["end_date"] < datetime.datetime(year=2022, month=10, day=1))]
    return df.reset_index()
def get_df(DF:pd.DateOffset, groundTruth:bool, veg_indices, BANDS):
    
    # cpied from learning_about-data.ipynb
    #DF = geopandas.read_file(f'../data/{file_name}.geojson')
    DF.rename(columns = {'is_within_period':'har_evnt'}, inplace = True)
    NUM_SAMPLES = len(np.unique(DF.image_idx)) - 1
    print(DF.shape)
    DF.drop_duplicates(subset=BANDS, keep="last", inplace=True)
    print(DF.shape)
    print(99999999)

    # cpied from learning_about-data.ipynb
    df = get_within_dates(DF.copy())
    df = df[(df.NDVI) != 0] # drop invalid points
    VEG_INDICES_NAMES = veg_indices.add_veg_indices(df) + ['NDVI']
    df, VEG_DIFF_NAMES, PREV_VEG_NAMES = veg_indices.get_added_veg_prev_and_diff(df, VEG_INDICES_NAMES)
    NUMERIC_COLS = BANDS + VEG_INDICES_NAMES + VEG_DIFF_NAMES + PREV_VEG_NAMES
    if not groundTruth:
        df = get_drop_after_harvest(df)# drop rows of non-harvest, after a harvest event in a farm ( a point )
        df = get_drop_after_finHarDat(df)

    # for some reason, around 26 rows have same values for B4 & B5, making MTCI give infinite values
    df = df.mask(df["MTCI"] == np.inf, np.nan).mask(df["MTCI"] == -np.inf, np.nan).dropna(subset=["MTCI"], axis=0)
    df = df.mask(df["MTCI_diff"] == np.inf, np.nan).mask(df["MTCI_diff"] == -np.inf, np.nan).dropna(subset=["MTCI_diff"], axis=0)

    df.reset_index(inplace=True)

    print(df.columns, df.shape)

    # For each 3-week image, standarize each column
    #df = utilities.get_rm_outlier_standarize(df, NUMERIC_COLS, rm_outliers=False)
    return df, NUMERIC_COLS, NUM_SAMPLES