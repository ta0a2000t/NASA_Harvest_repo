import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import math

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

def sort_by_points_images(df:pd.DataFrame) -> None:
    df['pt_idx'] = df.point_idx.apply(lambda x: int(x[1:]))
    df['img_idx'] = df.image_idx.apply(lambda x: int(x[1:]))
    df.sort_values(by=['pt_idx'], ascending=True)
    df.sort_values(by=['img_idx'], ascending=True)
    # sorted now
    

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
    stretch_numeric_df = pd.DataFrame({'value':np.array([]), 'class':np.array([]), 'har_evnt': np.array([]), 'sample_idx':np.array([])})
    for col_name in NUMERIC_COLS:
        curr_df = pd.DataFrame({'value': df[col_name], 'class':np.tile(np.array([col_name]), df.shape[0]), 'har_evnt':df['har_evnt'], 'sample_idx':df['sample_idx'], 'start_date':df['start_date']})
        stretch_numeric_df = pd.concat([stretch_numeric_df, curr_df])
    return stretch_numeric_df

def get_rmved_outliers(df:pd.DataFrame, NUMERIC_COLS: list[str], standarize:bool=False)->pd.DataFrame:
    res_df = None
    for sample_idx in df.sample_idx.unique():
        curr_df = df[df.sample_idx == sample_idx]
        for numeric_col_name in NUMERIC_COLS:
            # setting outliers to NaN
            curr_df[numeric_col_name] = rmved_outliers_iqr(curr_df[numeric_col_name])
            if(standarize):
                curr_df[numeric_col_name] = get_standarized_arr(curr_df[numeric_col_name])
        if(type(res_df) == type(None)):
            res_df = curr_df
        else:
            res_df = pd.concat([res_df, curr_df])
    return res_df


def downsample(arr, length) -> np.array:
    indices = np.random.choice(len(arr), length, replace=False)
    return np.take(arr, indices)

def get_Bhattacharyya_coefficient(arr1:np.array, arr2:np.array, num_buckets=50):
    def bhattacharyya(h1, h2):
        # credit: https://gist.github.com/3rd3 https://gist.github.com/jstadler/c47861f3d86c40b82d4c
        '''Calculates the Byattacharyya distance of two histograms.'''

        def normalize(h):
            return h / np.sum(h)

        return 1 - np.sum(np.sqrt(np.multiply(normalize(h1), normalize(h2))))

    hist1, _ = np.histogram(arr1, bins = num_buckets)
    hist2, _ = np.histogram(arr2, bins = num_buckets)
    
    return bhattacharyya(hist1, hist2)
    

    

    
    
    