import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from distinctipy import distinctipy # get n distinct colors

import numpy as np
import pandas as pd
import math
import scipy
from scipy.stats import norm

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

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
    df = df.sort_values(by=['pt_idx'], ascending=True)
    df = df.sort_values(by=['img_idx'], ascending=True)
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

def get_rm_outlier_standarize(df:pd.DataFrame, NUMERIC_COLS: list[str], standarize:bool=True, rm_outliers:bool=True)->pd.DataFrame:
    res_df = None
    for sample_idx in df.sample_idx.unique():
        curr_df = df[df.sample_idx == sample_idx]
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