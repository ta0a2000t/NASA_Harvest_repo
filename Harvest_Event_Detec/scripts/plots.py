import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from scripts import utilities

def get_fig_seperability_coef(coef_function, df: pd.DataFrame, col_names: list[str], title: str) -> plt.figure:
    
    har_df = df[df.har_evnt]
    not_har_df = df[df.har_evnt == False]
    y, x = [], []
    for name in col_names:
        arr1 = har_df[name]
        arr2 = not_har_df[name]

        # remove nan values
        arr1, arr2 = arr1[~np.isnan(arr1)], arr2[~np.isnan(arr2)]

        coef = coef_function(arr1, arr2)
        
        y.append(coef)
        x.append(name)

    plotting_df = pd.DataFrame({"class": np.array(x), "value": np.array(y).astype(float)})
    plotting_df = plotting_df.sort_values(by=['value'], ascending=True)
    fig = plt.figure(figsize=(16,len(col_names)))
    sns.barplot(plotting_df, x='value', y='class').set_title(title)
    plt.close()

    return fig



def plot_per_period(df: pd.DataFrame, col_names: list[str], filename: str, metric:int):
    """ 
    filename example: "../plots/box_plots/everything_trimmed_standarized.pdf"
    
    metric: 0 -> Bhattacharyya
        1 -> Hellinger Distance
        2 -> Root Mean Squared Error (RMSE)
        3 -> Cosine Similarity
        4 -> Euclidean Distance
        5 -> Mean Absolute Error (MAE)

    """
    coef_function = None
    title = "NO TITLE"
    if metric == 0:
        coef_function = utilities.get_Bhattacharyya_coef
        title = "Bhattacharyya"
    elif metric == 1:
        coef_function = utilities.get_Hellinger_distance
        title = "Hellinger Distance"
    elif metric == 2:
        coef_function = utilities.get_RMSE_coef
        title = "Root Mean Squared Error (RMSE)"
    elif metric == 3:
        coef_function = utilities.get_cosine_similarity_coef
        title = "Cosine Similarity"
    elif metric == 4:
        coef_function = utilities.get_euclidean_dist
        title = "Euclidean Distance"
    elif metric == 5:
        coef_function = utilities.get_MAE_coef
        title = "Mean Absolute Error (MAE)"
    pp = PdfPages(filename)
    for sampleIdx in df.sample_idx.unique():
        curr_df = df[df.sample_idx == sampleIdx]
        fig = get_fig_seperability_coef(coef_function, df, col_names, title=(title +' | ' + str(sampleIdx)))
        
        fig.savefig(pp, format='pdf')
        plt.close() # closing figure
    pp.close()
