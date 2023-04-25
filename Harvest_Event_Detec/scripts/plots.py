import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from scripts import utilities


def get_fig_seperability_coef(coef_function, colors_dic: dict, df: pd.DataFrame, col_names: list[str], title: str) -> plt.figure:
    
    har_df = df[df.har_evnt]
    not_har_df = df[df.har_evnt == False]
    y, x = [], []
    for name in col_names:
        arr1 = har_df[name].to_numpy()
        arr2 = not_har_df[name].to_numpy()

        # remove nan values
        arr1, arr2 = arr1[~np.isnan(arr1)], arr2[~np.isnan(arr2)]
        if(min(len(arr1), len(arr2)) == 0):
            continue
        coef = coef_function(arr1, arr2)
        if(type(coef) != type(None) and type(coef) != type(np.nan)):
            y.append(coef)
            x.append(name)
    if(len(y) == 0):
        return None

    plotting_df = pd.DataFrame({"class": np.array(x), "value": np.array(y).astype(float)})
    plotting_df = plotting_df.sort_values(by=['value'], ascending=True)
    fig = plt.figure(figsize=(16,len(col_names)))
    colors = [] # fixing a color on each class
    for name in plotting_df["class"].unique():
        colors.append(colors_dic[name])

    sns.barplot(plotting_df, x='value', y='class', palette=colors).set_title(title)
    ax = plt.gca()
    for i in ax.containers:
        ax.bar_label(i,)

    plt.close()

    return fig



def plot_per_period(colors_dic: dict, df: pd.DataFrame, col_names: list[str], filename: str, metric:int):
    """ 
    filename example: "../plots/box_plots/everything_trimmed_standarized.pdf"
    
    metric: 
        0 -> Bhattacharyya \n
        1 -> Hellinger Distance \n
        2 -> Two-sample Kolmogorov–Smirnov test \n
        3 -> Jensen-Shannon Divergence (JSD) \n
        __________ \n
        NOTUSED \n
        4 -> Euclidean Distance \n
        5 -> Mean Absolute Error (MAE) \n
        6 -> Root Mean Squared Error (RMSE) \n
        7 -> Cosine Similarity \n

    """
    # ["Euclidean Distance", "Mean Absolute Error (MAE)", , "Root Mean Squared Error (RMSE)", "Cosine Similarity"]
    titles = [ "Bhattacharyya Distance (MORE means separate)", "Hellinger Distance (MORE means separate)", 
                "Two-sample Kolmogorov–Smirnov test P-Value (LESS means separate)", "Jensen-Shannon Divergence (JSD) (LESS means separate)"]
    functions = [ utilities.get_Bhattacharyya_coef, utilities.get_Hellinger_distance, 
                utilities.get_K_S_Test, utilities.get_JSD_distance]
    title = titles[metric]
    coef_function = functions[metric]

    """ NOT USED
    elif metric == 4:
        coef_function = utilities.get_euclidean_dist
        title = "Euclidean Distance"
    elif metric == 5:
        coef_function = utilities.get_MAE_coef
        title = "Mean Absolute Error (MAE)"
    elif metric == 6:
        coef_function = utilities.get_RMSE_coef
        title = "Root Mean Squared Error (RMSE)"
    elif metric == 7:
        coef_function = utilities.get_cosine_similarity_coef
        title = "Cosine Similarity"
    """
    pp = PdfPages(filename)
    for sampleIdx in sorted(df.sample_idx.unique()):
        if(int(sampleIdx[1:]) < 6 or int(sampleIdx[1:]) > 9):
            continue # based on the box plots, we want sample idx within [6, 9]
        curr_df = df[df.sample_idx == sampleIdx]
        fig = get_fig_seperability_coef(coef_function,colors_dic, curr_df, col_names, title=(title +' | ' + str(sampleIdx)))
        if(type(fig) == type(None)):
            continue
        fig.savefig(pp, format='pdf')
        plt.close() # closing figure
    pp.close()



    


    