import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

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