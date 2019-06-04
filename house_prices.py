import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from visualize_data import ZivsVisualizer
from process_data import ZivsProcessor
from itertools import product
from sklearn import tree
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

if __name__ == '__main__':
    data = pd.read_csv('data/train.csv')
    # test_data = pd.read_csv('data/test.csv')

    # # convert string to int of: LotFrontage, MasVnrArea, GarageYrBlt
    # data['LotFrontage'] = pd.to_numeric(data['LotFrontage'], errors='coerce')
    # data['MasVnrArea'] = pd.to_numeric(data['MasVnrArea'], errors='coerce')
    # data['GarageYrBlt'] = pd.to_numeric(data['GarageYrBlt'], errors='coerce')

    # get highest corr features, and plot heatmap by it
    highest_corr_features = data.corr().nlargest(12, 'SalePrice')['SalePrice'].index
    ZivsVisualizer.save_heatmap(data[highest_corr_features])

    # go over the pair plot and decide which shows the best linear data
    ZivsVisualizer.pair_plot(data, highest_corr_features)

    # then eliminate their values





