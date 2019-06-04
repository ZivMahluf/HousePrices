import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from visualize_data import ZivsVisualizer as zv
from process_data import ZivsProcessor as zp
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

    # # get highest corr features, and plot heatmap by it
    # highest_corr_features = data.corr().nlargest(12, 'SalePrice')['SalePrice'].index
    # ZivsVisualizer.save_heatmap(data[highest_corr_features])
    #
    # # go over the pair plot and decide which shows the best linear data
    # ZivsVisualizer.pair_plot(data, highest_corr_features)

    highest_corr_features = data.corr().nlargest(12, 'SalePrice')['SalePrice'].index
    print(highest_corr_features)
    new_data = data[highest_corr_features]

    # -------- correcting ------
    zp.correct_extreme_values(data, 'GrLivArea', 0, 4000)  # GrLivArea > 4000
    zp.correct_extreme_values(data, 'GarageCars', 0, 3)
    zp.correct_extreme_values(data, 'GarageArea', 0, 1200)
    zp.correct_extreme_values(data, 'TotalBsmtSF', 0, 3000)
    zp.correct_extreme_values(data, '1stFlrSF', 0, 3000)
    zp.correct_extreme_values(data, 'TotRmsAbvGrd', 0, 13)

    # --------- completing -------
    # # check how many null values each column have
    # zv.print_and_return_cols_with_null(new_data)
    # print('-' * 10)

    # GarageYrBlt has 81 null values, all else 0
    zp.complete_missing_data(new_data, 'GarageYrBlt', 0)

    # # validate
    # zv.print_and_return_cols_with_null(new_data)
    # print('-' * 10)

    # --------- creating -------
    new_data['TotalArea'] = new_data['GrLivArea'] + new_data['GarageArea']
    new_data['TotalSF'] = new_data['TotalBsmtSF'] + new_data['1stFlrSF']

    # --------- converting -----
    converted_features = ['OverallQual', 'GarageCars', 'FullBath']
    converted_data = zp.convert_features_with_label_encoder(new_data, converted_features)

    # # validate
    # zv.print_and_return_cols_with_null(new_data)
    # print('-' * 10)

    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(data['GarageArea'], data['SalePrice'])
    # plt.show()
