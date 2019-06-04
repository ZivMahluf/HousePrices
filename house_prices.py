import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
import seaborn as sns

from visualize_data import ZivsVisualizer as zv
from process_data import ZivsProcessor as zp
from itertools import product
from sklearn import tree
from sklearn import feature_selection
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


if __name__ == '__main__':
    data = pd.read_csv('data/train.csv')
    # test_data = pd.read_csv('data/test.csv')

    # # get highest corr features, and plot heatmap by it
    # highest_corr_features = data.corr().nlargest(12, 'SalePrice')['SalePrice'].index
    # ZivsVisualizer.save_heatmap(data[highest_corr_features])
    #
    # # go over the pair plot and decide which shows the best linear data
    # ZivsVisualizer.pair_plot(data, highest_corr_features)

    highest_corr_features = data.corr().nlargest(12, 'SalePrice')['SalePrice'].index
    highest_corr_features = highest_corr_features.drop(labels=['GarageYrBlt'])
    # print(highest_corr_features)
    new_data = data[highest_corr_features]

    # -------- correcting ------
    zp.correct_extreme_values(new_data, 'GrLivArea', 0, 4000)  # GrLivArea > 4000
    zp.correct_extreme_values(new_data, 'GarageCars', 0, 3)
    zp.correct_extreme_values(new_data, 'GarageArea', 0, 1200)
    zp.correct_extreme_values(new_data, 'TotalBsmtSF', 0, 3000)
    zp.correct_extreme_values(new_data, '1stFlrSF', 0, 3000)
    zp.correct_extreme_values(new_data, 'TotRmsAbvGrd', 0, 13)

    # --------- completing -------
    # # check how many null values each column have
    # zv.print_and_return_cols_with_null(new_data)
    # print('-' * 10)

    # --------- creating -------
    new_data.assign(TotalArea=new_data['GrLivArea'] + new_data['GarageArea'])
    new_data.assign(TotalSF=new_data['TotalBsmtSF'] + new_data['1stFlrSF'])

    # --------- converting -----
    converted_features = ['GarageCars', 'FullBath']
    converted_data = zp.convert_features_with_label_encoder(new_data, converted_features)

    # # validate
    # zv.print_and_return_cols_with_null(converted_data)
    # print('-' * 10)

    # fit a model
    labels_of_converted = converted_data['SalePrice']
    converted_data.drop(columns=['SalePrice'], inplace=True)

    # # validate
    print('-' * 10)
    zv.print_and_return_cols_with_null(converted_data)
    print('-' * 10)

    # reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    # reg.fit(converted_data, labels_of_converted)
    # y_pred = reg.predict(converted_data)
    # score
    # print(math.sqrt(mean_squared_error(labels_of_converted, y_pred)))
