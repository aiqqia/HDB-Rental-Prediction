import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

# Suppress the FutureWarning related to is_categorical_dtype from TargetEncoder
warnings.filterwarnings("ignore", category=FutureWarning)

################################################
#               Preprocessing
################################################

def clean_data(data):
    cleaned_data = data
    cleaned_data = cleaned_data.drop(columns=['furnished', 'elevation', 'town', 'block', 'street_name', 'planning_area', 'subzone'])
    cleaned_data['flat_type'] = cleaned_data['flat_type'].str.replace(r'(2|3|4|5)-room|(\d) room', r'\1\2', regex=True)
    cleaned_data['flat_type'] = cleaned_data['flat_type'].str.replace('executive', '6')
    cleaned_data['flat_type'] = cleaned_data['flat_type'].astype(int)
    cleaned_data['rent_approval_date'] = cleaned_data['rent_approval_date'].str[2:].str.replace('-', '', regex=False)
    cleaned_data['rent_approval_date'] = cleaned_data['rent_approval_date'].astype(int)
    
    return cleaned_data

################################################
#                 Aux Data
################################################

def add_aux_data_count_in_radius(training_data_raw, training_coords, col_name, aux_data_raw, radius):
    geom_list_aux = [Point(lon,lat) for lon,lat in zip(aux_data_raw["longitude"], aux_data_raw["latitude"])]
    gdf_aux = gpd.GeoDataFrame(aux_data_raw, geometry=geom_list_aux, crs="EPSG:4326")

    # this uses the right projection to get the distance in m scale
    gdf_aux.to_crs(epsg=3414, inplace=True)
    aux_coords = np.array(gdf_aux.geometry.apply(lambda point: (point.x, point.y)).tolist())

    aux_tree = BallTree(aux_coords, leaf_size=20)
    
    # Perform the query
    count_aux_within_radius = aux_tree.query_radius(training_coords, r=radius, count_only=True)
    training_data_raw[col_name] = count_aux_within_radius

    return training_data_raw

def add_aux_data_nearest_dist(training_data_raw, training_coords, col_name, aux_data_raw):
    geom_list_aux = [Point(lon,lat) for lon,lat in zip(aux_data_raw["longitude"], aux_data_raw["latitude"])]
    gdf_aux = gpd.GeoDataFrame(aux_data_raw, geometry=geom_list_aux, crs="EPSG:4326")

    # this uses the right projection to get the distance in m scale
    gdf_aux.to_crs(epsg=3414, inplace=True)
    aux_coords = np.array(gdf_aux.geometry.apply(lambda point: (point.x, point.y)).tolist())

    aux_tree = BallTree(aux_coords, leaf_size=20)

    aux_distances, _ = aux_tree.query(training_coords, k=1)  # k=1 for finding the nearest point
    training_data_raw[col_name] = aux_distances

    return training_data_raw

################################################
#               Stock Data
################################################

def get_stock_data(average_monthly_data ,stock_name, year, month):
    return average_monthly_data.loc[(stock_name, year, month)]

def chunk(nameslist):
    for i in range(0, len(nameslist), 10):
        yield nameslist[i:i+10]

def normalize(group):
    min_val = group.min()
    max_val = group.max()
    group = (group - min_val) / (max_val - min_val)
    return group

def add_stock_data(org_dataset, is_test=False):
    stockdata = pd.read_csv("auxiliary-data/sg-stock-prices.csv")

    stockdata['date'] = pd.to_datetime(stockdata['date'])
    stockdata['year'], stockdata['month'] = stockdata['date'].dt.year, stockdata['date'].dt.month
    average_monthly_data = stockdata.groupby(['name', 'year', 'month']).mean(numeric_only=True).reset_index()

    names = list(set(stockdata['name']))

    average_monthly_data['normalized_value'] = average_monthly_data.groupby('name')['adjusted_close'].transform(normalize)

    stockdata_pivot = average_monthly_data.pivot_table(index=['year', 'month'], columns='name', values='adjusted_close').reset_index()
    stockdata_pivot['year'] = stockdata_pivot['year'].astype(int)
    stockdata_pivot['month'] = stockdata_pivot['month'].astype(int)

    org_dataset[['year', 'month']] = org_dataset['rent_approval_date'].str.split('-', expand=True)
    org_dataset['year'] = org_dataset['year'].astype(int)
    org_dataset['month'] = org_dataset['month'].astype(int)

    merged = pd.merge(org_dataset, stockdata_pivot, on=['year', 'month'], how='left')

    # Use interpolation to fill NaN values for each stock column
    for stock in average_monthly_data['name'].unique():
        merged[stock] = merged[stock].interpolate(method='nearest').ffill().bfill()

    pos_corr_stocks = ['Keppel',
    'Flex',
    'Jardine Cycle & Carriage',
    'Singapore Airlines',
    'Golden Agri-Resources',
    'OCBC Bank',
    'Genting Singapore',
    'DBS Group',
    'Singtel',
    'Sembcorp',
    'UOB']

    neg_corr_stocks = ['Great Eastern',
    'SATS',
    'Sea (Garena)',
    'Mapletree Industrial Trust',
    'Mapletree Commercial Trust',
    'Singapore Post',
    'Grab Holdings',
    'Yanlord',
    'Singapore Land',
    'Karooooo',
    'Riverstone Holdings',
    'ComfortDelGro',
    'IGG Inc',
    'Triterras',
    'Keppel REIT',
    'ASLAN Pharmaceuticals']

    # merged['average_stock_value'] = merged[names].mean(axis=1)
    merged['highest_pos_corr'] = merged[pos_corr_stocks].mean(axis=1)
    # merged['highest_neg_corr'] = merged[neg_corr_stocks].mean(axis=1)
    merged = merged.drop(names, axis=1)
    merged = merged.drop(['year', 'month'], axis=1)
    return merged

################################################
#           XGBoost Preprocessing
################################################

region_ohe = OneHotEncoder(sparse=False)
fm_ohe = OneHotEncoder(sparse=False)

def one_hot_encode(X, istest=False):
    # Prepare Model
    if not istest:
        region_ohe.fit(X[['region']])
    
    tr1 = region_ohe.transform(X[['region']])
    tr2 = pd.DataFrame(tr1, columns=region_ohe.get_feature_names_out(['region']))
    tr3 = pd.concat([X.reset_index(drop=True), tr2.reset_index(drop=True)], axis=1)
    tr3 = tr3.drop(columns=["region"])

    return tr3

def ohe_fm(X, istest=False):
    # Prepare Model
    if not istest:   
        fm_ohe.fit(X[['flat_model']])
    
    tr1 = fm_ohe.transform(X[['flat_model']])
    tr2 = pd.DataFrame(tr1, columns=fm_ohe.get_feature_names_out(['flat_model']))
    tr3 = pd.concat([X.reset_index(drop=True), tr2.reset_index(drop=True)], axis=1)
    tr3 = tr3.drop(columns=["flat_model"])

    return tr3

def add_aux_data(X):
    df_schools = pd.read_csv('auxiliary-data/sg-primary-schools.csv')
    gep_schools = ["Anglo-Chinese School (Primary)", "Catholic High School (Primary)", "Henry Park Primary School",
              "Nan Hua Primary School", "Nanyang Primary School", "Raffles Girls' Primary School", "Rosyth School",
              "St. Hilda's Primary School", "Tao Nan School"]
    df_gep_schools = df_schools[df_schools["name"].isin(gep_schools)]
    df_malls = pd.read_csv('auxiliary-data/sg-shopping-malls.csv')
    df_mrts = pd.read_csv('auxiliary-data/sg-mrt-existing-stations.csv')

    geom_list_training = [Point(lon,lat) for lon,lat in zip(X["longitude"], X["latitude"])]
    gdf_training = gpd.GeoDataFrame(X, geometry=geom_list_training, crs="EPSG:4326")
    # this uses the right projection to get the distance in m scale
    gdf_training.to_crs(epsg=3414, inplace=True)
    training_coords = np.array(gdf_training.geometry.apply(lambda point: (point.x, point.y)).tolist())

    X = add_aux_data_count_in_radius(X, training_coords,
                                                     'mrts_within_3km', df_mrts, 3000)
    X = add_aux_data_nearest_dist(X, training_coords, 'nearest_distance_to_mrt',
                                                  df_mrts)
    training_data_raw = X
    training_data_raw = add_aux_data_count_in_radius(training_data_raw, training_coords,
                                                     'pri_schs_within_6km', df_schools, 6000)
    training_data_raw = add_aux_data_count_in_radius(training_data_raw, training_coords,
                                                     'gep_schs_within_5km', df_gep_schools, 5000)
    training_data_raw = add_aux_data_count_in_radius(training_data_raw, training_coords,
                                                     'malls_within_3km', df_malls, 3000)
    training_data_raw = add_aux_data_count_in_radius(training_data_raw, training_coords,
                                                     'mrts_within_3km', df_mrts, 3000)

    training_data_raw = add_aux_data_nearest_dist(training_data_raw, training_coords, 'nearest_distance_to_gep',
                                                  df_gep_schools)
    training_data_raw = add_aux_data_nearest_dist(training_data_raw, training_coords, 'nearest_distance_to_mall',
                                                  df_malls)
    training_data_raw = add_aux_data_nearest_dist(training_data_raw, training_coords, 'nearest_distance_to_mrt',
                                                  df_mrts)
    return training_data_raw

def preprocess_xgboost(X, istest=False, encoder='target'):
    X = add_aux_data(X)
    X = add_stock_data(X)
    X = clean_data(X)
    X = one_hot_encode(X, istest)
    
    if encoder == 'ohe':
        X = ohe_fm(X, istest)
    elif encoder == 'label':
        X = label_encode(X)
    else:
        X = target_encode(X, istest)
    return X

t_encoder = TargetEncoder()

def target_encode(X, istest=False):
    if not istest:
        t_encoder.fit(X['flat_model'], X['monthly_rent'])
    X['flat_model'] = t_encoder.transform(X['flat_model'])
    return X

################################################
#      XGBoost Training Data Preparation
################################################

def prep_data_for_xgboost(enc='target'):
    training_data_raw = pd.read_csv('train.csv')
    X_train, X_val = train_test_split(training_data_raw, test_size=0.2, random_state=42)
    
    X_train = preprocess_xgboost(X_train, encoder=enc)
    X_train, y_train = X_train.drop('monthly_rent', axis=1), X_train[['monthly_rent']]
    X_val = preprocess_xgboost(X_val, True, enc)
    X_val, y_val = X_val.drop('monthly_rent', axis=1), X_val[['monthly_rent']]
    
    print("Shape of training data: ", X_train.shape)
    print("Shape of training label: ", y_train.shape)
    print("Shape of validation data: ", X_val.shape)
    print("Shape of validation label: ", y_val.shape)

    return X_train, y_train, X_val, y_val