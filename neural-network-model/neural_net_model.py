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
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings

# Suppress the FutureWarning related to is_categorical_dtype from TargetEncoder
warnings.filterwarnings("ignore", category=FutureWarning)

class MyTargetEncoder():
    def __init__(self, columns_to_target_encode, training_data):
        self.encoders = {}
        self.columns_to_target_encode = columns_to_target_encode
        for col in columns_to_target_encode:
            encoder = TargetEncoder()
            encoder.fit(training_data[col], training_data['monthly_rent'])
            self.encoders[col] = encoder
        
    def fit_data(self, encoded_data):
        for col, encoder in self.encoders.items():
            encoded_data[col] = encoder.transform(encoded_data[col])
        return encoded_data


def clean_data(data):
    cleaned_data = data
    # cleaned_data = cleaned_data.drop_duplicates(subset=None, keep='first', inplace=False)
    cleaned_data = cleaned_data.drop(columns=['furnished', 'elevation', 'town', 'block', 'street_name', 'planning_area', 'subzone'])
    cleaned_data['flat_type'] = cleaned_data['flat_type'].str.replace(r'(2|3|4|5)-room|(\d) room', r'\1\2', regex=True)
    cleaned_data['flat_type'] = cleaned_data['flat_type'].str.replace('executive', '6')
    cleaned_data['flat_type'] = cleaned_data['flat_type'].astype(int)
    cleaned_data['rent_approval_date'] = cleaned_data['rent_approval_date'].str[2:].str.replace('-', '', regex=False)
    cleaned_data['rent_approval_date'] = cleaned_data['rent_approval_date'].astype(int)
    
    return cleaned_data


def encode_data(train_org, training_cleaned, valid_cleaned, testing_cleaned):
    # First Target Encoding
    
    columns_to_target_encode = ['flat_model']
    myTargetEncoder = MyTargetEncoder(columns_to_target_encode, train_org)
    
    training_encoded = myTargetEncoder.fit_data(training_cleaned)
    valid_encoded = myTargetEncoder.fit_data(valid_cleaned)
    testing_encoded = myTargetEncoder.fit_data(testing_cleaned)
    
    # Now, One-Hot Encoding
    
    # Prepare Model
    myOneHotEncoder = OneHotEncoder(sparse=False)
    myOneHotEncoder.fit(training_encoded[['region']])
    
    # Fit on train data
    tr1 = myOneHotEncoder.transform(training_encoded[['region']])
    tr2 = pd.DataFrame(tr1, columns=myOneHotEncoder.get_feature_names_out(['region']))
    tr3 = pd.concat([training_encoded.reset_index(drop=True), tr2.reset_index(drop=True)], axis=1)

    training_encoded = tr3.drop(columns=["region"])
    
    # Fit on valid data
    va1 = myOneHotEncoder.transform(valid_encoded[['region']])
    va2 = pd.DataFrame(va1, columns=myOneHotEncoder.get_feature_names_out(['region']))
    va3 = pd.concat([valid_encoded.reset_index(drop=True), va2.reset_index(drop=True)], axis=1)

    valid_encoded = va3.drop(columns=["region"])
    
    # Fit on test data
    te1 = myOneHotEncoder.transform(testing_encoded[['region']])
    te2 = pd.DataFrame(te1, columns=myOneHotEncoder.get_feature_names_out(['region']))
    te3 = pd.concat([testing_encoded.reset_index(drop=True), te2.reset_index(drop=True)], axis=1)

    testing_encoded = te3.drop(columns=["region"])
    
    return training_encoded, valid_encoded, testing_encoded


def scale_data(training_encoded, validation_encoded, testing_encoded):
    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_encoded)
    validation_scaled = scaler.fit_transform(validation_encoded)
    testing_scaled = scaler.fit_transform(testing_encoded)
    return training_scaled, validation_scaled, testing_scaled

def preprocess_data(train_org, training_data_raw, valid_data_raw, testing_data_raw):
    
    training_cleaned = clean_data(training_data_raw)
    valid_cleaned = clean_data(valid_data_raw)
    testing_cleaned = clean_data(testing_data_raw)
    
    training_encoded, valid_encoded, testing_encoded = encode_data(train_org, training_cleaned, valid_cleaned, testing_cleaned)

    return training_encoded, valid_encoded, testing_encoded


class CustomDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x = torch.tensor(x_train, dtype=torch.float32)
        self.y = torch.tensor(y_train, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        features = self.x[idx]
        label = self.y[idx]

        return features, label
    
class TestDataset(Dataset):
    def __init__(self, test_data):
        self.data = test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        features = torch.FloatTensor(sample)  # Assuming the features are in a list or NumPy array
        return features

def get_data_loaders(X_train, y_train, X_val, y_val, X_test, batch_size):
    train_dataset = CustomDataset(X_train, y_train.to_numpy())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    valid_dataset = CustomDataset(X_val, y_val.to_numpy())
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TestDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader


class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32,1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


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

def add_aux_data(org_dataset):
    # Add auxiliary data
    df_schools = pd.read_csv('../auxiliary-data/sg-primary-schools.csv')
    gep_schools = ["Anglo-Chinese School (Primary)", "Catholic High School (Primary)", "Henry Park Primary School",
              "Nan Hua Primary School", "Nanyang Primary School", "Raffles Girls' Primary School", "Rosyth School",
              "St. Hilda's Primary School", "Tao Nan School"]
    df_gep_schools = df_schools[df_schools["name"].isin(gep_schools)]
    df_malls = pd.read_csv('../auxiliary-data/sg-shopping-malls.csv')
    df_mrts = pd.read_csv('../auxiliary-data/sg-mrt-existing-stations.csv')

    # org_dataset is either raw training or raw test data
    geom_list = [Point(lon,lat) for lon,lat in zip(org_dataset["longitude"], org_dataset["latitude"])]
    gdf_data = gpd.GeoDataFrame(org_dataset, geometry=geom_list, crs="EPSG:4326")
    # this uses the right projection to get the distance in m scale
    gdf_data.to_crs(epsg=3414, inplace=True)
    coords = np.array(gdf_data.geometry.apply(lambda point: (point.x, point.y)).tolist())

    org_dataset = add_aux_data_count_in_radius(org_dataset, coords,
                                                'pri_schs_within_6km', df_schools, 6000)
    org_dataset = add_aux_data_count_in_radius(org_dataset, coords,
                                                'gep_schs_within_5km', df_gep_schools, 5000)
    org_dataset = add_aux_data_count_in_radius(org_dataset, coords,
                                                'malls_within_3km', df_malls, 3000)
    org_dataset = add_aux_data_count_in_radius(org_dataset, coords,
                                                'mrts_within_3km', df_mrts, 3000)

    org_dataset = add_aux_data_nearest_dist(org_dataset, coords, 'nearest_distance_to_gep',
                                                  df_gep_schools)
    org_dataset = add_aux_data_nearest_dist(org_dataset, coords, 'nearest_distance_to_mall',
                                                  df_malls)
    org_dataset = add_aux_data_nearest_dist(org_dataset, coords, 'nearest_distance_to_mrt',
                                                  df_mrts)
    return org_dataset


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
    stockdata = pd.read_csv("../auxiliary-data/sg-stock-prices.csv")

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

    merged['highest_pos_corr'] = merged[pos_corr_stocks].mean(axis=1)
    merged = merged.drop(names, axis=1)
    merged = merged.drop(['year', 'month'], axis=1)
    return merged

if __name__ == "__main__":
    
    training_data_raw = pd.read_csv('../train.csv')
    testing_data_raw = pd.read_csv('../test.csv')
    
    training_data_raw = add_aux_data(training_data_raw)
    testing_data_raw = add_aux_data(testing_data_raw)
    
    training_data_raw = add_stock_data(training_data_raw)
    testing_data_raw = add_stock_data(testing_data_raw)
    
    train_X, train_y = training_data_raw.drop('monthly_rent', axis=1), training_data_raw[['monthly_rent']]
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.1, random_state=42)
    X_train, X_val, testing_data = preprocess_data(training_data_raw, X_train, X_val, testing_data_raw)

    
    X_train, X_val, X_test = scale_data(X_train, X_val, testing_data)
    
    print("Shape of training data: ", X_train.shape)
    print("Shape of training label: ", y_train.shape)
    print("Shape of validation data: ", X_val.shape)
    print("Shape of validation label: ", y_val.shape)
    print("Shape of testing data: ", X_test.shape)
        
    train_loader, val_loader, test_loader = get_data_loaders(X_train, y_train, X_val, y_val, X_test, batch_size=8)
        
    input_size = X_train.shape[1]
    
    device = "mps"
    model = RegressionModel(input_size)
    model.load_state_dict(torch.load("pretrained_model.pth"))
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=1,
        gamma=0.9,
        verbose=False,
    )
    
    max_epochs = 30
    
    train_losses = []
    valid_losses = []
    
    min_loss = float("inf")
    
    for e in range(max_epochs):
        
        # TRAINING
        model.train()
        losses = []
        for inp, label in tqdm(train_loader):
            inp = inp.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            y_pred = model(inp)
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
 
        epoch_train_loss = np.mean(losses)
        
        # VALIDATING
        model.eval()
        losses = []
        with torch.no_grad():
            for inp, label in tqdm(val_loader):
                inp = inp.to(device)
                label = label.to(device)
                y_pred = model(inp)
                loss = criterion(y_pred, label)

                losses.append(loss.item())

        epoch_valid_loss = np.mean(losses)
        
        scheduler.step()
        
        print("Epoch {}: Training loss: {} Validation Loss: {}\n".format(e+1, math.sqrt(epoch_train_loss), math.sqrt(epoch_valid_loss)))
        
        if epoch_valid_loss < min_loss:
            min_loss = epoch_valid_loss
            torch.save(model.state_dict(), "final_model.pth")
        
    
    fin_model = RegressionModel(X_test.shape[1])
    fin_model.to(device)
    fin_model.load_state_dict(torch.load("final_model.pth"))
    
    model.eval()
    losses = []
    final_pred = []
    with torch.no_grad():
        for inp in tqdm(test_loader):
            inp = inp.to(device)
            y_pred = model(inp)
            final_pred.append(y_pred)

    print("Length of final predictions is: ", len(final_pred))
    combined_tensor = torch.cat(final_pred, dim=0)
    numpy_array = combined_tensor.cpu().numpy()
    flattened_array = numpy_array.flatten()
    ids = np.arange(30000)
    df = pd.DataFrame({'Id': ids, 'Predicted': flattened_array})

    df.to_csv("submission_neural_net.csv", index=False)
