import os
from os.path import join, isfile
from glob import glob
import time
import joblib 
import pickle 
import yaml 
import pandas as pd
import numpy as np 
import dask.dataframe as dd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import rasterio

from sklearn.model_selection import train_test_split
from params import define_parameters

def pickle_write(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def joblib_write(model, filename):
    joblib.dump(model, filename)

def pickle_read(filename):
    with open(filename, 'rb') as file:
        loaded_model_pickle = pickle.load(file) 
    return loaded_model_pickle

def joblib_read(filename):
    return joblib.load(filename) 

def performance(y_train, y_train_pred,y_vali, y_val_pred):
        # Calculate additional regression metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_vali, y_val_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_vali, y_val_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_vali, y_val_pred)

    train_rmse = np.sqrt(train_mse)

    val_rmse = np.sqrt(val_mse)

    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'MSE', 'R2'],
    'Train': [train_rmse, train_mae, train_mse, train_r2],
    'Validation': [val_rmse, val_mae, val_mse, val_r2]
    })
    return metrics_df

# use pandas as i can fit the data into memory 
def train_catboost(pq_files, model_dir,sample_frac=1):#, sample_frac=1.0):
    ti = time.perf_counter()
    print('Loading parameters...')

    # Assuming define_parameters is a function that returns all necessary parameters
    rnd_seed, roi, mx, fcolydi, fcolyid, fcolref, fcolX, catboost_params, nboost, fcolY, FTCOLSC = define_parameters()
    numF = len(fcolX)

    print('Loading datasets...')
    # ddf = dd.read_parquet(pq_files, columns=FTCOLSC)
    # ddf = ddf.astype('float32')
    # ddf[fcolyid] = ddf[fcolref] - ddf[fcolydi]

    # # Shuffle the data
    # #ddf = ddf.sample(frac=sample_frac, random_state=42)#.shuffle()
    # #ddf = ddf.shuffle(on=fcolydi)#, random_state=42)


    # print('Splitting datasets...')
    # dtrain, dvalid = ddf.random_split([0.9, 0.1], random_state=42)#0.8,0.2
    # dtrain = dtrain.compute()
    # dvalid = dvalid.compute()
    # del ddf 
    # numR = len(dtrain)

    df = pd.read_parquet(pq_files, columns=FTCOLSC)
    df = df.astype('float32')
    df[fcolyid] = df[fcolref] - df[fcolydi]

    # Shuffle the data
    df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # Split the dataset
    print('Splitting datasets...')
    dtrain, dvalid = train_test_split(df, test_size=0.3, random_state=42) # do 30
    dtrain = df.copy()
    print(dtrain.shape, dvalid.shape)

    # Get the number of rows in the training set
    numR = len(dtrain)
  
    print('TRAINING...')
    for fcolYr in fcolY:
        print(f'TRAINING {fcolYr}')
        wdir_roi = join(model_dir, roi)
        os.makedirs(wdir_roi, exist_ok=True)
        
        # Drop NaN values in the target column
        dvalid = dvalid.dropna(subset=[fcolYr])
        dtrain = dtrain.dropna(subset=[fcolYr])

        X_vali = dvalid.drop(fcolY, axis=1)
        y_vali = dvalid[fcolYr]
        val_data = Pool(X_vali, label=y_vali)

        X_train = dtrain.drop(fcolY, axis=1)
        y_train = dtrain[fcolYr]
        train_data = Pool(X_train, label=y_train)

        model = CatBoostRegressor(**catboost_params)
        model.fit(train_data, eval_set=val_data, verbose=100)
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_vali)
    
        metrics_df = performance(y_train, y_train_pred, y_vali, y_val_pred)
        
        # Extract key parameters from the trained model
        #learning_rate = model.get_param('learning_rate')
        #depth = model.get_param('depth')
        #num_boost = model.get_param('iterations')
        #l2_leaf_reg = model.get_param('l2_leaf_reg')
        #loss_function = model.get_param('loss_function')
        #random_seed = model.get_param('random_seed')
        
        # Construct the output filename
        #outname = f'CTB_{roi}_{learning_rate}_{depth}_{num_boost}_{l2_leaf_reg}_{loss_function}_{random_seed}_{fcolYr}_{numR}_{numF}_{rnd_seed}_{mx}'
        outname = f'CTB_{roi}_{nboost}_{fcolYr}_{numR}_{numF}_{rnd_seed}'
        # Save metrics and model
        metrics_df.to_csv(join(wdir_roi, f'{outname}.csv'))
        model.save_model(join(wdir_roi, f'{outname}.cbm'))
        pickle_write(model, join(wdir_roi, f'{outname}.pkl'))
  
    tf = time.perf_counter() - ti 
    print('='*30)
    print(f'Run time = {tf/60:.2f} min(s)')



def get_file_size_in_gb(file_path):
    """
    Returns the size of the file at the given path in gigabytes (GB).

    :param file_path: Path to the file
    :return: Size of the file in gigabytes
    """
    try:
        # Get the file size in bytes
        file_size_bytes = os.path.getsize(file_path)
        # Convert bytes to gigabytes
        file_size_gb = file_size_bytes / (1024 ** 3)
        print(f"File size: {file_size_gb:.2f} GB")
        return file_size_gb
    except OSError as e:
        print(f"Error accessing file: {e}")
        return None
  


def load_yaml(file_path, keys_to_keep=None):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    if keys_to_keep:
        data = {key: data[key] for key in keys_to_keep if key in data}
    return data

def load_patch_to_df(paths, names):
    arrays = []
    min_size = None
    for path in paths:
        with rasterio.open(path) as src:
            array = src.read(1).flatten()  # Read the first band and flatten it
            if min_size is None:
                min_size = array.size
            else:
                min_size = min(min_size, array.size)
            arrays.append(array)
    # Truncate arrays to the minimum size found
    arrays = [array[:min_size] for array in arrays]
    stacked_arrays = np.column_stack(arrays)
    new_df = pd.DataFrame(stacked_arrays, columns=names)
    return new_df





########################################################################
########################################################################
########################################################################

# def get_patch(df, idx):
#     return df.loc[idx:idx,:]

# def get_patch_paths(df, idx,cols=None):
#     if cols is None:
#         return df.loc[idx:idx,:].values[0].tolist()
#     else:
#         df = df[cols].copy()
#         return df.loc[idx:idx,:].values[0].tolist()


# def tif2df(patch_paths,cols):
#     ds = Raster(patch_paths)
#     ds.names = cols
#    # tb = ds.to_pandas()
#     return  ds.to_pandas()

# def process_path(i, valid_paths, FTCOLSC):
#     patch_paths = get_patch_paths(valid_paths, i, FTCOLSC)
#     di = tif2df(patch_paths, FTCOLSC)
#     return di[FTCOLSC]

# def load_all_data(paths, FTCOLSC):
#     dilist = []
#     for i in range(len(paths)):
#         dilist.append(process_path(i,paths, FTCOLSC))
#     # Concatenate the dataframes
#     dfi = pd.concat(dilist, ignore_index=True)
#     return dfi

