from glob import glob 
import os 
from os import remove
from os.path import basename,splitext
from catboost import CatBoostRegressor 
import joblib
import pickle
import json 
import pandas as pd 
from utilsml import load_patch_to_df, load_yaml
# need a better downxk
import numpy as np
import rasterio
from lightgbm import LGBMRegressor
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

def coarsen_raster_ds(fi_fpath,fo_fpath, scale_factor=3, resampling_method='bilinear'):
    #output_raster_path = fi_fpath.replace('.tif', '_COARSE.tif')
    with rasterio.open(fi_fpath) as src:
        # Calculate the new shape
        new_height = src.height // scale_factor
        new_width = src.width // scale_factor

        # Read the data
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling[resampling_method]
        )

        # Scale image transform
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )

        # Update metadata
        metadata = src.meta.copy()
        metadata.update({
            'height': new_height,
            'width': new_width,
            'transform': transform
        })

        # Write the resampled raster to a new file
        with rasterio.open(fo_fpath, 'w', **metadata) as dst:
            dst.write(data)

def load_raster(file_path):
    """Load raster data and return the data array and transform."""
    with rasterio.open(file_path) as src:
        return src.read(1), src.transform

def resample_raster(src_file_path, dst_transform, dst_shape):
    """Resample raster data to match the destination resolution."""
    with rasterio.open(src_file_path) as src:
        with WarpedVRT(
            src,
            transform=dst_transform,
            height=dst_shape[0],
            width=dst_shape[1],
            resampling=Resampling.bilinear
        ) as vrt:
            return vrt.read(1)

def flatten_and_mask(ndvi_data, thermal_data):
    """Flatten arrays and remove NaN values."""
    ndvi_flat = ndvi_data.flatten()
    thermal_flat = thermal_data.flatten()
    mask = ~np.isnan(ndvi_flat) & ~np.isnan(thermal_flat)
    return ndvi_flat[mask], thermal_flat[mask]

def perform_regression(ndvi_flat, thermal_flat):
    """Perform regression using LightGBM."""
    model = LGBMRegressor(random_state=123) # make this do HPO and #P with thresholds
    model.fit(ndvi_flat.reshape(-1, 1), thermal_flat)
    return model

def predict_high_res_thermal(model, ndvi_data):
    """Apply the regression model to high-resolution NDVI data."""
    predicted_thermal = model.predict(ndvi_data.flatten().reshape(-1, 1))
    return predicted_thermal.reshape(ndvi_data.shape)

def save_raster(data, transform, file_path):
    """Save the raster data to a file."""
    with rasterio.open(
        file_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(data, 1)

def dem_downscaler(f_filepath, c_filepath, output_path):
    # make this run on GPU too
    """Overall function to process rasters and save the high-resolution thermal image."""
    # Load data
    ndvi_data, ndvi_transform = load_raster(f_filepath)
    thermal_data, thermal_transform = load_raster(c_filepath)

    # Resample thermal data
    thermal_resampled = resample_raster(c_filepath, ndvi_transform, ndvi_data.shape)

    # Flatten and mask data
    ndvi_flat, thermal_flat = flatten_and_mask(ndvi_data, thermal_resampled)

    # Perform regression
    model = perform_regression(ndvi_flat, thermal_flat)

    # Predict high-resolution thermal data
    predicted_thermal_image = predict_high_res_thermal(model, ndvi_data)

    # Save the result
    save_raster(predicted_thermal_image, ndvi_transform, output_path)


def subtract_rasters(tdem_fpath, mlid_fpath, output_fpath):
  """
  Subtracts the raster data of mlid_fpath from tdem_fpath and saves the result to output_fpath.

  Parameters:
  - tdem_fpath: str, path to the TDEM raster file.
  - mlid_fpath: str, path to the MLID raster file.
  - output_fpath: str, path where the output raster will be saved.
  """
  # Open the tdem and mlid rasters
  with rasterio.open(tdem_fpath) as tdem_src, rasterio.open(mlid_fpath) as mlid_src:
      # Ensure the rasters have the same dimensions and transform
      assert tdem_src.shape == mlid_src.shape, "Rasters must have the same dimensions"
      assert tdem_src.transform == mlid_src.transform, "Rasters must have the same transform"

      # Read the data from the rasters
      tdem_data = tdem_src.read(1)
      mlid_data = mlid_src.read(1)

      # Perform the subtraction
      result_data = tdem_data - mlid_data

      # Update metadata for the output file
      out_meta = tdem_src.meta.copy()
      out_meta.update({
          "driver": "GTiff",
          "height": tdem_src.height,
          "width": tdem_src.width,
          "transform": tdem_src.transform
      })

      # Write the result to a new file
      with rasterio.open(output_fpath, "w", **out_meta) as dest:
          dest.write(result_data, 1)

  print(f"Subtraction result saved to {output_fpath}")
  # null values treatment 


def load_tile_pdf(yaml_pattern,tilename,keys_to_keep,fcolydi,fcolyid, fcolref):
    yaml_files = glob(yaml_pattern)
    yaml_file = [i for i in yaml_files if tilename in i][0]
    ydict_f = load_yaml(yaml_file, keys_to_keep=keys_to_keep)
    paths = list(ydict_f.values())
    names = list(ydict_f.keys())
    assert len(paths) == len(names), 'paths != names'
    pdf =  load_patch_to_df(paths, names)
    pdf[fcolyid] = pdf[fcolref].subtract(pdf[fcolydi])
    return pdf,paths

def infer_model_format(model_path):
    """
    Infers the model format from the file extension.

    Parameters:
    - model_path: Path to the model file.

    Returns:
    - A string representing the model format ('json', 'joblib', 'pkl', 'cbm').
    """
    _, ext = splitext(model_path)
    ext = ext.lower()
    if ext == '.json':
        return 'json'
    elif ext == '.joblib':
        return 'joblib'
    elif ext == '.pkl':
        return 'pkl'
    elif ext == '.cbm':
        return 'cbm'
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def load_and_predict(model_path, X_test):
    """
    Loads a model from a given path, makes predictions, and returns the predictions.

    Parameters:
    - model_path: Path to the model file.
    - X_test: Data to make predictions on.

    Returns:
    - predictions: Predictions made by the model.
    """
    model_format = infer_model_format(model_path)

    if model_format == 'json' or model_format == 'cbm':
        model = CatBoostRegressor()
        model.load_model(model_path, format=model_format)
    elif model_format == 'joblib':
        model = joblib.load(model_path)
    elif model_format == 'pkl':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"Unsupported model format: {model_format}")

    predictions = model.predict(X_test)
    return predictions

def predict_di_and_id(pdf,fcolY,fcolX,modelpaths,paths):
    tdem_pattern = "tdem_DEM_EF_filled.tif"
    ldem_pattern = "multi_DTM_LiDAR.tif"
    tdem_fpath = [i for i in paths if tdem_pattern in i][0]
    ldem_fpath = [i for i in paths if ldem_pattern in i][0]
    
    #ndf = pd.DataFrame()  # New DataFrame to store predictions
    raster_out_list = []
    for fcolYr in fcolY:
        print(fcolYr)
        models_list= [i for i in modelpaths if fcolYr in i]
        for modelpath in models_list:
            modelbname = basename(modelpath)[:-4]
            print(modelbname)
            print(f'Passed: {modelpath}')
            prediction = load_and_predict(modelpath, pdf[fcolX])
            #ndf[modelbname] = prediction
            metadata = rio_get_meta(tdem_fpath)
            predml_fpath = tdem_fpath.replace(tdem_pattern, f'_{modelbname}.tif')
            print(predml_fpath)
            raster_out_list.append(predml_fpath)

            width, height = metadata['width'], metadata['height']  # Ensure these match the original raster
            reshaped_data = prediction.reshape((height, width))
            write_raster(predml_fpath, reshaped_data, metadata)

    raster_out_list = sorted(raster_out_list)
    mldi_fpath = raster_out_list[0]
    mlzdif_fpath = raster_out_list[1]
    mlid_fpath = mlzdif_fpath.replace('_zdif_', '_Mzdif_')
    subtract_rasters(tdem_fpath, mlzdif_fpath, mlid_fpath)
    ml_paths = [mldi_fpath, mlid_fpath]
    return ml_paths,tdem_fpath


def downscalling_va(ml_paths,tdem_fpath):
    # _vb Me
  fc_filepath_list = []
  for fi in ml_paths:
      fo = fi.replace('.tif', '_COARSE.tif')
      coarsen_raster_ds(fi, fo)
      c_filepath= fo 
      f_filepath = tdem_fpath
      fc_filepath = fo.replace('_COARSE.tif', '__DXK.tif')
      
      print(fc_filepath)
      dem_downscaler(f_filepath, c_filepath, output_path=fc_filepath) # better
      fc_filepath_list.append(fo)

  for f in fc_filepath_list: 
      remove(f)

def prediction_workflow(yaml_pattern,tilename,keys_to_keep,fcolydi,fcolyid, fcolref,fcolY,fcolX,modelpaths):
    # from keys_to_keep extract fcolydi,fcolyid, fcolref,fcolY,fcolX
    pdf,paths = load_tile_pdf(yaml_pattern,tilename,keys_to_keep,fcolydi,fcolyid, fcolref)
    ml_paths,tdem_fpath = predict_di_and_id(pdf,fcolY,fcolX,modelpaths,paths)
    downscalling_va(ml_paths,tdem_fpath)


def write_raster(raster_path, data, metadata, block_size=256):
  height, width = data.shape
  dtype = data.dtype

  with rasterio.open(
      raster_path,
      'w',
      driver='GTiff',
      height=height,
      width=width,
      count=1,
      dtype=dtype,
      crs=metadata['crs'],
      transform=metadata['transform'],  # Use the original transform
      tiled=True,  # Enable tiling
      compress='lzw',
      blockxsize=block_size,  # Set block size for x
      blockysize=block_size   # Set block size for y
  ) as dst:
      for i in range(0, height, block_size):
          for j in range(0, width, block_size):
              # Calculate the block size
              block_height = min(block_size, height - i)
              block_width = min(block_size, width - j)
              
              # Extract the block
              block = data[i:i+block_height, j:j+block_width]
              
              # Write the block
              dst.write(block, 1, window=((i, i+block_height), (j, j+block_width)))


def rio_get_meta(raster_path):
    with rasterio.open(raster_path) as src:
            metadata = src.meta
    return metadata
