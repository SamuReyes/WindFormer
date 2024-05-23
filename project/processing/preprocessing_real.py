import pandas as pd
import numpy as np
import pickle
import os
from utils.config_loader import load_config
import h5py

def process_raw_real_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Process raw data from a DataFrame to compute wind vector components and format time.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the raw wind data with columns for year ('y'),
                       month ('m'), day ('d'), hour ('h'), direction ('dir'), and velocity ('vel').

    Returns:
    pd.DataFrame: A DataFrame indexed by timestamp with columns for U and V components.
    """
    # Format date
    df['date'] = df['d'].astype(str) + '-' + df['m'].astype(str) + '-' + df['y'].astype(str) + ' ' + df['h'].astype(str)
    df['timestamp'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H')
    
    # Compute U and V components
    dir_rad = np.radians(df['dir'])
    df['U'] = -df['vel'] * np.sin(dir_rad)
    df['V'] = -df['vel'] * np.cos(dir_rad)
    
    df.set_index('timestamp', inplace=True)

    return df[['U', 'V']]

def get_anomalies(data:pd.DataFrame, geo_idx:list, u10_idx:int, v10_idx:int, climatologies:dict) -> pd.DataFrame:
    """
    Adjust wind data by subtracting climatological values to compute anomalies.

    Parameters:
    data (pd.DataFrame): DataFrame indexed by timestamp containing wind vector components 'U' and 'V'.
    geo_idx (tuple): Indices for the geographical location in the climatology data.
    u10_idx (int), v10_idx (int): Index for the U and V component in the climatology data.
    climatologies (dict): A dictionary of multi-dimensional array with climatological values.

    Returns:
    pd.DataFrame: An adjusted DataFrame with anomaly values for U and V components.
    """
    for index, row in data.iterrows():
        month = index.month
        day = index.day
        
        if not pd.isna(row['U']):
            row['U'] -= climatologies[month][day-1, geo_idx[0], geo_idx[1], u10_idx]
        if not pd.isna(row['V']):
            row['V'] -= climatologies[month][day-1, geo_idx[0], geo_idx[1], v10_idx]
    
    return data

def standarize_real_data(data:pd.DataFrame, u_mean:float, u_std:float, v_mean:float, v_std:float) -> pd.DataFrame:
    """
    Standardize the wind data by subtracting the mean and dividing by the standard deviation.

    Parameters:
    data (pd.DataFrame): DataFrame containing U and V wind components.
    u_mean (float), v_mean (float): The mean of the U and V component.
    u_std (float), v_std (float): The standard deviation of the U and V component.

    Returns:
    pd.DataFrame: A DataFrame with standardized U and V components.
    """
    data['U'] = (data['U'] - u_mean) / u_std
    data['V'] = (data['V'] - v_mean) / v_std
    
    return data

def load_reanalysis_data(data_path:str, start_year:int, end_year:int, geo_idx:list, u10_idx:int, v10_idx:int) -> pd.DataFrame:
    """
    Load wind vector components from reanalysis data stored in an HDF5 file for specified years and geographical indices.

    Parameters:
    data_path (str): The file path for the HDF5 file containing reanalysis data.
    start_year (int), end_year (int): Range of years of the data to load.
    geo_idx (tuple): Tuple of indices (latitude, longitude) to slice the geographical data.
    u10_idx, v10_idx (int): Indices for U and V wind components in the data array.

    Returns:
    pd.DataFrame: A DataFrame containing the U and V components indexed by hourly intervals over the specified years.
    """

    rea_list = []
    with h5py.File(data_path, 'r') as file:
        for year in range(start_year, end_year + 1):
            year_str = str(year)
            if year_str in file.keys():
                rea_data = file[year_str]['surface']['data'][:, geo_idx[0], geo_idx[1], u10_idx:v10_idx+1]
                rea_list.append(rea_data)

    reanalysis = np.concatenate(rea_list, axis=0)

    date_range = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31 23:00:00', freq='h')

    reanalysis_df = pd.DataFrame(reanalysis, columns=['U', 'V'], index=date_range)

    return reanalysis_df

def impute_real_data(data:pd.DataFrame, reanalysis:pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing data in the real dataset using corresponding values from the reanalysis data.

    Parameters:
    data (pd.DataFrame): DataFrame containing the actual real data with potential missing values in U and V.
    reanalysis (pd.DataFrame): DataFrame containing reanalysis data to use for imputation.

    Returns:
    pd.DataFrame: The DataFrame after imputation of missing values.
    """
    data['U'] = data['U'].combine_first(reanalysis['U'])
    data['V'] = data['V'].combine_first(reanalysis['V'])

    return data

def save_real_data(output_file_path:str, jara:pd.DataFrame, pena:pd.DataFrame, minon:pd.DataFrame, start_year:int, end_year:int, image_size:list, jara_idx:list, pena_idx:list, minon_idx:list, u10_idx:int, v10_idx:int):
    """
    Save processed wind data to an HDF5 file for each year, incorporating data from multiple locations.

    Parameters:
    output_file_path (str): The file path to save the HDF5 output.
    jara, pena, minon (pd.DataFrame): DataFrames for different locations.
    start_year (int), end_year (int): Range of years to process.
    image_size (tuple): Dimensions to initialize the dataset arrays.
    jara_idx, pena_idx, minon_idx (tuple): Indices for different geographical locations.
    u10_idx, v10_idx (int): Indices for U and V wind components in the data array.
    """
    with h5py.File(output_file_path, 'w') as output_file:
        for year in range(start_year, end_year + 1):
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31 23:00:00"
            date_range = pd.date_range(start=start_date, end=end_date, freq='h')
            
            data_array = np.zeros((len(date_range), *image_size))
            
            for df, idx in [(jara, jara_idx), (pena, pena_idx), (minon, minon_idx)]:
                year_df = df[df.index.year == year]
                
                for time in date_range:
                    if time in year_df.index:
                        lat_idx, lon_idx = idx
                        data_array[date_range.get_loc(time), lat_idx, lon_idx, u10_idx] = year_df.at[time, 'U']
                        data_array[date_range.get_loc(time), lat_idx, lon_idx, v10_idx] = year_df.at[time, 'V']
            
            year_group = output_file.require_group(str(year))
            surface_group = year_group.require_group('surface')
            surface_group.create_dataset('data', data=data_array)
        
    
def preprocess_real_data(config:dict):
    """
    Preprocess real weather data based on the specified configuration settings.

    This function orchestrates the loading, processing, and saving of real weather data, including handling missing values,
    standardizing data, and calculating anomalies using climatological and reanalysis data.

    Parameters:
    config (dict): Configuration dictionary containing paths, data locations, variable indices, and other necessary settings.
    """

    # Load configs
    years = config['real_data']['years']
    image_size = config['model']['image_size_2d']

    # Set paths
    real_data_path = os.path.join(config['global']['path'], config['global']['real_data_path'])
    reanalysis_data_path = os.path.join(config['global']['path'], config['global']['processed_data_path'], config['global']['data_file'])
    constants_path = os.path.join(config['global']['path'], config['global']['constants_path'])
    output_file_path = os.path.join(config['global']['path'], config['global']['processed_data_path'], 'real_data.hdf5')

    # Get climatologies
    climatologies_path = os.path.join(constants_path, 'climatologies.pickle')
    with open(climatologies_path, 'rb') as file:
        climatologies = pickle.load(file)['surface_climatologies']

    # Get index of lon and lat for each data point
    longitudes = np.load(os.path.join(constants_path, 'longitude.npy'))
    latitudes = np.load(os.path.join(constants_path, 'latitude.npy'))

    pena_lat = config['real_data']['locations']['pena']['latitude']
    pena_lon = config['real_data']['locations']['pena']['longitude']
    jara_lat = config['real_data']['locations']['jara']['latitude']
    jara_lon = config['real_data']['locations']['jara']['longitude']
    minon_lat = config['real_data']['locations']['minon']['latitude']
    minon_lon = config['real_data']['locations']['minon']['longitude']

    jara_idx = (np.where(latitudes == jara_lat)[0][0], np.where(longitudes == jara_lon)[0][0])
    pena_idx = (np.where(latitudes == pena_lat)[0][0], np.where(longitudes == pena_lon)[0][0])
    minon_idx = (np.where(latitudes == minon_lat)[0][0], np.where(longitudes == minon_lon)[0][0])

    # Get index for wind variables
    var_names = config['preprocessing']['surface_var_names'][1:]
    u10_idx = var_names.index('u10')
    v10_idx = var_names.index('v10')

    # Read real data
    jara = pd.read_csv(os.path.join(real_data_path, 'jara.csv'), names=['d', 'm', 'y', 'h', 'vel', 'dir'], header=None)
    minon = pd.read_csv(os.path.join(real_data_path, 'minon.csv'), names=['d', 'm', 'y', 'h', 'vel', 'dir'], header=None)
    pena = pd.read_csv(os.path.join(real_data_path, 'pena.csv'), names=['d', 'm', 'y', 'h', 'vel', 'dir'], header=None)

    # Mark not available data
    jara.loc[jara['vel'] == -99, ['vel','dir']] = np.nan
    minon.loc[minon['vel'] == -99, ['vel','dir']] = np.nan
    pena.loc[pena['vel'] == -99, ['vel','dir']] = np.nan

    # Process data
    jara = process_raw_real_data(jara)
    minon = process_raw_real_data(minon)
    pena = process_raw_real_data(pena)

    # Get anomalies
    jara = get_anomalies(jara, jara_idx, u10_idx, v10_idx, climatologies)
    minon = get_anomalies(minon, minon_idx, u10_idx, v10_idx, climatologies)
    pena = get_anomalies(pena, pena_idx, u10_idx, v10_idx, climatologies)
    
    # Standarize data
    jara = standarize_real_data(jara, jara.U.mean(), jara.U.std(), jara.V.mean(), jara.V.std())
    minon = standarize_real_data(minon, minon.U.mean(), minon.U.std(), minon.V.mean(), minon.V.std())
    pena = standarize_real_data(pena, pena.U.mean(), pena.U.std(), pena.V.mean(), pena.V.std())

    # Load reanalysis data
    pena_reanalysis = load_reanalysis_data(reanalysis_data_path, years[0], years[1], pena_idx, u10_idx, v10_idx)
    jara_reanalysis = load_reanalysis_data(reanalysis_data_path, years[0], years[1], jara_idx, u10_idx, v10_idx)
    minon_reanalysis = load_reanalysis_data(reanalysis_data_path, years[0], years[1], minon_idx, u10_idx, v10_idx)

    # Impute missing values with real data
    jara = impute_real_data(jara, jara_reanalysis)
    minon = impute_real_data(minon, minon_reanalysis)
    pena = impute_real_data(pena, pena_reanalysis)

    # Save data in hdf5 format
    save_real_data(output_file_path, jara, pena, minon, years[0], years[1], image_size, jara_idx, pena_idx, minon_idx, u10_idx, v10_idx)
