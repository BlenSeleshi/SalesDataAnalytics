import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(train_path, test_path, store_path):
    logging.info("Loaded datastes from file...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    store = pd.read_csv(store_path)
    return train, test, store
    

# Function to display missing values and their percentage in the DataFrame
def missing_values_table(df):
    logging.info("Displaying Missing Value Percentages for Each Column....")
    mis_val = df.isnull().sum()
    
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    mis_val_dtype = df.dtypes
    
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis = 1)
    
    mis_val_table_ren_columns = mis_val_table.rename (
        columns={0: 'Missing Values', 1: '% of Total Values', 2: 'DType'}
    )
    
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
    
    print ("The dataframe has " + str(df.shape[1]) + "columns.\n"
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.\n")
           
    return mis_val_table_ren_columns
    
# Handle missing values (two strategies: median or zero)
def handle_missing_values_median(df, columns):
    """
    Fills missing values with the median for the specified columns.
    :param df: DataFrame to process
    :param columns: List of column names to fill missing values with the median
    :return: DataFrame with missing values filled
    """
    logging.info("Replacing Missing Values with Median....")
    for col in columns:
        df[col].fillna(df[col].median(), inplace=True)
    return df

def handle_missing_values_zero(df, columns):
    """
    Fills missing values with zero for the specified columns.
    :param df: DataFrame to process
    :param columns: List of column names to fill missing values with zero
    :return: DataFrame with missing values filled
    """
    logging.info("Replacing missing values with 0....")
    for col in columns:
        df[col].fillna(0, inplace=True)
    return df

# Function to fill missing values for categorical columns with 'Unknown'
def handle_missing_categorical(df, columns, fill_value='Unknown'):
    logging.info("Replacing Missing Values with 'Unknown'....")
    for column in columns:
        if column in df.columns:
            df[column].fillna(fill_value, inplace=True)
            print(f"Missing values in categorical column '{column}' filled with '{fill_value}'")
    return df

# Feature engineering for dates
def extract_date_features(df):
    """
    Extracts useful date features from the given date column.
    """
    logging.info("Extracting Date Features from the dataset....")

    # Handle missing values
    # Remove rows where stores are closed or have 0 sales but are open
    df = df[(df['Open'] == 1) & (df['Sales'] > 0)]

    # Feature extraction from Date column
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
        # Create holiday features
    df['IsStateHoliday'] = df['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})
    # Extract day types
    df['MonthPhase'] = pd.cut(df['Day'], bins=[0, 10, 20, 31], labels=['Beginning', 'Mid', 'End'])
    
    # calulate the number of days the store was opened
        # Ensure 'Open' column is numeric
    df['Open'] = pd.to_numeric(df['Open'])
    
    # Calculate total open days for each store
    total_open_days = df[df['Open'] == 1].groupby('Store')['Date'].nunique().reset_index()
    total_open_days.columns = ['Store', 'TotalOpenDays']
    
    # Merge TotalOpenDays back into the original dataframe
    df = df.merge(total_open_days, on='Store', how='left')
    
    df['StateHoliday'] = df['StateHoliday'].astype('category')
    df['Assortment'] = df['Assortment'].astype('category')
    df['StoreType'] = df['StoreType'].astype('category')
    df['PromoInterval']= df['PromoInterval'].astype('category')
    df['MonthPhase'] = df['MonthPhase'].astype('category')
    
    df['StateHoliday_cat'] = df['StateHoliday'].cat.codes
    df['Assortment_cat'] = df['Assortment'].cat.codes
    df['StoreType_cat'] = df['StoreType'].cat.codes
    df['PromoInterval_cat'] = df['PromoInterval'].cat.codes
    df['MonthPhase_cat'] = df['MonthPhase'].cat.codes
    
    df['StateHoliday_cat'] = df['StateHoliday_cat'].astype('float')
    df['Assortment_cat'] = df['Assortment_cat'].astype('float')
    df['StoreType_cat'] = df['StoreType_cat'].astype('float')
    df['PromoInterval_cat'] = df['PromoInterval_cat'].astype('float')
    df['MonthPhase_cat'] = df['MonthPhase_cat'].astype('float')
    
    df = pd.get_dummies(df, columns=["Assortment", "StoreType","PromoInterval","MonthPhase"], prefix=["is_Assortment", "is_StoreType","is_PromoInteval","MonthPhase"])
    

    del df['StateHoliday']
    del df['StateHoliday_cat'] 
    del df['Assortment_cat']
    del df['StoreType_cat'] 
    del df['PromoInterval_cat'] 
    del df['MonthPhase_cat'] 
    

    logging.info("Feature extraction completed")
    return df

def difference_series(df, lag=1):
    """
    Apply differencing to ensure stationarity for time series modeling.
    """
    logging.info("Applying differencing to time series.")
    df['Sales_diff'] = df.groupby('Store')['Sales'].diff(lag)
    df.dropna(inplace=True)
    return df   

import pandas as pd
from statsmodels.tsa.stattools import adfuller

def create_supervised_data(series, window):
    """
    Create sliding window supervised data for LSTM.
    """
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:(i + window)])
        y.append(series[i + window])
    return np.array(X), np.array(y)

def difference_data(df, column, diff_order=1):
    """
    Perform differencing to make the time series stationary.
    """
    df[f'{column}_diff'] = df[column].diff(diff_order)
    return df

def check_stationarity(series):
    """
    Perform the Augmented Dickey-Fuller test to check for stationarity.
    """
    result = adfuller(series.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    return result[1] < 0.05  # Returns True if the data is stationary

# Merge train/test datasets with store.csv
def merge_datasets(df, store_df, on_column):
    logging.info("Merging the two datasets....")
    return pd.merge(df, store_df, how='inner',on=on_column)