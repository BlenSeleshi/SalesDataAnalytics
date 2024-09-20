import pandas as pd
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    logging.info("Loading data from file...")
    df = pd.read_csv(file_path, parse_dates=['Date'])
    #df.set_index('Date', inplace=True)
    logging.info(f"Data loaded with shape {df.shape}")
    return df

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

# Feature engineering for dates
def extract_date_features(df, date_column):
    """
    Extracts useful date features from the given date column.
    :param df: DataFrame to process
    :param date_column: Name of the date column to extract features from
    :return: DataFrame with new date-related features
    """
    logging.info("Extracting Date Features from the dataset....")
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract Year, Month, Day, and Week of the Year
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['Day'] = df[date_column].dt.day
    df['WeekOfYear'] = df[date_column].dt.isocalendar().week
    
    return df

# Merge train/test datasets with store.csv
def merge_datasets(df, store_df, on_column):
    logging.info("Merging the two datasets....")
    return pd.merge(df, store_df, how='inner',on=on_column)