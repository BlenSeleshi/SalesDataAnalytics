import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

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
    
    # Extract day types
    df['MonthPhase'] = pd.cut(df['Day'], bins=[0, 10, 20, 31], labels=['Beginning', 'Mid', 'End'])
    
    # Competition-related features
    df['CompetitionOpenSince'] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear, 
                                                     month=df.CompetitionOpenSinceMonth, day=15), errors='coerce')
    df['CompetitionDaysOpen'] = (df['Date'] - df['CompetitionOpenSince']).dt.days
    df['CompetitionDaysOpen'] = df['CompetitionDaysOpen'].apply(lambda x: x if x > 0 else 0)
    
    # Promotion-related features
    df['Promo2Since'] = pd.to_datetime(dict(year=df.Promo2SinceYear, week=df.Promo2SinceWeek, day=1), errors='coerce')
    df['Promo2DaysActive'] = (df['Date'] - df['Promo2Since']).dt.days
    df['Promo2DaysActive'] = df['Promo2DaysActive'].apply(lambda x: x if x > 0 else 0)
    
    # Feature: Days to/after holiday
    holiday_dates = df[df['StateHoliday'] != '0']['Date'].unique()
    df['DaysToNextHoliday'] = df['Date'].apply(lambda x: (holiday_dates - x).min().days)
    df['DaysAfterHoliday'] = df['Date'].apply(lambda x: (x - holiday_dates).min().days)
    
    #Storage age days
    df['Store_age_days'] = (df['Date'] - pd.to_datetime(df['Store_open_date'])).dt.days
    
    # Generate lagged features
    df = df.sort_values(['Store', 'Date'])
    df['Sales_lag_7'] = df.groupby('Store')['Sales'].shift(7)
    df['Sales_lag_14'] = df.groupby('Store')['Sales'].shift(14)
    df['Sales_lag_30'] = df.groupby('Store')['Sales'].shift(30)
    
    # Rolling window features
    df['Sales_roll_mean_7'] = df.groupby('Store')['Sales'].shift(1).rolling(window=7).mean()
    df['Sales_roll_std_7'] = df.groupby('Store')['Sales'].shift(1).rolling(window=7).std()
    logging.info("Feature extraction completed")
    
    # Separate numerical and categorical columns
    # numerical_columns = ['Store', 'Customers', 'Open', 'Promo', 'Year', 'Month', 'Day', 
    #                      'DayOfWeek', 'IsWeekend', 'IsMonthStart', 'IsMonthEnd', 
    #                      'CompetitionDaysOpen', 'Promo2DaysActive', 'DaysToNextHoliday', 'DaysAfterHoliday','Sales_lag_7', 'Sales_lag_14', 'Sales_lag_30',
    #                      'Sales_roll_mean_7', 'Sales_roll_std_7', 'Store_age_days','CompetitionDistance']
    # categorical_columns = ['StateHoliday', 'StoreType', 'Assortment','MonthPhase']
    
    # X = df[numerical_columns + categorical_columns]
    # y = df['Sales']
    return df
   
def difference_series(df, lag=1):
    """
    Apply differencing to ensure stationarity for time series modeling.
    """
    logging.info("Applying differencing to time series.")
    df['Sales_diff'] = df.groupby('Store')['Sales'].diff(lag)
    df.dropna(inplace=True)
    return df   

# function to encode and scale data
def create_pipeline(numerical_columns, categorical_columns):
    logging.info("Starting encoding and scaling")
    
    # Column transformer to scale numerical data and one-hot encode categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ]
    )
    
    logging.info("Encoding and scaling completed")
    return preprocessor

def train_test_split_custom(df, test_size=0.2):
    """
    Perform train-test split ensuring temporal order is preserved.
    """
    logging.info("Performing train-test split.")
    train_df = df.iloc[:-int(test_size*len(df))]
    test_df = df.iloc[-int(test_size*len(df)):]
    
    X_train = train_df.drop('Sales', axis=1)
    y_train = train_df['Sales']
    X_test = test_df.drop('Sales', axis=1)
    y_test = test_df['Sales']
    
    return X_train, X_test, y_train, y_test

# Merge train/test datasets with store.csv
def merge_datasets(df, store_df, on_column):
    logging.info("Merging the two datasets....")
    return pd.merge(df, store_df, how='inner',on=on_column)