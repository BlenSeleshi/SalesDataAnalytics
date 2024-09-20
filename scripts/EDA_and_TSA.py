import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import logging

logging.basicConfig(level=logging.INFO)

# Load and preprocess dataset (add your file path here)
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    logging.info("Data loaded successfully.")
    return df


# Function 1: Rolling Mean and Standard Deviation
def plot_rolling_statistics(df, window=12):
    plt.figure(figsize=(12, 6))
    rolling_mean = df['Sales'].rolling(window=window).mean()
    rolling_std = df['Sales'].rolling(window=window).std()

    plt.plot(df['Date'], df['Sales'], label='Original Sales', color='blue')
    plt.plot(df['Date'], rolling_mean, label=f'{window}-Month Rolling Mean', color='red')
    plt.plot(df['Date'], rolling_std, label=f'{window}-Month Rolling Std', color='green')

    plt.title(f'Rolling Mean & Standard Deviation (Window={window})')
    plt.legend(loc='best')
    plt.show()


# Function 2: Decompose Time-Series
def decompose_time_series(df, freq=365):
    df = df.set_index('Date')
    decomposition = seasonal_decompose(df['Sales'], model='additive', period=freq)
    
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    plt.show()


# Function 3: ACF and PACF Plot
def plot_acf_pacf(df, lags=30):
    plt.figure(figsize=(14, 6))

    plt.subplot(121)
    plot_acf(df['Sales'], lags=lags, ax=plt.gca())
    plt.title("Autocorrelation")

    plt.subplot(122)
    plot_pacf(df['Sales'], lags=lags, ax=plt.gca())
    plt.title("Partial Autocorrelation")
    
    plt.show()


# Function 4: Sales Distribution by Store and Year
def sales_distribution_by_store_year(df):
    df['Year'] = df['Date'].dt.year
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(x='Year', y='Sales', hue='StoreType', data=df)
    plt.title("Sales Distribution by Year and Store Type")
    plt.xticks(rotation=45)
    plt.show()



# Function 6: Detect Outliers (IQR Method)
def detect_outliers_iqr(df):
    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df['Sales'] < lower_bound) | (df['Sales'] > upper_bound)]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(df['Sales'])
    plt.title("Sales Distribution with Outliers")
    plt.show()
    
    logging.info(f"Number of outliers detected: {outliers.shape[0]}")
    return outliers


# Function 7: Calculate Sales Growth Rate
def calculate_sales_growth(df):
    df['SalesGrowth'] = df['Sales'].pct_change()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='SalesGrowth', data=df)
    plt.title("Sales Growth Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales Growth Rate")
    plt.show()
    
    return df


# Function 8: Promotion Effectiveness by Weekday
def promo_effectiveness_by_weekday(df):
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    promo_sales = df[df['Promo'] == 1].groupby('DayOfWeek')['Sales'].mean()
    non_promo_sales = df[df['Promo'] == 0].groupby('DayOfWeek')['Sales'].mean()
    
    plt.figure(figsize=(12, 6))
    promo_sales.plot(kind='bar', color='blue', alpha=0.7, label='Promo')
    non_promo_sales.plot(kind='bar', color='red', alpha=0.7, label='Non-Promo')
    
    plt.title("Promotion Effectiveness by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Sales")
    plt.legend()
    plt.show()


# Function 9: Count of State Holidays
def count_state_holidays(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['StateHoliday'])
    plt.title("Count of State Holidays")
    plt.show()


# Function 10: Count of School Holidays
def count_school_holidays(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['SchoolHoliday'])
    plt.title("Count of School Holidays")
    plt.show()


# Function 11: Count of Store Types
def count_store_types(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['StoreType'])
    plt.title("Count of Store Types")
    plt.show()


# Function 12: Count of Assortment Types
def count_assortment_types(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['Assortment'])
    plt.title("Count of Assortment Types")
    plt.show()


# Function 13: Correlation with Sales (Top 10 Features)
def plot_top_correlations(df, target='Sales'):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate the correlation and sort values
    corr = numeric_df.corr()[target].sort_values(ascending=False).head(11)  # Top 10, including 'Sales' itself
    
    # Plot the correlation
    plt.figure(figsize=(10, 6))
    sns.barplot(x=corr.index, y=corr.values, palette='coolwarm')
    plt.title(f'Top 10 Features Correlated with {target}')
    plt.xticks(rotation=45)
    plt.show()


