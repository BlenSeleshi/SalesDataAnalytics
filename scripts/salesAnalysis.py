# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load the merged dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Checking distribution of promotions in training and test sets
def promo_distribution(df):
    print("Promotion Distribution in Training Set:")
    promo_train_dist = df['Promo'].value_counts(normalize=True) * 100
    print(promo_train_dist)
    
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Promo')
    plt.title("Promotion Distribution in Training Set")
    plt.show()

# Analyze sales behavior around holidays
def holiday_sales_behavior(df, holiday_col='StateHoliday'):
    holiday_sales = df.groupby([holiday_col, 'Date'])['Sales'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=holiday_sales, x='Date', y='Sales', hue=holiday_col)
    plt.title("Sales Behavior Before, During, and After Holidays")
    plt.xticks(rotation=45)
    plt.show()

# Analyze seasonal behavior (Christmas, Easter, etc.)
def seasonal_sales_behavior(df):
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    seasonal_sales = df.groupby('Month')['Sales'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=seasonal_sales, x='Month', y='Sales', palette='coolwarm')
    plt.title("Seasonal Sales Behavior")
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.show()

# Correlation between sales and number of customers
def sales_customers_correlation(df):
    corr, _ = pearsonr(df['Sales'], df['Customers'])
    print(f"Correlation between Sales and Customers: {corr:.2f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Customers', y='Sales', alpha=0.5)
    plt.title(f"Sales vs Customers (Correlation: {corr:.2f})")
    plt.show()

# Promo effect on sales and customers
def promo_effect_on_sales(df):
    promo_sales = df.groupby('Promo')['Sales'].mean().reset_index()
    promo_customers = df.groupby('Promo')['Customers'].mean().reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=promo_sales, x='Promo', y='Sales', ax=ax[0], palette='viridis')
    ax[0].set_title("Effect of Promo on Sales")

    sns.barplot(data=promo_customers, x='Promo', y='Customers', ax=ax[1], palette='viridis')
    ax[1].set_title("Effect of Promo on Customers")

    plt.tight_layout()
    plt.show()

# Store promo effectiveness analysis
def store_promo_effectiveness(df):
    promo_stores = df.groupby('Store')['Promo'].mean().reset_index()
    promo_sales = df.groupby('Store')['Sales'].mean().reset_index()

    df_promo = pd.merge(promo_stores, promo_sales, on='Store')
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_promo, x='Promo', y='Sales', alpha=0.6)
    plt.title("Store Promo Effectiveness (Promo % vs Sales)")
    plt.show()

# Customer behavior during store opening/closing times
def store_opening_closing_behavior(df):
    open_sales = df.groupby('Open')['Sales'].mean().reset_index()

    plt.figure(figsize=(8, 6))
    sns.barplot(data=open_sales, x='Open', y='Sales', palette='coolwarm')
    plt.title("Sales During Store Open and Close Times")
    plt.show()

# Weekday sales analysis
def weekday_sales(df):
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    weekday_sales = df.groupby('DayOfWeek')['Sales'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=weekday_sales, x='DayOfWeek', y='Sales', palette='magma')
    plt.title("Average Sales by Day of the Week")
    plt.xlabel("Day of Week (0 = Monday, 6 = Sunday)")
    plt.ylabel("Average Sales")
    plt.show()

# Assortment type impact on sales
def assortment_sales(df):
    assortment_sales = df.groupby('Assortment')['Sales'].mean().reset_index()

    plt.figure(figsize=(8, 6))
    sns.barplot(data=assortment_sales, x='Assortment', y='Sales', palette='plasma')
    plt.title("Effect of Assortment Type on Sales")
    plt.show()

# Competitor distance impact on sales
def competitor_distance_sales(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='CompetitionDistance', y='Sales', alpha=0.5)
    plt.title("Competition Distance vs Sales")
    plt.show()

# Competitor opening impact on sales
def competitor_opening_impact(df):
    df['CompetitionOpenSinceYear'] = pd.to_datetime(df['CompetitionOpenSinceYear'], errors='coerce')
    comp_sales = df.groupby('CompetitionOpenSinceYear')['Sales'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=comp_sales, x='CompetitionOpenSinceYear', y='Sales')
    plt.title("Impact of Competitor Opening on Sales Over Time")
    plt.xticks(rotation=45)
    plt.show()
