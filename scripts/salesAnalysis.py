# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the merged dataset

def load_data(file_path):
    logging.info("Loading the Dataset datastes from file...")
    return pd.read_csv(file_path)

# Checking distribution of promotions in training and test sets
def promo_distribution(df):
    logging.info("Calculating and plotting the distribution of promotion....")
    print("Promotion Distribution in Training Set:")
    promo_train_dist = df['Promo'].value_counts(normalize=True) * 100
    print(promo_train_dist)
    
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Promo')
    plt.title("Promotion Distribution in Training Set")
    plt.show()

# Analyze sales behavior around holidays
def analyze_holiday_effects(df):
    logging.info("Plotting the Sales behaviour round the holidays...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='StateHoliday', y='Sales', data=df)
    plt.title('Sales Behavior During Holidays')
    plt.xlabel('State Holiday')
    plt.ylabel('Sales')
    plt.show()
    
    # Exploring the trends before, during, and after holidays
    df['DayBeforeHoliday'] = (df['StateHoliday'] != '0') & (df['Date'].dt.dayofweek == 0)
    df['DayAfterHoliday'] = (df['StateHoliday'] != '0') & (df['Date'].dt.dayofweek == 2)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Sales', hue='DayBeforeHoliday', data=df[df['StateHoliday'] != '0'])
    plt.title('Sales Before and After Holidays')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

# Analyze seasonal behavior (Christmas, Easter, etc.)
def seasonal_sales_behavior(df):
    logging.info("Plotting the Seasonal Sales Behaviour...")
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    seasonal_sales = df.groupby('Month')['Sales'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=seasonal_sales, x='Month', y='Sales', palette='coolwarm')
    plt.title("Seasonal Sales Behavior")
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.show()
    
def plot_day_of_week_sales(df):
    logging.info("Plotting average sales by day of week...")
    df['DayOfWeek'] = df.index.dayofweek
    day_of_week_sales = df.groupby('DayOfWeek')['Sales'].mean()

    plt.figure(figsize=(10, 6))
    day_of_week_sales.plot(kind='bar')
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week (1=Monday, 7=Sunday)')
    plt.ylabel('Average Sales')
    plt.show()
    
# Statistics of sales on holidays and day of the week
def print_statistics(df):
    logging.info("Printing summary statistics...")
    print(df.groupby('DayOfWeek')['Sales'].describe())
    print("\nHoliday vs Non-Holiday Sales:")
    print(df.groupby('StateHoliday')['Sales'].describe())
    print("\n School Holiday vs Non-Holiday Sales:")
    print(df.groupby('SchoolHoliday')['Sales'].describe())
    

# Correlation between sales and number of customers
def sales_customers_correlation(df):
    logging.info("Analyzing the correlation between sales and customers....")
    corr, _ = pearsonr(df['Sales'], df['Customers'])
    print(f"Correlation between Sales and Customers: {corr:.2f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Customers', y='Sales', alpha=0.5)
    plt.title(f"Sales vs Customers (Correlation: {corr:.2f})")
    plt.show()

# Promo effect on sales and customers
def promo_effect_on_sales(df):
    logging.info("Plotting the effect of promotion on sales....")
    promo_sales = df.groupby('Promo')['Sales'].mean().reset_index()
    promo_customers = df.groupby('Promo')['Customers'].mean().reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=promo_sales, x='Promo', y='Sales', ax=ax[0], palette='viridis')
    ax[0].set_title("Effect of Promo on Sales")
    
    logging.info("Plotting the effect of promotion on customers...")
    sns.barplot(data=promo_customers, x='Promo', y='Customers', ax=ax[1], palette='viridis')
    ax[1].set_title("Effect of Promo on Customers")

    plt.tight_layout()
    plt.show()

# Store promo effectiveness analysis
def store_promo_effectiveness(df):
    
    logging.info("Plotting the effectiveness of promotion on based on store types...")
    promo_stores = df.groupby('Store')['Promo'].mean().reset_index()
    promo_sales = df.groupby('Store')['Sales'].mean().reset_index()

    df_promo = pd.merge(promo_stores, promo_sales, on='Store')
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_promo, x='Promo', y='Sales', alpha=0.6)
    plt.title("Store Promo Effectiveness (Promo % vs Sales)")
    plt.show()

# Customer behavior during store opening/closing times
def store_opening_closing_behavior(df):
    logging.info("Plotting the behaviour of customers on opening times...")
    open_sales = df.groupby('Open')['Sales'].mean().reset_index()

    plt.figure(figsize=(8, 6))
    sns.barplot(data=open_sales, x='Open', y='Sales', palette='coolwarm')
    plt.title("Sales During Store Open and Close Times")
    plt.show()

# Weekday sales analysis
def weekday_sales(df):
    
    logging.info("Analyzing weekly sales ...")
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
    
    logging.info("Plotting the Impact of Assortment on Sales...")
    assortment_sales = df.groupby('Assortment')['Sales'].mean().reset_index()

    plt.figure(figsize=(8, 6))
    sns.barplot(data=assortment_sales, x='Assortment', y='Sales', palette='plasma')
    plt.title("Effect of Assortment Type on Sales")
    plt.show()

# Competitor distance impact on sales
def competitor_distance_sales(df):
    
    logging.info("Plotting the Impact of Competitor Distance on Sales...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='CompetitionDistance', y='Sales', alpha=0.5)
    plt.title("Competition Distance vs Sales")
    plt.show()

# Competitor opening impact on sales
def competitor_opening_impact(df):
    
    logging.info("Plotting the Impact of Competitor Opening on Sales...")
    df['CompetitionOpenSinceYear'] = pd.to_datetime(df['CompetitionOpenSinceYear'], errors='coerce')
    comp_sales = df.groupby('CompetitionOpenSinceYear')['Sales'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=comp_sales, x='CompetitionOpenSinceYear', y='Sales')
    plt.title("Impact of Competitor Opening on Sales Over Time")
    plt.xticks(rotation=45)
    plt.show()
