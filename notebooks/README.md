# Notebooks Overview for Rossmann Sales Forecasting

This section explains the structure and purpose of the notebooks used for preprocessing, exploratory data analysis (EDA), and sales analysis. Each notebook includes detailed code for specific tasks related to Rossmann sales forecasting.

---

## 1. **Preprocessing Notebook: `preprocess.ipynb`**

### Purpose

This notebook handles the data preprocessing tasks required to prepare the dataset for further analysis and forecasting. It includes tasks like loading datasets, handling missing values, merging datasets, and exporting the cleaned data.

### Key Sections

- **Importing Packages**:  
  Essential libraries like `pandas`, `numpy`, and `seaborn` are imported.

- **Loading the Data**:  
  The datasets (train, test, store) are loaded using the custom preprocessing script (`preprocessor.py`).

- **Missing Values Handling**:

  - Missing numeric values (e.g., `CompetitionDistance`) are filled with the median.
  - Certain columns (e.g., `Promo2SinceYear`) are filled with zero for missing values.
  - Categorical missing values (e.g., `PromoInterval`) are filled with 'Unknown'.

- **Merging Datasets**:  
  The store data is merged with the train dataset to include store-specific features.

- **Exporting Clean Data**:  
  The cleaned and merged data is exported for future use.

### Example Code

```python
# Handling missing values in store dataset
column_to_replace = ['CompetitionDistance']
store_df = psr.handle_missing_values_median(store_df, column_to_replace)

# Merging train and store datasets
train_store = psr.merge_datasets(train_df, store_df, 'Store')

# Exporting the merged dataset
train_store.to_csv('train_store.csv', index=False)
```

---

## 2. **EDA Notebook: `EDA_and_TSA.ipynb`**

### Purpose

This notebook performs exploratory data analysis (EDA) on the cleaned dataset to uncover patterns, trends, and correlations. It also includes time series decomposition and visualization.

### Key Sections

- **Rolling Statistics**:  
  Plots the rolling mean and standard deviation to observe trends and seasonality in sales over a 12-month window.

- **Time Series Decomposition**:  
  Decomposes the sales time series into trend, seasonality, and residuals to better understand underlying patterns.

- **Autocorrelation (ACF) and Partial Autocorrelation (PACF)**:  
  Plots the ACF and PACF to analyze lagged correlations in the sales data.

- **Sales Distribution**:  
  Visualizes the sales distribution across different stores and over time (by year).

- **Outlier Detection**:  
  Detects and visualizes outliers in the sales data using the Interquartile Range (IQR) method.

- **Sales Growth Rate**:  
  Calculates and visualizes the growth rate of sales over time.

- **Promo Effectiveness by Day of Week**:  
  Analyzes how promotions impact sales performance depending on the day of the week.

### Example Code

```python
# Plot rolling mean and standard deviation
eda.plot_rolling_statistics(df, window=12)

# Decompose the time series
eda.decompose_time_series(df, freq=365)

# Autocorrelation and Partial Autocorrelation plots
eda.plot_acf_pacf(df, lags=30)
```

---

## 3. **Sales Analysis Notebook: `salesAnalysis.ipynb`**

### Purpose

This notebook digs deeper into the sales data to analyze the impact of promotions, holidays, store types, and competitor activity on sales performance. It also investigates customer behavior and store-specific factors affecting sales.

### Key Sections

- **Promotion Distribution**:  
  Visualizes the distribution of promotions in both the training and test datasets.

- **Holiday Effects**:  
  Analyzes how holidays influence sales patterns before and after the holiday periods.

- **Seasonal Behavior**:  
  Investigates seasonal trends, such as Christmas and Easter, and their impact on sales.

- **Day of Week Sales**:  
  Visualizes sales performance across different days of the week.

- **Sales and Customer Correlation**:  
  Analyzes the correlation between sales and the number of customers.

- **Promo Effect on Sales**:  
  Investigates the effect of promotions (Promo1 and Promo2) on sales across different weekdays and months.

- **Store Type Performance**:  
  Compares the sales performance of different store types, both with and without promotions.

- **Competitor Effects**:  
  Analyzes how competitor distance and store openings affect sales performance.

### Example Code

```python
# Analyze promo distribution in the training set
sa.promo_distribution(train_store)

# Effect of promotions on sales by day of the week
sa.promo_effect_on_sales(train_store)

# Effect of competitor distance on sales
sa.competitor_distance_sales(train_store)

# Analyze seasonal sales behavior (e.g., Christmas)
sa.seasonal_sales_behavior(train_store)
```

---

### Recommended Workflow

1. **Preprocessing Notebook**  
   Start with the preprocessing notebook to clean and prepare the dataset for analysis. Handle missing values and merge relevant datasets.

2. **EDA Notebook**  
   Perform exploratory data analysis using the EDA notebook. Investigate sales patterns, trends, and correlations to better understand the data.

3. **Sales Analysis Notebook**  
   Finally, use the sales analysis notebook to gain deeper insights into factors affecting sales, such as promotions, holidays, and competition.

---

## 4. **Sales Prediction Notebook: `store_sales_prediction.ipynb`**

Data Loading: Load the Rossmann sales dataset and preprocess it using preprocessing.py.
Model Training: Train the Random Forest and LSTM models by calling the respective scripts (train_random_forest.py and train_lstm.py).
Model Comparison: Compare the performance of Random Forest and LSTM models using RMSE and visualize the results.
Post-Prediction Analysis: Use the functions from post_prediction_analysis.py to plot feature importance, estimate confidence intervals, and visualize the predicted vs actual sales.
How to Use:
Ensure that all the required Python scripts are located in the scripts/ folder.
Launch the notebook from the command line:
bash
Copy code
jupyter notebook notebook/rossmann_sales_analysis.ipynb
Follow the steps in the notebook to run the entire analysis pipeline, from data preprocessing to model evaluation and post-prediction analysis.
