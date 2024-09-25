# Rossmann Sales Forecasting - README

## Overview

This repository contains a comprehensive analysis of the Rossmann sales dataset aimed at predicting and understanding sales patterns for Rossmann Pharmaceuticals. It includes notebooks for preprocessing, exploratory data analysis (EDA), and sales analysis, alongside various scripts for data handling, model training, and evaluation. The project focuses on time series forecasting and delves into factors impacting sales, such as promotions, holidays, and competition.

---

## Notebooks Overview

### 1. **Preprocessing Notebook: `preprocess.ipynb`**

#### Purpose

Handles the data preprocessing tasks required to prepare the dataset for further analysis and forecasting. This includes loading datasets, handling missing values, merging datasets, and exporting the cleaned data.

#### Key Sections

- **Importing Packages**: Essential libraries like `pandas`, `numpy`, and `seaborn`.
- **Loading the Data**: Loads the datasets (train, test, store) using the custom preprocessing script (`preprocessor.py`).
- **Missing Values Handling**: Various strategies for handling missing values in numeric and categorical columns.
- **Merging Datasets**: Merges store data with the train dataset.
- **Exporting Clean Data**: Exports the cleaned and merged data.

#### Example Code

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

### 2. **EDA Notebook: `EDA_and_TSA.ipynb`**

#### Purpose

Performs exploratory data analysis (EDA) to uncover patterns, trends, and correlations in the cleaned dataset. Includes time series decomposition and visualization.

#### Key Sections

- **Rolling Statistics**: Plots rolling mean and standard deviation.
- **Time Series Decomposition**: Decomposes sales into trend, seasonality, and residuals.
- **ACF and PACF Analysis**: Analyzes lagged correlations.
- **Sales Distribution**: Visualizes sales distribution by store and time.
- **Outlier Detection**: Detects and visualizes outliers.
- **Promo Effectiveness**: Analyzes promotions’ impact on sales.

#### Example Code

```python
# Plot rolling mean and standard deviation
eda.plot_rolling_statistics(df, window=12)

# Decompose the time series
eda.decompose_time_series(df, freq=365)

# Autocorrelation and Partial Autocorrelation plots
eda.plot_acf_pacf(df, lags=30)
```

---

### 3. **Sales Analysis Notebook: `sales_analysis_notebook.ipynb`**

#### Purpose

Analyzes the impact of promotions, holidays, store types, and competitor activity on sales performance.

#### Key Sections

- **Promotion Distribution**: Visualizes distribution of promotions.
- **Holiday Effects**: Analyzes sales patterns related to holidays.
- **Seasonal Behavior**: Investigates seasonal trends.
- **Day of Week Sales**: Compares sales across days of the week.
- **Sales and Customer Correlation**: Analyzes correlation between sales and customers.
- **Store Type Performance**: Compares performance across store types.

#### Example Code

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

### 4. **Sales Prediction Notebook: `store_sales_prediction.ipynb`**

#### Purpose

This notebook focuses on predicting sales using machine learning models, specifically Random Forest and LSTM.

#### Key Sections

- **Data Loading**: Loads the dataset and preprocesses it.
- **Model Training**: Trains both Random Forest and LSTM models using respective scripts.
- **Model Comparison**: Compares performance using RMSE.
- **Post-Prediction Analysis**: Visualizes prediction results and feature importance.

#### How to Use

Ensure that all required scripts are located in the `scripts/` folder, then launch the notebook from the command line:

```bash
jupyter notebook notebook/store_sales_prediction.ipynb
```

---

## Scripts Overview

### 1. **Preprocessing Script: `preprocessing.py`**

#### Purpose

Cleans the dataset and prepares it for analysis, handling missing values and merging datasets.

#### Key Functions

- **`load_data(train_path, test_path, store_path)`**: Loads datasets.
- **`handle_missing_values_median(df, columns)`**: Fills missing numeric values with the median.
- **`merge_datasets(df, store_df, on_column)`**: Merges datasets.

#### How to Use

```python
from preprocessing import load_data, handle_missing_values_median, merge_datasets

# Load datasets
train, test, store = load_data('train.csv', 'test.csv', 'store.csv')

# Handle missing values
train = handle_missing_values_median(train, ['CompetitionDistance'])
```

---

### 2. **EDA Script: `eda.py`**

#### Purpose

Provides functions to visualize the dataset and gain insights.

#### Key Functions

- **`plot_rolling_statistics(df, window=12)`**: Plots rolling statistics.
- **`decompose_time_series(df, freq=365)`**: Decomposes sales time series.

#### How to Use

```python
from eda import plot_rolling_statistics, decompose_time_series

# Plot rolling statistics
plot_rolling_statistics(train, window=12)
```

---

### 3. **Sales Analysis Script: `sales_analysis.py`**

#### Purpose

Analyzes specific aspects like promotions and holidays.

#### Key Functions

- **`promo_distribution(df)`**: Analyzes promo distribution.
- **`analyze_holiday_effects(df)`**: Visualizes holiday effects on sales.

#### How to Use

```python
from sales_analysis import promo_distribution

# Analyze promo distribution
promo_distribution(train)
```

---

### 4. **Model Training Scripts**

#### Random Forest Model: `random_forest_model.py`

- **Functions**: Sets up and tunes a Random Forest model.

#### LSTM Model: `lstm_model.py`

- **Functions**: Builds and trains an LSTM model for time-series forecasting.

---

### 5. **Model Serialization: `model_serialization.py`**

#### Purpose

Handles model serialization and loading.

#### Key Functions

- **`serizalize_model(model, filepath)`**: Saves the model.
- **`load_model(filepath)`**: Loads a model.

---

### 6. **Post-Prediction Analysis: `post_prediction_analysis.py`**

#### Purpose

Contains functions for analyzing model predictions.

#### Key Functions

- **`plot_feature_importance(model, feature_names)`**: Plots feature importance.
- **`visualize_predictions(predicted_sales, actual_sales)`**: Visualizes predictions vs actual sales.

---

## Recommended Workflow

1. **Preprocessing**: Clean and prepare the dataset using `preprocessing.py`.
2. **EDA**: Explore the dataset with `eda.py` to visualize trends and correlations.
3. **Sales Analysis**: Use `sales_analysis.py` for deeper insights into sales factors.
4. **Model Training**: Train models using `random_forest_model.py` and `lstm_model.py`.
5. **Model Evaluation**: Analyze predictions and model performance with `post_prediction_analysis.py`.

---

## Dependencies

- **Python**: Version 3.8+
- **Libraries**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `tensorflow`
  - `matplotlib`
  - `statsmodels`
