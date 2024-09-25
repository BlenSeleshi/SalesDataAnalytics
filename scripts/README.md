# Rossmann Sales Forecasting - Scripts Overview

This folder contains scripts for preprocessing, exploratory data analysis (EDA), and detailed sales analysis to help predict and understand Rossmann Pharmaceuticals' sales. Below is a breakdown of each script and its purpose, along with the sequence in which they should be used.

## 1. **Preprocessing Script: `preprocessing.py`**

### Purpose

The preprocessing script is responsible for cleaning the dataset and preparing it for analysis. This script handles missing values, feature extraction from date columns, and merges the datasets (train, test, and store).

### Key Functions

- **`load_data(train_path, test_path, store_path)`**  
  Loads the train, test, and store datasets.

- **`missing_values_table(df)`**  
  Displays the missing values and their percentage for each column.

- **`handle_missing_values_median(df, columns)`**  
  Fills missing numeric values with the median for the specified columns.

- **`handle_missing_values_zero(df, columns)`**  
  Fills missing numeric values with zero.

- **`handle_missing_categorical(df, columns, fill_value='Unknown')`**  
  Fills missing values in categorical columns with 'Unknown' or a specified value.

- **`extract_date_features(df, date_column)`**  
  Extracts useful date-related features such as Year, Month, Day, and Week from a date column.

- **`merge_datasets(df, store_df, on_column)`**  
  Merges the main dataset (train/test) with the store dataset on a common column, typically `Store`.

### How to Use

```python
from preprocessing import load_data, handle_missing_values_median, extract_date_features, merge_datasets

# Load datasets
train, test, store = load_data('train.csv', 'test.csv', 'store.csv')

# Handle missing values
train = handle_missing_values_median(train, ['CompetitionDistance', 'Promo2'])

# Extract date features
train = extract_date_features(train, 'Date')

# Merge datasets
train = merge_datasets(train, store, 'Store')
```

---

## 2. **EDA Script: `eda.py`**

### Purpose

The exploratory data analysis script provides a set of functions to visualize the dataset and gain insights into the underlying patterns, seasonality, and correlations in the data.

### Key Functions

- **`plot_rolling_statistics(df, window=12)`**  
  Plots the rolling mean and standard deviation for sales, helping to visualize trends and volatility.

- **`decompose_time_series(df, freq=365)`**  
  Decomposes the sales time series into trend, seasonality, and residuals.

- **`plot_acf_pacf(df, lags=30)`**  
  Displays the autocorrelation (ACF) and partial autocorrelation (PACF) plots for sales.

- **`detect_outliers_iqr(df)`**  
  Detects and visualizes outliers in the sales data using the interquartile range (IQR) method.

- **`count_state_holidays(df)`**  
  Counts and visualizes the distribution of state holidays in the dataset.

- **`count_school_holidays(df)`**  
  Counts and visualizes the distribution of school holidays in the dataset.

- **`count_store_types(df)`**  
  Displays the count of different store types in the dataset.

- **`count_assortment_types(df)`**  
  Visualizes the distribution of assortment types (variety of products) across stores.

- **`plot_top_correlations(df, target='Sales')`**  
  Plots the top 10 features most correlated with sales (numeric columns only).

### How to Use

```python
from eda import plot_rolling_statistics, decompose_time_series, count_state_holidays

# Plot rolling statistics
plot_rolling_statistics(train, window=12)

# Decompose time series
decompose_time_series(train, freq=365)

# Visualize the count of state holidays
count_state_holidays(train)
```

---

## 3. **Sales Analysis Script: `sales_analysis.py`**

### Purpose

The sales analysis script digs deeper into specific aspects of the data, including the impact of promotions, holidays, store types, assortment, and competitor activity on sales.

### Key Functions

- **`promo_distribution(df)`**  
  Analyzes and visualizes the distribution of promotions in the dataset.

- **`analyze_holiday_effects(df)`**  
  Visualizes the effect of holidays on sales behavior, both before and after holidays.

- **`seasonal_sales_behavior(df)`**  
  Plots the average monthly sales to observe seasonal patterns like Christmas or Easter.

- **`plot_day_of_week_sales(df)`**  
  Displays the average sales for each day of the week.

- **`sales_customers_correlation(df)`**  
  Analyzes the correlation between sales and the number of customers using Pearson correlation.

- **`promo_effect_on_sales(df)`**  
  Visualizes the effect of promotions on both sales and customer traffic.

- **`competitor_distance_sales(df)`**  
  Analyzes the effect of competitor distance on sales using scatter plots.

- **`promotion_effect_by_store_type(df)`**  
  Plots the effect of promotions across different store types.

### How to Use

```python
from sales_analysis import promo_distribution, sales_customers_correlation, competitor_distance_sales

# Analyze promo distribution
promo_distribution(train)

# Correlation between sales and customers
sales_customers_correlation(train)

# Effect of competitor distance on sales
competitor_distance_sales(train)
```

---

## Recommended Workflow

1. **Preprocessing**: Start by loading and cleaning the dataset using the `preprocessing.py` script. Handle missing values, extract date features, and merge necessary datasets.

2. **EDA**: Once the data is preprocessed, move on to `eda.py` to explore the dataset. Visualize patterns, correlations, and outliers to better understand the data.

3. **Sales Analysis**: Finally, use `sales_analysis.py` to investigate the key factors driving sales, such as promotions, holidays, and competition, and uncover deeper insights.

---

4. random_forest_model.py
   This script builds and trains a Random Forest model using Scikit-learn pipelines and hyperparameter tuning with RandomizedSearchCV.

Key Functions:
build_random_forest_model(): Sets up and tunes a Random Forest model using a randomized grid search for hyperparameter optimization.
train_and_evaluate_random_forest(X_train, y_train, X_test, y_test): Trains the model and evaluates it on the test set. 5. lstm_model.py
This script builds and trains an LSTM model for time-series forecasting using TensorFlow/Keras.

Key Functions:
create_supervised_data(series, window): Converts the time-series data into supervised data format for LSTM.
build_lstm_model(input_shape): Defines and compiles the LSTM model with appropriate layers and dropout regularization.
train_and_evaluate_lstm(X_train, y_train, X_test, y_test): Trains and evaluates the LSTM model.

6. model_serialization.py
   This script handles model serialization and deserialization, ensuring that trained models can be saved and loaded for later use.

Key Functions:
save_model(model, filepath): Serializes and saves the trained model to disk.
load_model(filepath): Loads a pre-trained model from disk. 7. post_prediction_analysis.py
This script contains utility functions for post-prediction analysis, including plotting feature importance for the Random Forest model and visualizing prediction results.

Key Functions:
plot_feature_importance(model, feature_names): Plots the feature importance of the Random Forest model.
estimate_confidence_interval(predictions, y_true): Estimates the 95% confidence interval of prediction errors.
visualize_predictions(predicted_sales, actual_sales): Visualizes predicted vs actual sales.
Running the Scripts
Preprocessing: Run preprocessing.py to preprocess the raw dataset.

bash
Copy code
python scripts/preprocessor.py
Random Forest Training: Train the Random Forest model and save the trained model.

bash
Copy code
python scripts/random_forest_model.py
LSTM Training: Train the LSTM model using the preprocessed data.

bash
Copy code
python scripts/lstm_model.py

bash
Copy code
python scripts/model_serialization.py
Post-Prediction Analysis: Analyze model predictions and visualize results.

bash
Copy code
python scripts/post_prediction_analysis.py

Dependencies
Python 3.8+
Libraries:
pandas
numpy
scikit-learn
tensorflow
matplotlib
statsmodels
