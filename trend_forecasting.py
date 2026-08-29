# ============================================================
# WEEK 5 - APPAREL & TEXTILES TREND ANALYSIS & FORECASTING
# ============================================================

# Install required libraries before running:
# pip install kagglehub pandas numpy matplotlib scikit-learn statsmodels

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ============================================================
# 1. DOWNLOAD DATASET FROM KAGGLE
# ============================================================

print("Downloading H&M dataset from Kaggle...")

path = kagglehub.competition_download(
    "h-and-m-personalized-fashion-recommendations"
)

print("Dataset downloaded successfully.")
print("Dataset path:", path)


# ============================================================
# 2. LOAD TRANSACTION DATA
# ============================================================

transactions_file = Path(path) / "transactions_train.csv"

tx = pd.read_csv(transactions_file)

print("\nOriginal Dataset Shape:")
print(tx.shape)

print("\nFirst 5 Records:")
print(tx.head())


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\nStarting data cleaning...")

# Convert date column
tx["t_dat"] = pd.to_datetime(
    tx["t_dat"],
    errors="coerce"
)

# Convert price to numeric
tx["price"] = pd.to_numeric(
    tx["price"],
    errors="coerce"
)

# Remove records with missing essential values
tx = tx.dropna(
    subset=[
        "t_dat",
        "customer_id",
        "article_id",
        "price"
    ]
)

# Remove duplicate records
tx = tx.drop_duplicates()

# Keep only valid positive prices
tx = tx[
    tx["price"] > 0
]

print("Cleaned Dataset Shape:")
print(tx.shape)


# ============================================================
# 4. CREATE MONTHLY SALES TIME SERIES
# ============================================================

print("\nCreating monthly sales time series...")

monthly = (
    tx.set_index("t_dat")
      .resample("MS")["price"]
      .sum()
      .rename("sales")
      .reset_index()
)

print("\nMonthly Sales:")
print(monthly.head())


# ============================================================
# 5. MOVING AVERAGES
# ============================================================

# 3-month moving average
monthly["ma_3"] = (
    monthly["sales"]
    .rolling(window=3)
    .mean()
)

# 6-month moving average
monthly["ma_6"] = (
    monthly["sales"]
    .rolling(window=6)
    .mean()
)


# ============================================================
# 6. HISTORICAL TREND VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly["t_dat"],
    monthly["sales"],
    label="Monthly Sales"
)

plt.plot(
    monthly["t_dat"],
    monthly["ma_3"],
    label="3-Month Moving Average"
)

plt.plot(
    monthly["t_dat"],
    monthly["ma_6"],
    label="6-Month Moving Average"
)

plt.title(
    "Historical Monthly Sales and Moving Averages"
)

plt.xlabel("Month")
plt.ylabel("Sales Value")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "historical_sales_trend.png",
    dpi=150
)

plt.show()


# ============================================================
# 7. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

# The latest 6 months are kept for testing.
# Earlier months are used for training.

train = monthly.iloc[:-6].copy()
test = monthly.iloc[-6:].copy()

print("\nTraining Period:")
print(train["t_dat"].min(), "to", train["t_dat"].max())

print("\nTesting Period:")
print(test["t_dat"].min(), "to", test["t_dat"].max())


# ============================================================
# 8. LINEAR REGRESSION FORECAST
# ============================================================

print("\nRunning Linear Regression...")

# Create time index
train["t"] = np.arange(
    len(train)
)

test["t"] = np.arange(
    len(train),
    len(monthly)
)

# Create model
linear_model = LinearRegression()

# Train model
linear_model.fit(
    train[["t"]],
    train["sales"]
)

# Predict test period
test["linear_forecast"] = (
    linear_model.predict(
        test[["t"]]
    )
)


# ============================================================
# 9. LINEAR REGRESSION EVALUATION
# ============================================================

linear_mae = mean_absolute_error(
    test["sales"],
    test["linear_forecast"]
)

linear_rmse = (
    mean_squared_error(
        test["sales"],
        test["linear_forecast"]
    ) ** 0.5
)

print("\nLinear Regression Results")
print("-------------------------")
print("MAE :", linear_mae)
print("RMSE:", linear_rmse)


# ============================================================
# 10. HOLT-WINTERS FORECAST
# ============================================================

print("\nRunning Holt-Winters model...")

# Use additive trend and yearly seasonality.
# This requires sufficient monthly observations.

if len(train) >= 24:

    holt_model = ExponentialSmoothing(
        train["sales"],
        trend="add",
        seasonal="add",
        seasonal_periods=12
    ).fit()

    test["holt_winters_forecast"] = (
        holt_model.forecast(
            len(test)
        )
    )

    holt_mae = mean_absolute_error(
        test["sales"],
        test["holt_winters_forecast"]
    )

    holt_rmse = (
        mean_squared_error(
            test["sales"],
            test["holt_winters_forecast"]
        ) ** 0.5
    )

    print("\nHolt-Winters Results")
    print("--------------------")
    print("MAE :", holt_mae)
    print("RMSE:", holt_rmse)

else:

    holt_model = None
    holt_mae = None
    holt_rmse = None

    print(
        "Not enough observations for "
        "12-month seasonal Holt-Winters."
    )


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Holt-Winters"
    ],
    "MAE": [
        linear_mae,
        holt_mae
    ],
    "RMSE": [
        linear_rmse,
        holt_rmse
    ]
})

results = results.dropna()

print("\nModel Comparison")
print("================")
print(results)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

# Models with lower MAE and RMSE are preferred.
# MAE is used as the primary ranking metric here.

best_model_name = (
    results
    .sort_values(
        by=["MAE", "RMSE"]
    )
    .iloc[0]["Model"]
)

print("\nSelected Model:")
print(best_model_name)


# ============================================================
# 13. TRAIN FINAL MODEL ON COMPLETE DATA
# ============================================================

print("\nTraining final forecasting model...")

if best_model_name == "Holt-Winters":

    final_model = ExponentialSmoothing(
        monthly["sales"],
        trend="add",
        seasonal="add",
        seasonal_periods=12
    ).fit()

    future_forecast = (
        final_model.forecast(12)
    )

else:

    # Linear Regression on complete dataset

    monthly["t"] = np.arange(
        len(monthly)
    )

    final_model = LinearRegression()

    final_model.fit(
        monthly[["t"]],
        monthly["sales"]
    )

    future_t = np.arange(
        len(monthly),
        len(monthly) + 12
    )

    future_forecast = (
        final_model.predict(
            future_t.reshape(-1, 1)
        )
    )


# ============================================================
# 14. CREATE FUTURE FORECAST TABLE
# ============================================================

future_dates = pd.date_range(
    start=(
        monthly["t_dat"].max()
        + pd.offsets.MonthBegin(1)
    ),
    periods=12,
    freq="MS"
)

forecast_df = pd.DataFrame({
    "date": future_dates,
    "forecast_sales": np.asarray(
        future_forecast
    )
})

print("\n12-Month Forecast")
print("=================")
print(forecast_df)


# ============================================================
# 15. FORECAST VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly["t_dat"],
    monthly["sales"],
    label="Historical Sales"
)

plt.plot(
    forecast_df["date"],
    forecast_df["forecast_sales"],
    label="12-Month Forecast"
)

plt.title(
    "Apparel Sales Trend and 12-Month Forecast"
)

plt.xlabel("Month")
plt.ylabel("Sales Value")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "12_month_sales_forecast.png",
    dpi=150
)

plt.show()


# ============================================================
# 16. SAVE OUTPUT FILES
# ============================================================

output_dir = Path("output")

output_dir.mkdir(
    exist_ok=True
)

monthly.to_csv(
    output_dir / "monthly_sales_trend.csv",
    index=False
)

results.to_csv(
    output_dir / "forecast_model_comparison.csv",
    index=False
)

forecast_df.to_csv(
    output_dir / "12_month_forecast.csv",
    index=False
)

print("\nOutput files saved successfully.")


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n======================================")
print("TREND ANALYSIS & FORECASTING COMPLETE")
print("======================================")

print(
    f"Historical Months Analyzed: {len(monthly)}"
)

print(
    f"Selected Forecasting Model: {best_model_name}"
)

print(
    f"Validation MAE: "
    f"{results.loc[results['Model'] == best_model_name, 'MAE'].iloc[0]:,.2f}"
)

print(
    f"Validation RMSE: "
    f"{results.loc[results['Model'] == best_model_name, 'RMSE'].iloc[0]:,.2f}"
)

print("\nForecast generated for the next 12 months.")
