# Apparel & Textiles Trend Analysis and Forecasting

## Project Overview

This project focuses on conducting trend analysis and demand forecasting for the apparel and textiles retail industry using publicly available historical transaction data.

The project uses the **H&M Personalized Fashion Recommendations** dataset from Kaggle as the primary data source. Historical transaction data is cleaned, transformed and aggregated into a monthly time series to identify sales trends, seasonal patterns and changes in customer purchasing activity.

Forecasting techniques such as Moving Average, Linear Regression and Holt-Winters Exponential Smoothing are considered to estimate future demand. The forecasting models are evaluated using chronological validation and error metrics such as MAE and RMSE.

The purpose of this project is to combine quantitative analysis with business interpretation and provide useful insights for inventory planning, merchandising, pricing and supply-chain decision-making.

---

## Objectives

The main objectives of this project are:

- Analyze historical apparel retail transaction data
- Identify monthly sales trends and patterns
- Analyze changes in transaction volume and customer activity
- Calculate moving averages to identify underlying trends
- Apply suitable forecasting techniques
- Compare forecasting models using validation metrics
- Generate a 12-month future forecast
- Identify potential market impacts
- Highlight business risks and opportunities
- Provide data-driven strategic recommendations

---

## Dataset

The dataset used in this project is the **H&M Personalized Fashion Recommendations** dataset.

### Kaggle Dataset

**H&M Personalized Fashion Recommendations:**  
https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

The dataset contains historical retail transaction information along with article and customer information related to apparel purchases.

The main files used in the analysis include:

- `transactions_train.csv`
- `articles.csv`
- `customers.csv`

The original Kaggle dataset files are not stored directly in this repository because of their large size and Kaggle competition conditions.

---

## Historical Data Analysis

The historical transaction data is processed to identify important patterns in the apparel retail sector.

The analysis includes:

- Monthly sales analysis
- Transaction volume analysis
- Active customer analysis
- Average transaction value
- Moving-average trends
- Product/category performance
- Seasonal pattern identification
- Historical growth and decline analysis

The transaction-level data is aggregated into a monthly time series so that it can be used for trend analysis and forecasting.

---

## Methodology

The project follows the following workflow:

```text
Kaggle Dataset
      ↓
Data Collection
      ↓
Data Cleaning
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Monthly Aggregation
      ↓
Historical Trend Analysis
      ↓
Moving Average Analysis
      ↓
Forecasting Models
      ↓
Chronological Validation
      ↓
Model Comparison
      ↓
12-Month Forecast
      ↓
Business Recommendations
