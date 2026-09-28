# Retail Prices Analysis & Machine Learning Prediction

A complete data science pipeline analyzing retail pricing datasets using Exploratory Data Analysis (EDA), statistical hypothesis testing, and machine learning regression models.

---

## Project Overview

This project explores a dataset of over 22,000 retail pricing records to uncover commodity-specific price distributions, temporal inflationary trends, and feature correlations. Using these insights, we built and evaluated predictive regression models to forecast average retail prices.

---

## Core Methodology & Features

1. **Data Cleaning & Preprocessing:** 
   - Handled missing values and standardized numerical features using `StandardScaler` to ensure zero mean and unit variance before model training.
2. **Feature Engineering:**
   - **Price Spread:** Created derived metrics capturing price ranges across markets.
   - **Temporal Encoding:** Factorized and chronologically sorted week strings (e.g., `2021-05-W1`) into sequential identifiers (`w1`, `w2`, ...) and numeric index features (`Week_Num`) for time-series analysis.
3. **Statistical Testing:**
   - **Welch’s t-Test:** Evaluated price differences across commodities ($p < 0.05$).
   - **Pearson Correlation Test:** Assessed long-term price drift over time.
4. **Machine Learning Models:**
   - **Linear Regression:** Used as a baseline parametric model.
   - **Random Forest Regressor:** An ensemble tree-based model ($n\_estimators=50$) built to capture non-linear feature interactions and item-specific thresholds.

---

## Summary of Results

- **Data Distribution:** Retail prices exhibit a right-skewed distribution; everyday commodities cluster below Rs. 500, while specialized or luxury items exceed Rs. 3,000.
- **Hypothesis Testing:**
   - *Test 1 (Items):* Confirmed definitive statistical evidence that different items command significantly distinct price points ($p = 6.67 \times 10^{-66} < 0.05$).
   - *Test 2 (Time Trend):* Confirmed a statistically significant upward drift in retail prices over time ($r = 0.201$, $p = 6.01 \times 10^{-203} < 0.05$).
- **Model Performance:**
   - **Linear Regression:** Achieved strong predictive accuracy with an RMSE of $61.50$ and an $R^2$ score of $0.9939$.
   - **Random Forest Regressor:** Outperformed the baseline with a lower RMSE of $49.18$ and a higher $R^2$ score of $0.9961$, successfully handling complex non-linear patterns.

---

## Project Structure

```text
├── data/
│   └── retail_prices.csv      # Raw dataset
├── notebooks/
│   └── analysis.ipynb         # EDA, statistical tests, and modeling script
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

---

## Getting Started

### Prerequisites
Make sure you have Python installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Running the Analysis
1. Clone the repository and navigate to the project directory.
2. Open the Jupyter Notebook or Python script containing the pipeline:
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```
3. Run the cells sequentially to reproduce data cleaning, statistical tests, and machine learning evaluations.