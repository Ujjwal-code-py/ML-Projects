# Mutual Fund Predictive Analytics

##  App Link -> [https://mutualfundsreturnprediction.streamlit.app/)](https://mutualfundsreturnprediction.streamlit.app/)

## Abstract

This project predicts expected mutual fund returns for 1-year, 3-year, and 5-year investment horizons using historical mutual fund data, fund characteristics, and machine learning regression models. The project includes a data scraping pipeline, data cleaning script, model experimentation notebooks, a saved deployment pipeline, and a Streamlit application for interactive prediction and analytics.

The core idea is that a fund's return behavior can be partially explained by measurable fund attributes such as Assets Under Management (AUM), Net Asset Value (NAV), fund rating, equity allocation, debt allocation, risk profile, and fund category. The final modelling workflow uses feature engineering and ensemble regression to estimate future return percentages.

Important note: this is an educational predictive analytics project. It is not financial advice. Mutual fund returns depend on market, macroeconomic, manager, liquidity, and regulatory factors that are not fully captured in this dataset.

## Project Objectives

- Collect mutual fund data from Groww pages and Groww portfolio statistics endpoints.
- Clean raw financial strings such as rupee-formatted AUM and NAV values into numeric columns.
- Engineer finance-relevant features such as risk score, AUM/NAV ratio, allocation concentration, and fund type indicators.
- Compare regression algorithms for return prediction.
- Save a trained model pipeline for deployment.
- Build a Streamlit app where a user can enter fund characteristics and receive predicted returns.
- Provide an analytics dashboard for data exploration, risk-return analysis, and top-performing funds.

## Repository Structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit web app for prediction, analytics dashboard, and project explanation. |
| `scraper and extraction.py` | Scrapes mutual fund data from Groww listing/detail pages and portfolio API endpoints. |
| `Tranformation.py` | Cleans `raw_data.xlsx` by dropping unused columns and converting AUM/NAV fields. |
| `raw_data.xlsx` | Raw scraped dataset. |
| `data/cleaned_data.xlsx` | Cleaned dataset used for analysis/modeling. |
| `p1.ipynb` | Main end-to-end modelling notebook with feature engineering, model comparison, final model training, diagnostics, and pipeline export. |
| `Model testing for 1 year analysis.ipynb` | Earlier/individual experiments for 1-year return prediction. |
| `Model testing for 3 year analysis.ipynb` | Earlier/individual experiments for 3-year return prediction. |
| `Model testing for 5 year analysis.ipynb` | Earlier/individual experiments for 5-year return prediction. |
| `mutual_fund_returns_pipeline.pkl` | Saved deployment artifact containing trained models, scalers, feature columns, metrics, and metadata. |
| `requirements.txt` | Python dependencies needed to run the project. |
| `runtime.txt` | Python runtime target for deployment. |

## Research Paper Style Summary

### Title

Predictive Analytics for Mutual Fund Return Forecasting Using Machine Learning Regression Models

### Problem Statement

Retail investors often compare mutual funds using past returns, ratings, AUM, risk category, and portfolio allocation. However, these variables are usually interpreted manually. This project converts fund-level attributes into a machine learning problem: given a fund's measurable characteristics, predict its expected 1-year, 3-year, and 5-year return percentages.

### Research Question

Can fund characteristics such as AUM, NAV, rating, risk category, equity allocation, debt allocation, and fund category be used to predict mutual fund returns across different investment horizons?

### Hypothesis

Funds with higher equity allocation, stronger ratings, higher risk profiles, and favorable historical structural characteristics should show distinguishable return patterns. Machine learning models, especially ensemble tree models, should capture non-linear relationships better than simple linear regression.

### Dataset

The project data was collected from Groww mutual fund pages. The dataset contains around 1,065 records in the main notebook analysis, with target availability varying by investment horizon:

| Target | Available Samples | Availability |
| --- | ---: | ---: |
| 1-year returns | 966 | 90.7% |
| 3-year returns | 817 | 76.7% |
| 5-year returns | 696 | 65.4% |

The reduced training feature set uses fund attributes rather than fund names or URLs, so predictions are based on characteristics and not direct fund identity.

### Input Features

| Feature | Meaning |
| --- | --- |
| `aum_funds_individual_lst` | Assets Under Management, cleaned from rupee/crore text into numeric form. |
| `nav_funds_individual_lst` | Net Asset Value per unit. |
| `rating_of_funds_individual_lst` | Fund rating, typically from 1 to 5. |
| `minimum_funds_individual_lst` | Minimum investment amount. |
| `debt_per` | Percentage allocation to debt instruments. |
| `equity_per` | Percentage allocation to equity instruments. |
| `risk_score` | Numeric encoding of categorical risk profile. |
| `fund_type_encoded` | Numeric encoding of fund category. |
| `aum_nav_ratio` | AUM divided by NAV plus 1. |
| `equity_concentration` | Equity percentage normalized to 0 to 1. |
| `debt_concentration` | Debt percentage normalized to 0 to 1. |
| `is_equity_fund` | Binary flag for equity funds. |
| `is_hybrid_fund` | Binary flag for hybrid funds. |

### Target Variables

- `one_year_returns`: return percentage over 1 year.
- `three_year_returns`: return percentage over 3 years.
- `five_year_returns`: return percentage over 5 years.

### Methodology

1. Raw data was scraped from Groww mutual fund listing pages and individual fund pages.
2. Additional portfolio statistics were collected using Groww scheme code endpoints.
3. Unnecessary columns such as URLs, names, PE/PB, average maturity, and yield-to-maturity were removed for the cleaned modelling dataset.
4. AUM and NAV strings were converted into numeric values.
5. Risk profile and fund category were encoded into numeric model inputs.
6. Derived features were created for allocation concentration and AUM/NAV ratio.
7. Separate datasets were prepared for each target because return availability differs by horizon.
8. Missing feature values were handled using `KNNImputer` during notebook training.
9. Multiple regression algorithms were compared.
10. Final Gradient Boosting models were trained separately for 1-year, 3-year, and 5-year targets.
11. The final deployment pipeline was saved as `mutual_fund_returns_pipeline.pkl`.

### Algorithms Tested

| Algorithm | Why It Was Tested |
| --- | --- |
| Linear Regression | Simple baseline model for linear relationships. |
| Ridge Regression | Regularized linear model to reduce overfitting and multicollinearity. |
| Random Forest Regressor | Ensemble bagging model that captures non-linear interactions. |
| Gradient Boosting Regressor | Ensemble boosting model designed to improve prediction error sequentially. |

### Final Model

The final deployment pipeline uses `GradientBoostingRegressor` models for the three horizons. In the notebook, the final model configuration was:

```python
GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
```

Separate models and scalers are stored for:

- `1_year`
- `3_year`
- `5_year`

### Reported Model Performance

The main notebook reports the following final in-sample diagnostics:

| Horizon | R2 Score | RMSE | MAE | Samples | 95% Error Bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-Year | 0.853 | 2.97% | 1.999% | 966 | +/- 6.438% |
| 3-Year | 0.949 | 1.64% | 1.115% | 817 | +/- 3.766% |
| 5-Year | 0.959 | 0.78% | 0.554% | 696 | +/- 1.688% |

Interpretation: lower RMSE and MAE mean the predicted return is closer to the actual observed return. R2 measures how much variance in the target is explained by the model.

Important evaluation caution: the notebook also includes time-series cross-validation comparisons with much weaker and sometimes negative R2 values. This means the very strong final scores should be presented carefully as final fitted-model diagnostics, not guaranteed real-world forecasting accuracy. For a production-grade financial model, the project should add a strict out-of-sample holdout period and test on funds/dates not used during training.

### Key Observations

- Equity and debt allocation are highly important for 3-year and 5-year return behavior.
- Fund rating is especially important for 5-year prediction in the notebook feature importance output.
- NAV and AUM/NAV ratio contribute to prediction, but their effect should be interpreted carefully because NAV scale differs across funds.
- Risk score and equity concentration show strong correlation with medium and long-term returns.
- Long-horizon targets have fewer samples, so the model may be more sensitive to data availability.

## Application Workflow

The Streamlit app has three main pages:

### Returns Prediction

The user enters:

- AUM in crores.
- NAV.
- Fund rating.
- Equity allocation percentage.
- Risk profile.
- Fund category.
- Prediction period: 1 year, 3 years, or 5 years.

The app calculates debt allocation as `100 - equity allocation`, builds the engineered feature row, loads the saved pipeline, applies the correct scaler and model, then displays the predicted return and an interpretation.

If the model cannot be loaded or prediction fails, the app uses a heuristic fallback calculation so the UI remains usable.

### Analytics

The dashboard loads data from a Google Sheet first. If that fails, it falls back to `data/cleaned_data.xlsx`. It includes:

- Average returns by horizon.
- Returns distribution boxplot.
- Average returns by fund category.
- Equity allocation vs 3-year returns scatter plot.
- Risk category performance table.
- Risk category distribution charts.
- Top 10 funds by 3-year and 5-year returns.
- Data explorer with filters and CSV download.

### About Tool

This page explains how the app works, its intended use, investment assumptions, and disclaimer.

## How to Run Locally

### 1. Create and activate a virtual environment

Use Python 3.11. The project's `runtime.txt` currently specifies `python-3.11.4`, and the saved model pickle is not compatible with newer scikit-learn versions commonly installed under Python 3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The app should open in the browser. If it does not open automatically, use the local URL printed by Streamlit, usually `http://localhost:8501`.

## How to Rebuild the Data

Run the scraper:

```bash
python "scraper and extraction.py"
```

This creates or overwrites:

```text
raw_data.xlsx
```

Then run the transformation script:

```bash
python Tranformation.py
```

This script reads `raw_data.xlsx`, removes unused columns, converts selected string columns, and writes a cleaned Excel file.

Note: web scraping depends on Groww page structure and network availability. If Groww changes class names, endpoints, or response formats, the scraper may need updates.

## How to Rebuild the Model

Open and run:

```text
p1.ipynb
```

The notebook performs:

- Data loading.
- Data exploration.
- Feature engineering.
- Correlation analysis.
- Model comparison.
- Final model training.
- Diagnostics.
- Pipeline export.

The final export is:

```text
mutual_fund_returns_pipeline.pkl
```

## Presentation Talking Points

- This project solves a regression problem: predicting future return percentages from fund characteristics.
- The data pipeline starts with scraping, then cleaning, then feature engineering, then modelling, then app deployment.
- Multiple algorithms were tested to compare simple linear models against ensemble models.
- Gradient Boosting was selected because it can model non-linear relationships and performed strongly in the final notebook diagnostics.
- The deployed Streamlit app makes the model usable by non-technical users.
- The analytics dashboard supports visual explanation of fund categories, risk levels, and return distributions.
- The strongest limitation is that financial markets are non-stationary, so model accuracy can degrade when market conditions change.
- Strong reported model scores should be presented with the caveat that future validation should use strict out-of-sample testing.

## Known Limitations and Future Improvements

- The scraper is fragile because it depends on external website HTML classes.
- The model uses fund-level static attributes and does not include macroeconomic indicators, market index returns, interest rates, inflation, expense ratio trends, or fund manager changes.
- The final notebook metrics include fitted-model diagnostics; production validation should include a separate holdout set.
- The app currently asks for a small subset of user-friendly inputs and fills some engineered features automatically.
- The saved pipeline does not include the original imputers, so the app uses deterministic defaults for features not entered by the user.
- The saved pickle depends on the compatible Python/scikit-learn versions pinned in `requirements.txt`; retraining and exporting a newer pipeline is recommended before upgrading dependencies.
- Future versions should store preprocessing, imputation, encoding, scaling, and models in one sklearn `Pipeline` object per horizon.
- Future versions should include model explainability with SHAP or permutation importance in the app.
- Future versions should add unit tests for feature generation and prediction loading.

## Issues Fixed During Review

- Added `requirements.txt` so dependencies are installable reproducibly.
- Pinned `scikit-learn==1.3.0` because the saved model artifact was exported with that version and fails to load correctly on newer releases.
- Updated `app.py` to load the saved `joblib` model pipeline correctly.
- Updated prediction logic to select the correct model/scaler for 1-year, 3-year, or 5-year predictions.
- Added feature engineering in the app so user inputs match the saved model's expected feature columns.
- Added a local analytics fallback to `data/cleaned_data.xlsx` when Google Sheets loading fails.

## Disclaimer

This project is for academic and demonstration purposes only. Predicted returns are statistical estimates based on historical data and selected fund attributes. They should not be treated as guaranteed future returns or investment recommendations. Always consult a qualified financial advisor before making investment decisions.
