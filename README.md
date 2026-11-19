# Saudi Arabia Used Car Price Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/almanea00/saudi-used-car-price-prediction/blob/main/saudi_used_car_price_prediction.ipynb)

An end-to-end machine learning project that predicts the **asking price (in SAR)** of used cars in Saudi Arabia from listing details such as brand, model, trim level, age, mileage and engine size. It covers data cleaning, exploratory analysis, feature engineering, baseline models, hyperparameter tuning with **Optuna**, and an honest evaluation on a held-out test set.

> Built as a course project. The full workflow lives in the notebook: [`saudi_used_car_price_prediction.ipynb`](saudi_used_car_price_prediction.ipynb).

---

## Why this matters for Saudi businesses

### The problem: used car prices in the Kingdom are hard to judge

Used cars are a big consumer market in Saudi Arabia, and a large part of it runs through classifieds and marketplaces such as **Haraj**, dealer showrooms, and newer platforms like Syarah. In these channels, the price on a listing is whatever the seller decides to write. That creates real problems:

| Problem | Who it hurts |
|---|---|
| **No reference price.** Two near-identical cars (same brand, model, year and mileage) can be listed thousands of riyals apart, and the buyer has no neutral way to tell which one is fair. | Buyers overpay; honest sellers are undercut by unrealistic ads |
| **"Negotiable" / "call me" pricing.** Many Haraj-style ads hide the price or leave it open to haggling, so the real value is only found after calls and visits. | Buyers lose time; sellers get lowballed or ignored |
| **Information asymmetry.** The seller knows the car's history; the buyer does not. Without a benchmark, buyers cannot spot an ad that is too good to be true. | Buyers, especially first-time buyers, are exposed to bad deals and scams |
| **Inconsistent dealer pricing.** Showrooms price by gut feeling and recent experience, which makes margins and stock turnover unpredictable. | Dealers, fleet owners |
| **Manual valuation for finance and insurance.** Banks, leasing companies and insurers often rely on manual appraisals or rough depreciation tables to value a vehicle. | Lenders, insurers, claims teams |

### How this project helps

A data-driven price estimate gives every party a **fair-value benchmark in seconds**:

- **Classifieds and marketplaces (Haraj, Syarah and similar):** show a "fair price" badge or a "priced above / below market" flag on each listing, suggest a price when a seller posts an ad, and surface good deals. This builds trust and helps listings sell faster.
- **Used car dealers and showrooms:** price stock consistently, decide what to pay at trade-in or auction, and spot under-priced cars to buy.
- **Fleet and rental companies:** value cars at the point of disposal (end of lease or rental cycle) and time the sale.
- **Banks and auto-finance providers:** support loan-to-value checks and collateral valuation with a consistent number instead of case-by-case guesses.
- **Insurers:** support quicker, more consistent total-loss and claim valuations.
- **Buyers and sellers:** sanity-check any deal before spending a weekend driving between meetings.

It also fits the direction of **Vision 2030**: digital transformation, more transparent consumer markets, and data-driven decision-making in everyday commerce.

### What the model learned

The top drivers of price in the tuned model are **brand and model** (Mercedes, Rolls-Royce, Land Rover, BMW, Lexus, Aston Martin, and models such as LX, G, Levante, Range Rover, Land Cruiser, 911) and **options level** (e.g. "Full"). In other words, the Saudi market prices heavily on luxury, exotic and full-spec vehicles, which matches how the market behaves in practice.

---

## Dataset

- **Source:** [Saudi Arabia Used Cars Dataset](https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset) on Kaggle (`UsedCarsSA_Clean_EN.csv`, the cleaned English version).
- **Target:** `Price` (SAR).
- **Features:** brand (`Make`), model (`Type`), `Options` level, `Engine_Size`, `Mileage`, manufacturing `Year` (converted to `Age`), plus other categorical listing attributes.
- **After cleaning:** about 5,500 listings with a fixed price (rows marked *Negotiable* and rows with missing values are removed).

The data is downloaded automatically in the notebook with `kagglehub`, so nothing needs to be committed to this repo.

## Method

1. **Exploratory data analysis**: price distribution (linear and log scale), listings and average price by year, price spread across the top 15 brands, correlation matrix.
2. **Cleaning**: remove *Negotiable* listings, missing values, negative prices or mileage, and zero-size engines.
3. **Feature engineering**: replace `Year` with `Age = current year - Year` so depreciation is easier for models to represent.
4. **Preprocessing pipeline** (scikit-learn): median imputation plus standard scaling for numeric columns; most-frequent imputation plus one-hot encoding for categorical columns.
5. **Three-way split**: 60% train, 20% validation, 20% test. Tuning uses only the validation set, so the test set stays untouched until the end.
6. **Baseline models** with increasing complexity: Linear Regression, Random Forest, XGBoost.
7. **Hyperparameter tuning**: 100 Optuna trials on XGBoost, searching tree depth, learning rate, subsampling and strong L1/L2 regularization (`reg_alpha`, `reg_lambda`) to limit overfitting.
8. **Evaluation**: MAE, RMSE and R² on the test set, an actual-vs-predicted plot, and feature importances.

## Results

Baselines (default settings, R²):

| Model | Train R² | Test R² |
|---|---|---|
| Linear Regression | 0.7487 | 0.6492 |
| Random Forest | 0.9751 | 0.7587 |
| XGBoost | 0.9757 | 0.7852 |

The Random Forest and XGBoost baselines fit the training data almost perfectly (R² about 0.975) but drop sharply on unseen data, which is classic overfitting. That is why the tuning step adds regularization and picks parameters on a separate validation set.

After tuning with Optuna:

| Metric | Value |
|---|---|
| Best validation R² | 0.832 |
| **Test R²** | **0.820** |
| **Test MAE** | **about 15,434 SAR** |

Tuning improved test R² from 0.785 to 0.820 and cut the average error from about 17,072 SAR to 15,434 SAR compared with the XGBoost baseline.

Best parameters found: `n_estimators=291`, `learning_rate=0.082`, `max_depth=8`, `subsample=0.54`, `colsample_bytree=0.87`, `reg_alpha=15.3`, `reg_lambda=0.30`. The Optuna sampler is seeded, so these numbers reproduce when you re-run the notebook.

## Limitations (and what I would do next)

- **Asking prices, not sale prices.** The model predicts what sellers ask, not what the car finally sold for. Real transaction data would be more accurate.
- **Negotiable listings are excluded**, so the model only learns from fixed-price ads. Those ads may not represent the whole market.
- **A snapshot in time.** Listing dates are not in the data, so the model cannot track price trends or market shocks, and it will need retraining as prices move.
- **MAE of about 15k SAR** is useful as a benchmark, but too coarse for cheap cars, where it is a large share of the price. Predicting `log(Price)` or reporting percentage error would be fairer across price ranges.
- **No condition data.** Accident history, service records and paint/body condition are missing, and they matter to real buyers.
- **Next steps:** try log-transforming the target, add cross-validation, test CatBoost/LightGBM, add SHAP explanations, and wrap the model in a small web app or API that a marketplace could call.

## How to run

The easiest way is **Google Colab**: open the notebook and choose *Runtime > Run all*. The dataset is fetched from Kaggle automatically.

To run locally:

```bash
git clone https://github.com/almanea00/saudi-used-car-price-prediction.git
cd saudi-used-car-price-prediction
pip install -r requirements.txt
jupyter notebook saudi_used_car_price_prediction.ipynb
```

`kagglehub` may ask you to log in to Kaggle the first time you download the dataset.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, Optuna, matplotlib, seaborn, kagglehub.

## Acknowledgements

Dataset by [turkibintalib](https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset) on Kaggle.
