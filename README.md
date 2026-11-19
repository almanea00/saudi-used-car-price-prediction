<div align="center">

# Saudi Arabia Used Car Price Prediction

**Predict the fair asking price (SAR) of a used car in the Kingdom with XGBoost and Optuna.**

[![Open In Colab](https://img.shields.io/badge/Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/almanea00/saudi-used-car-price-prediction/blob/main/saudi_used_car_price_prediction.ipynb)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge)
![Optuna](https://img.shields.io/badge/Optuna-2A6DB2?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

![Test R2](https://img.shields.io/badge/Test_R%C2%B2-0.85-2ea44f?style=for-the-badge)
![Average error](https://img.shields.io/badge/Average_error-14.6k_SAR-orange?style=for-the-badge)
[![Stars](https://img.shields.io/github/stars/almanea00/saudi-used-car-price-prediction?style=for-the-badge&logo=github)](https://github.com/almanea00/saudi-used-car-price-prediction/stargazers)
[![Forks](https://img.shields.io/github/forks/almanea00/saudi-used-car-price-prediction?style=for-the-badge&logo=github)](https://github.com/almanea00/saudi-used-car-price-prediction/forks)
![Last commit](https://img.shields.io/github/last-commit/almanea00/saudi-used-car-price-prediction?style=for-the-badge)

[**The Problem**](#the-problem) · [**Business Value**](#why-this-matters-for-saudi-businesses) · [**Method**](#how-it-works) · [**Results**](#results) · [**Next Steps**](#next-steps) · [**Run It**](#run-it)

</div>

---

## The Problem

Used cars are a big consumer market in Saudi Arabia, and a large part of it runs through classifieds such as **Haraj**, dealer showrooms, and newer marketplaces such as **Syarah**. On these channels the price is whatever the seller types in, and nobody has a neutral benchmark.

```mermaid
flowchart LR
    A["Seller posts a car<br/>Haraj · Syarah · showroom"] --> B{"Is the price fair?"}
    B -- "No benchmark today" --> C["Overpaying, haggling,<br/>wasted visits, scams"]
    B -- "With this model" --> D["Instant fair-price estimate"]
    D --> E["Buyer knows what to pay"]
    D --> F["Seller prices realistically"]
    D --> G["Bank or insurer values the car"]
```

| Problem | Who it hurts |
|---|---|
| **No reference price.** Two near-identical cars can be listed thousands of riyals apart, and the buyer cannot tell which is fair. | Buyers overpay, honest sellers get undercut |
| **"Negotiable" / "call me" pricing.** Many Haraj-style ads hide the price, so the real value only shows up after calls and visits. | Buyers lose time, sellers get lowballed |
| **Information gap.** The seller knows the car's history, the buyer does not. | First-time buyers, exposed to bad deals and scams |
| **Gut-feeling dealer pricing.** Showrooms price on experience, so margins and stock turnover are unpredictable. | Dealers, fleet owners |
| **Manual valuation.** Banks, leasing companies and insurers often rely on hand appraisals. | Lenders, insurers, claims teams |

## Why This Matters for Saudi Businesses

A data-driven estimate gives every party a **fair-value benchmark in seconds**.

```mermaid
flowchart TD
    M(("Price<br/>model"))
    M --> A["Marketplaces<br/>Haraj · Syarah<br/>fair-price badge, price suggestion, deal flags"]
    M --> B["Dealers & showrooms<br/>consistent stock pricing,<br/>smarter trade-in offers"]
    M --> C["Fleet & rental companies<br/>value cars at disposal,<br/>time the sale"]
    M --> D["Banks & auto finance<br/>collateral and loan-to-value checks"]
    M --> E["Insurers<br/>faster, consistent claim valuations"]
    M --> F["Buyers & sellers<br/>a neutral benchmark<br/>before negotiating"]
```

It also fits the direction of **Vision 2030**: digital transformation, more transparent consumer markets and data-driven decisions in everyday commerce.

## How It Works

```mermaid
flowchart LR
    A["Kaggle<br/>dataset"] --> B["Explore"] --> C["Clean +<br/>Age feature"] --> D["Split<br/>60 / 20 / 20"]
    D --> E["Baselines<br/>Linear · Forest · XGBoost"] --> F["Optuna tuning<br/>on validation set"] --> G["Refit on train + val,<br/>one final test check"] --> H["Feature<br/>importance"]
```

- **Data:** [Saudi Arabia Used Cars Dataset](https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset) on Kaggle (cleaned English version). The notebook downloads it automatically, so no data is stored in this repo.
- **Target:** the asking `Price` in SAR. Listings marked *Negotiable* have no usable price and are left out.
- **Feature engineering:** `Year` becomes `Age`, so depreciation is easier for models to learn.
- **No leakage:** tuning uses only the validation set. The test set is touched once, at the end.
- **Regularized XGBoost:** the search includes `min_child_weight`, `gamma` and L1/L2 penalties, to help limit overfitting.
- **Final model:** the best settings are refit on train + validation (80% of the data) before the single test-set check. Tuning alone reaches about 0.83 test R², and the extra data brings it to about 0.85.

### What the data looks like

<p align="center">
  <img src="images/03_price_by_brand.png" alt="Price distribution by brand" width="85%">
</p>

*Brand matters a lot: Land Rover, Mercedes, BMW and Lexus sit far above everyday brands, and luxury models have the widest price spread.*

<p align="center">
  <img src="images/01_price_distribution.png" alt="Price distribution" width="85%">
</p>

*Most cars are cheap and a few are very expensive, so prices are heavily skewed. The log scale on the right makes the shape easier to read.*

## Results

### Each step improves on the last

<p align="center">
  <img src="images/08_model_comparison.png" alt="Model comparison on the test set" width="95%">
</p>

*Left: higher R² is better. Right: lower error is better. The last bar, the final tuned XGBoost (retrained on 80% of the data), is the best on both.*

### Tuning the model

<p align="center">
  <img src="images/05_train_vs_test.png" alt="Train versus test R2" width="70%">
</p>

*Out of the box, Random Forest and XGBoost score higher on the training data (blue) than on unseen cars (orange). That is why the final model is tuned on a separate validation set.*

<p align="center">
  <img src="images/06_baseline_vs_tuned.png" alt="Baseline versus tuned XGBoost" width="70%">
</p>

*Tuning lifts the score on unseen cars (green) above the baseline.*

### How close are the predictions?

<p align="center">
  <img src="images/07_actual_vs_predicted.png" alt="Actual versus predicted price" width="70%">
</p>

*Each dot is a car from the unseen test set. The closer to the red line, the better the prediction. The model is accurate for most cars and loses precision on the rare, very expensive ones.*

### What drives the price?

<p align="center">
  <img src="images/09_feature_importance.png" alt="Top 20 feature importances" width="75%">
</p>

*Luxury and exotic names (LX, G-Class, Mercedes, Bentley, Rolls-Royce, Land Rover, Porsche 911, Aston Martin) and a "Full" options level dominate. The Saudi market prices heavily on brand and trim, which matches how people actually shop.*

## Next Steps

This is a strong first version, and there is a clear path to make it better:

- **Real sale prices.** Train on final sale prices, not only asking prices, to get closer to true market value.
- **Include negotiable listings.** Model them separately so the full market is covered.
- **Add listing dates.** This would let the model follow price trends over time.
- **Add condition data.** Accident history, service records and body condition matter to real buyers.
- **Fairer accuracy for cheap cars.** Predicting `log(Price)` would give a more even error across price ranges.
- **Tighter generalization.** Cross-validation and stronger regularization to narrow the gap between training and test scores, plus CatBoost and LightGBM for comparison.
- **Explainability and deployment.** SHAP explanations, and a small web app or API a marketplace could call.

## Run It

### In Google Colab (easiest)

[![Open In Colab](https://img.shields.io/badge/Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/almanea00/saudi-used-car-price-prediction/blob/main/saudi_used_car_price_prediction.ipynb)

### Locally

```bash
git clone https://github.com/almanea00/saudi-used-car-price-prediction.git
cd saudi-used-car-price-prediction
pip install -r requirements.txt
jupyter notebook saudi_used_car_price_prediction.ipynb
```

`kagglehub` may ask you to log in to Kaggle the first time it downloads the dataset.

## Acknowledgements

Dataset by [turkibintalib](https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset) on Kaggle. Built as a course project.
