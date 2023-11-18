---
title:  "Get to TOP 10% in House Price Prediction"
metadate: "hide"
date : 2023-11-17 18:00:00 +0900
categories: [ ML/DL ]
image: "/assets/images/ImageFolder.png" 
---  
## Get to TOP 10% in House Price Prediction
LogisticRegression의 대표적인 Kaggle 대회인 'House Price Prediction'에 참가해, Leaderboard Top 10%에 들기 위해 다양한 테크닉들을 사용해보고 기록해보고자 한다.

> *House Prices - Advanced Regression Techniques*  
> https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/

### Day 1 @ 2023.10.26
**EDA**
- draw `Correlation matrix` (heatmap style)
- draw 'SalePrice' correlation matrix(`zoomed heatmap style`)
- draw `Scatter plots` between 'SalePrice' and correlated variables
- dealing with `Missing data` with `drop`
- dealing with `Outliers` with `Univariate analysis` & `Bivariate analysis`
- make data `Normally distributed` with `transformations`
- convert `categorical variable into dummy`

[Kaggle Code](https://www.kaggle.com/code/outoftime/house-price-prediction-eda)

<hr>

### Day 2 @ 2023.10.27
**use XGBoost**
- XGBRegressor
- model tunning with early stopping

> **Score : 0.14547**  
> **Rank : 2128**

[Kaggle Code](https://www.kaggle.com/code/outoftime/xgboost-with-eda)

<hr>
