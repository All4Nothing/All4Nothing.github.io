---
title: "Bike Sharing Demand"
excerpt: "Competition Review"

categories:
  - Machine Learning
tags:
  - [Competition]
use_math: true

permalink: /machine-learning/bike-sharing-competition/

toc: true
toc_sticky: true

date: 2024-04-04
last_modified_at: 2024-04-04
---  
[Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand)

[Bike Sharing Demand](https://www.kaggle.com/code/outoftime/bike-sharing-demand)

**Day 1**

- Ridge : 1.04966
- Lasso : 1.04126
- Random Forest : 0.43310
- **Gradient Boosting : 0.42744**
- 535등 (Top 15%)

> ***Score : 0.42744
Rank : 535 (Top 16.49%)***
> 

**Day 2**

Model을 train 시킬 때, 평소처럼 sklearn의 train_test_split을 사용했는데, 생각해보니 time series 데이터를 이런식으로 split 하면 문제가 있을 것 같다는 생각을 하고 코드를 수정하기로 한다.

방법1 - 시간순으로 정렬 후 앞에서부터 80%만큼을 train 데이터셋으로, 나머지 20%를 validation 데이터셋으로 설정한다.

- Ridge : 1.04723 (+0.23%)
- Lasso : 1.05059 (-0.89%)
- **Random Forest : 0.41850 (+3.38%)**
- Gradient Boosting : 0.43600 (-2%)
- 535등 (Top 15%)

> ***Score :* 0.41850 
*Rank : 428 (Top 13.19%)***
> 

방법2 - sklearn의 TimeSeriesSplit과 GridSearchCV를 이용하여 하나의 class로 정의한다.

```python
class TimeSeries_GridCV_Model:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        
    def fit_and_evaluate(self, model, params, scorer, X, y):
        for train_index, test_index in self.tscv.split(dataTrain):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            grid_search = GridSearchCV(model, params, scoring = scorer, cv=5, error_score='raise')
            grid_search.fit(X_train.values, np.log1p(y_train))
            
            y_pred = grid_search.predict(X_test.values)
            
            error = rmsle(np.exp(np.log1p(y_test)), np.exp(y_pred), False)
            print(grid_search.best_params_)
            print("RMSLE Value:", error)
            
        return grid_search     
    
    def plot_parameters(self, grid_search):        
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 5)
        df = pd.DataFrame(grid_search_result.cv_results_)
        df["alpha"] = df["param_alpha"]
        df["rmsle"] = -df["mean_test_score"]
        sns.pointplot(data=df, x='alpha', y='rmsle', ax=ax)
        
    def predict(self, grid_search, dataTest):
        predsTest = grid_search.predict(X = dataTest)
        return predsTest
```

- Ridge : 1.04990
- Lasso : 1.04370

- windspeed와 humidity 0값은 실제 0값이 아닌 null값이 0으로 표시된 거였다 → `RandomForestClassifier` 를 이용해 null값 채움

---

**유용했던 Techinque**

**`pd.DatetimeIndex`**

- dataset에 `datetime` feature에서 `hour`, `day`, `month`, `year` 정보를 추출한다.

**`Detect special missing value like 0`**

- null 값이 없어보였지만, `windspeed` , `humidity` feature에서 발견되는 0 값은 실제 세계에서는 존재할 수 없는 데이터이기에 missing value임을 알아낼 수 있다.

**`Fill missing value with RandomForestClassifier`**

- missing value가 있는 feature를 missing value가 없는 feature들을 이용하여 RandomForestClassifier를 학습시켜 값을 채울 수 있다.
- 기존 통계값들로만 채우는 보다 의미있는 값을 채울 수 있다.

**`metrics.make_scorer`**

- `metrics.make_scorer` 을 이용하여 내가 직접 정의한 오차 함수를 `GridSearCV` 의 scorer로 사용할 수 있다.

**`Check distribution`**

- trian 데이터의 label값들의 분포와, model의 predict 값들의 분포를 시각화 후 비교해서 학습 정도를 파악해볼 수 있다.