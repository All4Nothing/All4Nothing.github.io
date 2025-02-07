---
title: "Titanic - Machine Learning from Disaster"
excerpt: "Competition Review"

categories:
  - Machine Learning
tags:
  - [Competition]
use_math: true

permalink: /machine-learning/titanic-competition/

toc: true
toc_sticky: true

date: 2024-03-17
last_modified_at: 2024-03-17
---

[Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

1. [Baseline] Logistic Regression with Only Simple Data PreProcessing
- Complete Missing Value
- Drop Feature
- Convert a categorical feature
- Model : Logistic Regression

> Score : 0.76555
> 
> 
> **Rank : 12142 (Top 78%)**
> 
1. Logistic Regression with Feature Engineering

> Score : 0.77272
> 
> 
> **Rank : 10045 (Top 64%)**
> 
1. Various Model with Deeper Feature Engineering

Support Vector Machine

> Score : 0.76794
> 

K - Nearest Neighbors

> Score : 0.76794
> 

Decision Tree

> Score : 0.76076
> 

Random Tree

> Score : 0.76076
> 
1. Ensemble(Voting) RandomForest, ExtraTrees, SVC, AdaBoost, GradientBoosting with GridSearchCV with Deeper Feature Engineering 

> Score : 0.78229
> 
> 
> **Rank : 2707 (Top 16%)**
> 

**유용했던 Techinque**

**`Analyze by pivoting features`**

- 데이터를 특정 feature를 기준으로 그룹화하여 관계를 파악하는데 유용했다.
- 여기서는 데이터의 각 feature(`Pclass`, `Sex`, `SibSp`, `Parch`)와 target(`Survived`)과의 상관성(생존 여부에 영향을 주는 feature인지)을 파악하는데 사용했다.
- 추가로, 새로 만든 `Has_Cabin`feature가 도움이 될만한지 확인하기 위해 사용하기도 했다.

```python
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```

**`FacetGrid`** 

- Grid 형태로 여러 개의 subplot을 생성하고 각각의 subplot에 데이터를 시각화할 수 있게 해줘, 데이터셋의 여러 측면을 살펴보기에 유용했다.
- 여기서는 `Age` feature의 null값을 연관있는 feature를 이용해 추정한 값으로 대체하기 위해, `Pclass`와 `Survived` 에 따른 `Age`의 분포를 파악하는데 사용했다.

```python
grid = sns.FacetGrid(train, col="Survived", row="Pclass", height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
```

**`Null값 추정`**

- `Age` feature의 null값을 채우기 위해, 단순 대치가 아닌 `Pclass`와 `Survived`에 따른 `Age` 분포를 이용하여, `Age` feature의 null값을 `Pclass`와 `Survived` 의 값에 따른 분포의 median값으로 대체하였다.

**`Numerical Feature의 Band feature 생성`**

- `Age` 와 `Fare` 와 같은 numerical feature를 그대로 사용하는 것이 아닌, 각 영역대로 분류하여 기존 feature 대신 `Age Band` , `Fare Band` feature를 만들어 사용하였다.

**`기존 feature들을 결합하여 새로운 feature 생성`**

- `Parch` feature와 `SibSp` feautre를 이용하여 `FamilySize` feature를 생성하고, 이로부터 `IsAlone` feature를 새롭게 도출하여 사용하였다.

**`정보 추출`**

- 학습에 도움이 되지 않을 것 같아 보통 drop 하는 `Name` feature에서 `Title`(호칭 ex) Mr, Mrs, Dr)정보를 추출하여 새로운 feature로 만들어 사용하였다.

**`pd.crosstab`**

- 데이터프레임을 사용하여 crosstabulation table을 만들어, 각 feature의 조합에 대한 빈도수를 확인할 수 있었다
- 여기서는 `Title` feature의 각 value에 대한 `Survived` 여부를 확인해볼 수 있었다.

```python
print(pd.crosstab(train['Title'],train['Survived']))
```

**`SklearnHelper class 정의`** 

- 각 모델들을 학습할 때 공통으로 사용되는 부분을 하나의 class로 정의하여 효율적으로 코드를 짤 수 있었다.

```python
class SklearnHelper(object):
    def __init__(self, clf, seed = 0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        
    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self, x, y):
        return self.clf.fit(x,y)
    
    def feature_importances(self, x, y):
        return self.clf.fit(x,y).feature_importances_
```

**`Plotly scatterplots`**

- Plotly package를 이용하여 각 모델의 feature importances 값들을 시각화할 수 있었다.

**`Plot learning curves`**

- `plot_learning_curve` 함수를 정의하여 learning curve를 시각화하여 overfitting, underfitting을 확인해볼 수 있었다.

Reference 

[Titanic Data Science Solutions](https://www.kaggle.com/code/startupsci/titanic-data-science-solutions)