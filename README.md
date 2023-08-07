# Financial-Habits-Analysis-for-Loan-Prediction
### Abstract:

This study aims to analyze the impact of financial habits on loan prediction. By examining various financial behaviors and patterns, we seek to identify key factors that can predict an individual's creditworthiness. The analysis is based on a diverse dataset comprising financial records, credit histories, and borrower information. Employing machine learning techniques, this research can assist lenders in making more informed decisions while extending credit to potential borrowers. 
　　 <br /> 

![Photo by [Ryoji Iwata]([https://unsplash.com/@ryoji__iwata?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/photos/a-qsFZimp1M?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)](https://images.unsplash.com/photo-1512799545738-0625ef92a288?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb](https://images.unsplash.com/photo-1512799545738-0625ef92a288?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1169&q=80))

Photo by [Ryoji Iwata](https://unsplash.com/@ryoji__iwata?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/photos/a-qsFZimp1M?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 
  <br /> 

### Background and motivation

In today's rapidly evolving financial landscape, lending institutions face an ever-increasing need to make informed and prudent decisions when granting loans. The economic environment, individual financial behaviors, and creditworthiness have a profound impact on the repayment capacity of borrowers. Traditional lending practices, although effective to some extent, often fail to capture the complexity of modern borrowers' financial habits. As a result, financial institutions encounter challenges in accurately predicting the likelihood of loan defaults, leading to potential losses and increased risk.

---

### Case Study

Our case study focuses on a specialized P2P lending company that aims to identify potential borrowers who qualify for loans based on their financial information. The company intends to use the completion of the final stage, "e-signature," in the loan application process as a criterion for assessing eligibility. The goal is to offer new loan proposals tailored to borrowers who have not completed the e-signature, thereby increasing the overall loan application completion rate.

### Objectives of the Study:

The primary objective of this research is to conduct a comprehensive analysis of financial habits and their impact on loan prediction. By analyzing a diverse dataset comprising financial records, credit histories, and borrower information, we aim to uncover patterns and trends that can serve as valuable predictors of creditworthiness. Through a combination of machine learning techniques and feature engineering, our study seeks to develop a robust loan prediction model capable of assisting financial institutions in making more accurate and reliable lending decisions.

The primary objectives of this research are as follows:

1. **Conducting Financial Habits Analysis**: We analyze borrower information to identify key financial habits influencing loan repayment capacity, providing a comprehensive view of borrowers' financial health.
2. **Feature Engineering and Model Development**: We engineer relevant features based on insights from financial habits analysis and build a tailored, robust machine learning model for P2P lending data.
3. **Enhancing Loan Default Prediction**: Our model is fine-tuned for high accuracy, precision, and recall, effectively identifying high-risk and creditworthy borrowers.
4. **Assessing Model Performance**: We evaluate the model's effectiveness using Accuracy, Precision, Recall, F1 score, and AUC-ROC metrics, employing cross-validation to ensure reliability and generalizability.
5. **Practical Implementation and Implications:** We discuss actionable strategies for P2P lending platforms to optimize loan portfolios and mitigate risk based on our findings.

---

### **Data**

The data is from the "[**E-Signing of Loan Based on Financial History**](https://www.kaggle.com/datasets/yashpaloswal/esigning-of-loanbased-on-financial-history)" dataset on Kaggle, comprising 17,908 records with 21 features (17 numerical, 1 categorical, and 3 boolean). It includes loan request characteristics, borrowers' personal statistics, and credit-related information.

Data Description:

| Feature Name: | Definition Explanation: |
| --- | --- |
| entry_id | The identification number of the borrower. |
| age | Age of the borrower. |
| pay_schedule | The frequency of the borrower's salary received (weekly, bi-weekly, monthly, semi-monthly). |
| home_owner | Whether the borrower owns a property (Yes: 1, No: 0). |
| income | Monthly income of the borrower (US dollars). |
| months_employed | Number of months employed by the borrower. |
| years_employed | Number of years employed by the borrower. |
| current_address_year | Number of years the borrower has been at the current address. |
| personal_account_m | Number of months the borrower has owned a personal account. |
| personal_account_y | Number of years the borrower has owned a personal account. |
| has_debt | Whether the borrower has existing debt (Yes: 1, No: 0). |
| amount_requested | The loan amount applied for by the borrower (US dollars). |
| risk_score | Risk scores calculated by different algorithms. |
| risk_score_2 | Risk scores calculated by different algorithms. |
| risk_score_3 | Risk scores calculated by different algorithms. |
| risk_score_4 | Risk scores calculated by different algorithms. |
| risk_score_5 | Risk scores calculated by different algorithms. |
| ext_quality_score | Credit quality scores calculated by different algorithms. |
| ext_quality_score_2 | Credit quality scores calculated by different algorithms. |
| inquiries_last_month | Number of inquiries made by the borrower in the previous month. |
| e_signed | Whether electronic signature has been completed (Yes: 1, No: 0). |

---

[Google Colaboratory](https://colab.research.google.com/drive/18As73eKeJvVxsxcAq0Kx9eVzO7mj8URh?usp=sharing)

### ****Load Essential Python Libraries****

```python
# Data analysis suite (numpy, pandas)
import numpy as np 
import pandas as pd 

# Data visualization suite (matplotlib, seaborn)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import random
import time
sns.set_style("whitegrid")

random.seed(100)
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 80)
pd.set_option('display.float', '{:.2f}'.format)
data = pd.read_csv('Financial-Data.csv')
```

```python
# Output Data Characteristics
# 17908 total data and 21 features
data
```

![Output Data Characteristics](https://drive.google.com/uc?export=view&id=1X1yTkaYbNkLwLsEQnbndhHG4OGuykMVw)

```python
# Statistical variables of the data structure
data.describe()
```

![Statistical variables of the data structure](https://drive.google.com/uc?export=view&id=1TsBLhjpDeQ6UHQ-40oGfEw3uyERCy5dZ)


```python
# Identify missing values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
data.isnull().sum()
```

![Identify missing values](https://drive.google.com/uc?export=view&id=19vesqZcif7X-r9Cn_7YM4GV_Dmkkd2X6)

---

### EDA (Exploratory Data Analysis)

**Histograms of Numerical Columns**

```python
# Histograms of Numerical Columns
dataset = data.drop(columns = ['Entry_id', 'pay_schedule', 'e_signed'])
fig = plt.figure(figsize=(15, 12),dpi=300)
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset.columns.values[i])

    vals = np.size(dataset.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
plt.hist(dataset.iloc[:, i], bins=vals, color='#3d6dae')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Histograms of Numerical Columns', transparent=True)
```

![**Histograms of Numerical Columns**](https://drive.google.com/file/d/1pIrf4mxn2629Dq9iQBmelfM6cSUNCkrs/view?usp=sharing)

**Histograms of Numerical Columns**

The histograms of the dataset reveal that **most features exhibit a normal distribution**, while the **months employed (months_employed) feature demonstrates a skewed distribution**.

- The primary age range of borrowers falls between 30 and 50 years old
- Monthly income is concentrated in the range of 2000 to 4000 US dollars
- The majority of borrowers have at least two years of work experience
- Borrowers commonly possess both debts and assets
- The average loan amount is below 4000 US dollars

****Bivariate Analysis****

```python
# Bivariate Analysis
plt.figure(figsize=(20, 32),dpi=300)
for i, column in enumerate(data.drop('e_signed', axis=1).columns, 1):
    plt.subplot(6, 4, i)
    data[data["e_signed"] == 0][column].hist(bins=35, color='#da3b46', label='e_signed = NO', alpha=0.75,edgecolor = 'r')
    data[data["e_signed"] == 1][column].hist(bins=35, color='#477ca8', label='e_signed = YES', alpha=0.7,edgecolor = 'b')
    plt.legend()
    plt.xlabel(' ')
    plt.title(f'{column}')
plt.savefig('Bivariate Analysis', transparent=True)
```

![Histogram of Two Variables](https://drive.google.com/file/d/1NMollMFk-VWCZpad_aBXAEySP8BpNxsj/view?usp=sharing)

Histogram of Two Variables

The Bivariate Analysis reveal that **borrower's age, income, and amount requested are positively correlated with default risk**.

- **The older the borrower, the higher the default probability**: Borrowers aged 50 or older have an increasing likelihood of default.
- **Lower income leads to higher default risk**: Borrowers with a monthly income below 4000 US dollars have a higher chance of default.
- **Lower loan amounts are associated with higher default risk**: When the loan amount is relatively low, the probability of default increases.

**Correlation with E-Signed**

```python
# Correlation with E-Signed
plt.figure(dpi=300)
data2=data.drop(columns=['entry_id','e_signed'])
data2.corrwith(data.e_signed).plot.bar(
        figsize = (20, 10), fontsize = 15, rot = 45, grid = True ,color='#3d6dae')
sns.set_style("white")
plt.suptitle('Correlation with E-Signed', fontsize=20)
plt.savefig('Correlation with E-Signed.png', transparent=True)
```

![**Correlation with E-Signed**](https://drive.google.com/file/d/1_kQIiWguzoCQjqkP8xJXL21E7UugLSTm/view?usp=sharing)

**Correlation with E-Signed**

The bar chart shows that **financial factors** like income, loan mount, and debt status are positively correlated, while **personal factors** exhibit a negative correlation. Additionally, **age is influenced by interactions with various factors**.

**Positive correlation**: risk_score, amount_requested, has_debt, monthers(years)_employed, income.

**Negative correlation**: age, home owner, personal_account_m, inquiries_last_month,  current_address_year, risk scores calculated by different algorithms, Credit quality scores.

---

### Data Preprocessing

**Data Cleaning**

Data cleaning resulted in the **removal of 5 features**, leaving 19 features remaining.

| Removed Feature | Reason |
| --- | --- |
| entry_id | Identifiers of customers do not provide assistance in predicting whether the loan is completed. |
| months/years_employed | Merge the years into months for features with the same time units, and delete the year column afterward. |
| personal_account_m/y | Merge the years into months for features with the same time units, and delete the year column afterward. |

```python
# Combine yearly minutes into months
data['months_employed']=(data.months_employed+data.years_employed*12)
data.drop(columns=['years_employed'],inplace=True)

data['personal_account_m']=(data.personal_account_m+data.personal_account_y*12)
data.drop(columns=['personal_account_y'],inplace=True)
data.head()# Combine yearly minutes into months
data['months_employed']=(data.months_employed+data.years_employed*12)
data.drop(columns=['years_employed'],inplace=True)

data['personal_account_m']=(data.personal_account_m+data.personal_account_y*12)
data.drop(columns=['personal_account_y'],inplace=True)
data.head()
```

**Categorical feature handling**

The pay_schedule, representing the frequency of salary received, is converted into 0 or 1 dummy variables using one-hot encoding. One feature is dropped to avoid multicollinearity.

![Untitled](Financial%20Habits%20Analysis%20for%20Loan%20Prediction%20cabaab69c17c49d4b3f0ed3f2c160797/Untitled%203.png)

```python
# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
# one hot encoding
enc = OneHotEncoder(sparse=False)
pay_schedule_onehot = enc.fit_transform(data[['pay_schedule']])
#to print the encoded features for train data
pd.DataFrame(pay_schedule_onehot, columns=list(enc.categories_[0]))

# Converting Characteristics to Values
data = pd.get_dummies(data)
data.drop(columns = ['pay_schedule_weekly'], axis = 1, inplace = True)
```

---

### ****Feature Engineering****

**Splitting the data into "Training Set" and "Test Set"**

Data is split into a "Training Set" (80%) and a "Test Set" (20%) for model training and evaluation.

```python
# Splitting the data into "Training Set" and "Test Set"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = ["e_signed", "entry_id"], axis =1), data.e_signed, test_size =0.2, random_state =0)

y_tr = pd.DataFrame(y_train)
y_ts = pd.DataFrame(y_test)
```

**Feature scaling  and Normalization**

Feature scaling and normalization **ensure each feature's values have a mean of 0 and a standard deviation of 1**, making them suitable for machine learning algorithms.

```python
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2

X_train
```

![Untitled](Financial%20Habits%20Analysis%20for%20Loan%20Prediction%20cabaab69c17c49d4b3f0ed3f2c160797/Untitled%204.png)

---

### Model Algorithm Training

Multiple algorithms were employed for model training, such as Logistic Regression, Support Vector Machine, Random Forest, XGBoost, and Back Propagation Neural Network.

**Import Performance Metrics**

During model training, the following performance metrics were used:

| Metric | Description |
| --- | --- |
| Confusion Matrix | Evaluates predicted outcomes against actual ones |
| Accuracy | Measures overall correct predictions |
| Precision | Assesses the model's ability to avoid false positives |
| Recall (Sensitivity) | Measures the model's ability to identify positive cases |
| F1 Score | Provides a balanced evaluation of precision and recall |

```python
# Import Performance Metrics (Confusion Matrix, Accuracy, Precision, Recall, F1 Score)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,f1_score
# Import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
# Import Model evaluation and comparison Report
from sklearn.metrics import classification_report
```

**Grid Search**

Grid search optimizes model performance by trying different hyperparameter combinations on the training set and selecting the **best performing** set for the model.

- **Logistic Regression**

```python
# GridSearch
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV

# Set all possible parameter ranges
model = LogisticRegressionCV()
solvers = ['newton-cg', 'lbfgs', 'liblinear',]
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

# Define the number of layers for grid search and cross-checking
# scoring is evaluated by accuracy
grid = dict(solver=solvers,penalty=penalty,Cs=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=10, scoring='accuracy',error_score=0)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=10, scoring='precision',error_score=0)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=10, scoring='recall',error_score=0)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=10, scoring='f1',error_score=0)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=10, scoring='roc_auc',error_score=0)

grid_result = grid_search.fit(X_train, y_train)

# Parameters for best output performance
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print("%f (%f) with: %r" % (mean, stdev, param))
```

```python
# Logistic Regression
from sklearn.linear_model import LogisticRegressionCV
lr = LogisticRegressionCV(Cs = 100, penalty = 'l2',solver='liblinear')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results
```

**Cross-Validation**

Cross-validation is a method for **testing model performance** by dividing data into segments, using one for testing and another for training. After multiple rounds of testing, results are averaged to ensure **stable model performance**.

```python
# cross_val_score
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lr, X= X_train, y = y_train,scoring='accuracy')
precision = cross_val_score(estimator = lr, X= X_train, y = y_train, scoring='precision')
recall = cross_val_score(estimator = lr, X= X_train, y = y_train, scoring='recall')
f1_scores=cross_val_score(estimator = lr, X= X_train, y = y_train,scoring='f1')

print("Logistic Regression Accuracy: %0.3f (+/- %0.3f)"  % (accuracies.mean(), accuracies.std() * 2))
print("Logistic Regression Precision: %0.3f (+/- %0.3f)"  % (precision.mean(), precision.std() * 2))
print("Logistic Regression Recall: %0.3f (+/- %0.3f)"  % (recall.mean(), recall.std() * 2))
print("Logistic Regression F1-Score: %0.3f (+/- %0.3f)"  % (f1_scores.mean(), f1_scores.std() * 2))
print(classification_report(y_test, y_pred))
```

Please find more examples of grid search algorithms, including Support Vector Machine, Random Forest, and XGBoost, BPNN in the attached file link.

---

### Model evaluation and comparison

**ROC/AUC Curve**

The ROC curve evaluates classification models by **plotting the true positive rate (sensitivity) against the false positive rate (1 - specificity)**. The **AUC** quantifies overall model performance.

```python
# predict probabilities
pred_prob1 = lr.predict_proba(X_test)
pred_prob2 = svm_linear.predict_proba(X_test)
pred_prob3 = svm_rbf.predict_proba(X_test)
pred_prob4 = rfen.predict_proba(X_test)
pred_prob5 = xgb.predict_proba(X_test)
```

```python
from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_test, pred_prob3[:,1], pos_label=1)
fpr4, tpr4, thresh4 = roc_curve(y_test, pred_prob4[:,1], pos_label=1)
fpr5, tpr5, thresh5 = roc_curve(y_test, pred_prob5[:,1], pos_label=1)

# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
```

```python
from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])
auc_score3 = roc_auc_score(y_test, pred_prob3[:,1])
auc_score4 = roc_auc_score(y_test, pred_prob4[:,1])
auc_score5 = roc_auc_score(y_test, pred_prob5[:,1])

print(auc_score1, auc_score2 ,auc_score3,auc_score4,auc_score5)
```

```python
# matplotlib
import matplotlib.pyplot as plt
plt.figure(dpi=300)
plt.style.use('default')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='-',color='tab:blue', label='Logistic Regression (AUC = 0.602)')
plt.plot(fpr2, tpr2, linestyle='-',color='tab:orange', label='SVM_Linear (AUC = 0.601)')
plt.plot(fpr3, tpr3, linestyle='-',color='tab:green', label='SVM_RBF (AUC = 0.647)')
plt.plot(fpr4, tpr4, linestyle='-',color='tab:red', label='Random Forest (AUC = 0.696 )')
plt.plot(fpr5, tpr5, linestyle='-',color='tab:pink', label='XGBooost (AUC = 0.71 )')
plt.plot(fpr_keras, tpr_keras, linestyle='-',color="turquoise",label='BPNN (AUC=0.664)')

plt.plot([0, 1], [0, 1], 'k--',color="grey")
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();
```

![Untitled](Financial%20Habits%20Analysis%20for%20Loan%20Prediction%20cabaab69c17c49d4b3f0ed3f2c160797/Untitled%205.png)

---

### ****Feature Importance****

Top influencing features from the high-performing XGBoost model are **amount_requested, rick_scores, personal_account_m, home_owner** and **age**. **has_debt** and **income** also matter in loan decisions.

```python
import numpy as np
import matplotlib.pyplot as plt

# Sort feature importances in descending order
indices = np.argsort(xgb.feature_importances_)[::1]
sorted_features = [X_train.columns[i] for i in indices]
sorted_importances = xgb.feature_importances_[indices]

# Plot sorted feature importances
plt.figure(dpi=300)
plt.style.use('default')
plt.title('Feature Importance')
plt.barh(sorted_features, sorted_importances, color='#3d6dae', height=0.6)
plt.show()
```

![Untitled](Financial%20Habits%20Analysis%20for%20Loan%20Prediction%20cabaab69c17c49d4b3f0ed3f2c160797/Untitled%206.png)

---

### **Conclusion & Final Results**

Summing up the evaluation from all indicators, here are the conclusions:

| Model | Accuracy | Precision | Recall | F1 Score | AUC Score |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.577 | 0.585 | 0.736 | 0.652 | 0.602 |
| SVM (Linear) | 0.585 | 0.694 | 0.725 | 0.653 | 0.601 |
| SVM (RBF) | 0.620 | 0.629 | 0.716 | 0.669 | 0.647 |
| Random Forest | 0.635 | 0.645 | 0.701 | 0.674 | 0.696 |
| XGBoost | 0.648 | 0.661 | 0.711 | 0.685 | 0.710 |
| BPNN | 0.618 | 0.609 | 0.810 | 0.695 | 0.664 |
1. **XGBoost** demonstrates the highest overall performance, followed by Random Forests. Therefore, **XGBoost is recommended** as the primary choice for practical implementation.
2. **Support Vector Machines (SVM)** and **Backpropagation Neural Networks (BPNN)** exhibit superior recall performance, effectively identifying a greater number of actual loan completions. For the pursuit of **higher recall**, these models should be given priority consideration.

[Google Colaboratory](https://colab.research.google.com/drive/16OsatkJ3jg6Ffrr26mUyF_Am6FER4l2u?usp=sharing)

[https://colab.research.google.com/drive/16OsatkJ3jg6Ffrr26mUyF_Am6FER4l2u?usp=sharing](https://colab.research.google.com/drive/16OsatkJ3jg6Ffrr26mUyF_Am6FER4l2u?usp=sharing)
