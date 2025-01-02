# Wine-quality-predection
#importing libraries
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns
from sklearn import metrics

from warnings import filterwarnings
filterwarnings(action='ignore')

#reading csv
wine = pd.read_csv("winequalityN.csv")
print("Successfully Imported Data!")
wine.head()
successfully imported data!

print(wine.shape)

#information
wine.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14596 entries, 0 to 14595
Data columns (total 13 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   type                  14596 non-null  object 
 1   fixed acidity         14596 non-null  float64
 2   volatile acidity      14596 non-null  float64
 3   citric acid           14596 non-null  float64
 4   residual sugar        14596 non-null  float64
 5   chlorides             14596 non-null  float64
 6   free sulfur dioxide   14596 non-null  float64
 7   total sulfur dioxide  14596 non-null  float64
 8   density               14596 non-null  float64
 9   pH                    14596 non-null  float64
 10  sulphates             14596 non-null  float64
 11  alcohol               14596 non-null  float64
 12  quality               14596 non-null  int64  
dtypes: float64(11), int64(1), object(1)
memory usage: 1.4+ MB

#description
wine.describe(include='all')

#Types of data
wine.dtypes
type                     object
fixed acidity           float64
volatile acidity        float64
citric acid             float64
residual sugar          float64
chlorides               float64
free sulfur dioxide     float64
total sulfur dioxide    float64
density                 float64
pH                      float64
sulphates               float64
alcohol                 float64
quality                   int64
dtype: object

#finding null values
print(wine.isna().sum())

type                    0
fixed acidity           0
volatile acidity        0
citric acid             0
residual sugar          0
chlorides               0
free sulfur dioxide     0
total sulfur dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
dtype: int64

wine.update(wine.fillna(wine.mean()))
wine.head()
print(wine.isna().sum())

type                    0
fixed acidity           0
volatile acidity        0
citric acid             0
residual sugar          0
chlorides               0
free sulfur dioxide     0
total sulfur dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
dtype: int64

wine.groupby('quality').mean()

DATA ANALYSIS
import matplotlib.pyplot as plt

total = len(wine.index)
white_wine = wine[wine['type']=='white']
print("Percentage of white wine: ",(len(white_wine.index)/total)*100,"%")
red_wine = wine[wine['type']=='red']
print("Percentage of red wine: ",(len(red_wine.index)/total)*100,"%")

plt.pie([len(white_wine.index),len(red_wine.index)], colors = ['#635e6b','#EF5350'], labels = ['white wine','red wine'],startangle=90)
plt.title('Types in the wine chemical datasets')
plt.show()

Percentage of white wine:  67.11427788435188 %
Percentage of red wine:  32.88572211564812 %

![image](https://github.com/user-attachments/assets/9593d0b2-f7d2-4a51-be5b-f7fa69d9c9d1)


Countplot
sns.countplot(wine['type'])
plt.show()

sns.countplot(wine['quality'])
plt.show()

sns.countplot(wine['pH'])
plt.show()

sns.countplot(wine['alcohol'])
plt.show()

sns.countplot(wine['fixed acidity'])
plt.show()

sns.countplot(wine['volatile acidity'])
plt.show()

sns.countplot(wine['citric acid'])
plt.show()

sns.countplot(wine['density'])
plt.show()

KDE Plot
sns.kdeplot(wine.query('quality > 2').quality)

DISTPLOT
sns.distplot(wine['alcohol'])

wine.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)

Histogram
wine.hist(figsize=(10,10),bins=50)
plt.show()

PAIR PLOT
sns.pairplot(wine)

wine.corr()

corr = wine.corr()

wine.corr()['quality'].sort_values()

plt.figure(figsize=[19,10], facecolor = 'white')
sns.heatmap(wine.corr(),annot=True)

plt.figure(figsize=(14,4))
plt.subplot(1,4,1)
sns.barplot(x = 'quality', y = 'alcohol', data = wine,palette='rainbow')
plt.subplot(1,4,2)
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine,palette='rainbow')
plt.tight_layout()
plt.subplot(1,4,3)
sns.barplot(x = 'quality', y = 'sulphates', data = wine,palette='rainbow')
plt.tight_layout()
plt.subplot(1,4,4)
sns.barplot(x = 'quality', y = 'citric acid', data = wine,palette='rainbow')
plt.tight_layout()

violing plot
sns.violinplot(x='quality', y='alcohol', data=wine)

FINDING FEATURES WITH MAX CORELATION VALUE
for a in range(len(wine.corr().columns)):
    for b in range(a):
        if abs(wine.corr().iloc[a,b]) > 0.7 :
            name = wine.corr().columns[a]
            print(name)

REMOVING THE FEATURE
new_wine = wine.drop(name,axis=1)
new_wine.head()

FINDING NUMBER OF NULL VALUES IN EACH FEATURE COLUMN
new_wine.isnull().sum()

FILLING THE NULL VALUES
new_wine.update(new_wine.fillna(new_wine.mean()))
new_wine.head()

HANDELING CATEGORICAL VARIABLES
next_wine = pd.get_dummies(new_wine,drop_first = True)
next_wine

CREATING BEST QUALITY COLUMN
next_wine['best quality'] = [1 if x>=7 else 0 for x in wine.quality]
next_wine

next_wine['quality'].value_counts()

next_wine['best quality'].value_counts()

CREATING INPUT AND OUTPUT DATABASE
x = next_wine.drop(['quality','best quality'], axis=1)
x
y = next_wine['best quality']
y
SPILTING DATABASE TO TRAINING AND TESTING
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=40)

print(x.shape, x_test.shape, x_train.shape)

print(y.shape, y_test.shape, y_train.shape)

NORMALIZE
from sklearn.preprocessing import MinMaxScaler
# creating normalization object 
norm = MinMaxScaler()
# fit data
norm_fit = norm.fit(x_train)
new_xtrain = norm_fit.transform(x_train)
new_xtest = norm_fit.transform(x_test)
# display values
print(new_xtrain)
az=sns.displot(new_xtrain,kind="kde",color="#e64e4e",height=10,aspect=2,linewidth=5)

LOGISTICREGRESSION:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score

lr = LogisticRegression()
lr.fit(new_xtrain, y_train)
prediction = lr.predict(new_xtest)
print("accuracy_score is:",accuracy_score(prediction, y_test))
print(classification_report(prediction, y_test))
sns.heatmap(data=confusion_matrix(prediction, y_test), annot=True, cmap=plt.cm.Reds_r, fmt='d')
RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(new_xtrain, y_train)
prediction_rfc = rfc.predict(new_xtest)
print("accuracy_score is:",accuracy_score(prediction_rfc, y_test))
print(classification_report(prediction_rfc, y_test))
sns.heatmap(data=confusion_matrix(prediction_rfc, y_test), annot=True, cmap=plt.cm.Reds_r, fmt='d')

SVC
from sklearn.svm import SVC
svc = SVC(probability=True)
svc.fit(new_xtrain, y_train)
prediction_svc = svc.predict(new_xtest)
print("accuracy_score is:",accuracy_score(prediction_svc, y_test))
print(classification_report(prediction_svc, y_test))
sns.heatmap(data=confusion_matrix(prediction_svc, y_test), annot=True, cmap=plt.cm.Reds_r, fmt='d')

KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(new_xtrain, y_train)
prediction_knn = knn.predict(new_xtest)
print("accuracy_score is:",accuracy_score(prediction_knn, y_test))
print(classification_report(prediction_knn, y_test))
sns.heatmap(data=confusion_matrix(prediction_knn, y_test), annot=True, cmap=plt.cm.Reds_r, fmt='d')

XGBOOST
**from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(new_xtrain, y_train)
prediction_xgb = xgb.predict(new_xtest)
print("accuracy_score is:",accuracy_score(y_test, prediction_xgb))
print(classification_report(prediction_xgb, y_test))
sns.heatmap(data=confusion_matrix(prediction_xgb, y_test), annot=True, cmap=plt.cm.Reds_r, fmt='d')**

GAUSSIAN NB:
from sklearn.naive_bayes import GaussianNB
gnb =GaussianNB()
gnb.fit(new_xtrain, y_train)
prediction_gnb = gnb.predict(new_xtest)
print("accuracy_score is:",accuracy_score(y_test, prediction_gnb))
print(classification_report(prediction_gnb, y_test))
sns.heatmap(data=confusion_matrix(prediction_gnb, y_test), annot=True, cmap=plt.cm.Reds_r, fmt='d')

DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dtc =DecisionTreeClassifier(criterion='entropy',random_state=7)
dtc.fit(new_xtrain, y_train)
prediction_dtc = dtc.predict(new_xtest)
print("accuracy_score is:",accuracy_score(y_test, prediction_dtc))
print(classification_report(prediction_dtc, y_test))
sns.heatmap(data=confusion_matrix(prediction_dtc, y_test), annot=True, cmap=plt.cm.Reds_r, fmt='d')

MODEL EVALUTION
R2 score
Higher R2 score is better
lr_score = metrics.r2_score(y_test, prediction)
rfc_score = metrics.r2_score(y_test, prediction_rfc)
knn_score = metrics.r2_score(y_test, prediction_knn)
xgb_score = metrics.r2_score(y_test, prediction_xgb)
gnb_score = metrics.r2_score(y_test, prediction_gnb)
dtc_score = metrics.r2_score(y_test, prediction_dtc)
svc_score = metrics.r2_score(y_test, prediction_svc)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("*"*20, "R2 Score", "*"*20)

print("-"*50)
print("| LogisticRegression : ", lr_score)
print("-"*50)

print("-"*50)
print("| KNearest Neighbors: ", knn_score)
print("-"*50)

print("-"*50)
print("| Decision Tree: ", dtc_score)
print("-"*50)

print("-"*50)
print("| Gaussian Naïve Bayes: ", gnb_score)
print("-"*50)

print("-"*50)
print("| XGBoost: ", xgb_score)
print("-"*50)

print("-"*50)
print("| Support Vector Classifier: ", svc_score)
print("-"*50)

print("-"*50)
print("| Random Forest: ", rfc_score)
print("-"*50)


metric_val = {
    "R2 score": {
    "Logistic Regression ": lr_score,
    "KNearest Neighbors": knn_score,
    "Decision Tree": dtc_score,
    "Gaussian Naïve Bayes": gnb_score,
    "XGBoost": xgb_score,
    "Support Vectore Classifier": svc_score,
    "Random Forest": rfc_score,
    }
}

ax = pd.DataFrame(metric_val).plot(kind="bar", 
                             figsize = (20,10), 
                             legend =False, 
                             title = "R2 Score",
                             color = '#4633FF');
                    
for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.005, p.get_height() * 1.005))

Mean square error
lower MSE is better

lr_score_MSE = metrics.mean_squared_error(y_test, prediction)
knn_score_MSE = metrics.mean_squared_error(y_test, prediction_knn)
dtc_score_MSE = metrics.mean_squared_error(y_test, prediction_dtc)
gnb_score_MSE = metrics.mean_squared_error(y_test, prediction_gnb)
xgb_score_MSE = metrics.mean_squared_error(y_test, prediction_xgb)
svc_score_MSE = metrics.mean_squared_error(y_test, prediction_svc)
rfc_score_MSE = metrics.mean_squared_error(y_test, prediction_rfc)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("*"*20, "Mean Squared Error", "*"*20)

print("-"*50)
print("| Logistic Regression: ", lr_score_MSE)
print("-"*50)

print("-"*50)
print("| KNearest Neighbors: ", knn_score_MSE)
print("-"*50)

print("-"*50)
print("| Decision Tree: ", dtc_score_MSE)
print("-"*50)

print("-"*50)
print("| Gaussian Naïve Bayes: ", gnb_score_MSE)
print("-"*50)

print("-"*50)
print("| XGBoost: ", xgb_score_MSE)
print("-"*50)

print("-"*50)
print("| Support Vector Classifier: ", svc_score_MSE)
print("-"*50)

print("-"*50)
print("| Random Forest: ", rfc_score_MSE)
print("-"*50)

metric_val = {
    "Mean Squared Error": {
    "Logistic Regression ": lr_score_MAE,
    "KNearest Neighbors": knn_score_MAE,
    "Decision Tree": dtc_score_MAE,
    "Gaussian Naïve Bayes": gnb_score_MAE,
    "XGBoost:": xgb_score_MAE,
    "Support Vector Classifier": svc_score_MAE,
    "Random Forest": rfc_score_MAE,
    }
}

ax = pd.DataFrame(metric_val).plot(kind="bar", 
                             figsize = (20,10), 
                             legend =False, 
                             title = "Mean Absolute Error",
                             color = '#4633FF');
                    
for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.005, p.get_height() * 1.005))

ROC AUC SCORE AND CURVE
from sklearn.metrics import roc_auc_score, roc_curve
ans_lr = lr.predict_proba(new_xtest)
ans_knn = knn.predict_proba(new_xtest)
ans_svc = svc.predict_proba(new_xtest)
ans_rfc = rfc.predict_proba(new_xtest)
ans_xgb = xgb.predict_proba(new_xtest)
ans_gnb = gnb.predict_proba(new_xtest)
ans_dtc = dtc.predict_proba(new_xtest)


print(f'Logistics Regression : {roc_auc_score(y_test, ans_lr[:,1])}')
print(f'Random Forest Classifier : {roc_auc_score(y_test, ans_rfc[:,1])}')
print(f'KNN : {roc_auc_score(y_test, ans_knn[:,1])}')
print(f'SVC : {roc_auc_score(y_test, ans_svc[:,1])}')
print(f'XGB : {roc_auc_score(y_test, ans_xgb[:,1])}')
print(f'GNB : {roc_auc_score(y_test, ans_gnb[:,1])}')
print(f'DTC : {roc_auc_score(y_test, ans_dtc[:,1])}')

    plt.figure(figsize=(6,6), dpi=300)

fpr, tpr, thresholds = roc_curve(y_test, ans_lr[:,1])
plt.plot(fpr, tpr, color='orange', marker='o', linestyle='dashed',linewidth=2, markersize=2,
         label='Logistic Regression')

fpr_rfc, tpr_rfc, thresholds_rfc = roc_curve(y_test, ans_rfc[:,1])
plt.plot(fpr_rfc, tpr_rfc, color='green', marker='o', linestyle='dashed',linewidth=2, markersize=2,
         label='Random Forest Classifier')

fpr_knn, tpr_knn, thresholds = roc_curve(y_test, ans_knn[:,1])
plt.plot(fpr_knn, tpr_knn, color='blue', marker='o', linestyle='dashed',linewidth=2, markersize=2,
         label='KNN')
fpr_svc, tpr_svc, thresholds = roc_curve(y_test, ans_svc[:,1])
plt.plot(fpr_svc, tpr_svc, color='red', marker='o', linestyle='dashed',linewidth=2, markersize=2,
         label='SVC')
fpr_xgb, tpr_xgb, thresholds = roc_curve(y_test, ans_xgb[:,1])
plt.plot(fpr_xgb, tpr_xgb, color='pink', marker='o', linestyle='dashed',linewidth=2, markersize=2,
         label='XGB')
fpr_xgb, tpr_xgb, thresholds = roc_curve(y_test, ans_gnb[:,1])
plt.plot(fpr_xgb, tpr_xgb, color='#4633FF', marker='o', linestyle='dashed',linewidth=2, markersize=2,
         label='GNB')
fpr_xgb, tpr_xgb, thresholds = roc_curve(y_test, ans_dtc[:,1])
plt.plot(fpr_xgb, tpr_xgb, color='yellow', marker='o', linestyle='dashed',linewidth=2, markersize=2,
         label='DTC')



plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()

USER INPUT
input_data = (6.2,0.66,0.48,1.2,0.029,29,0.9892,3.33,0.39,12.8,1)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = rfc.predict(input_data_reshaped)
print(prediction)

if(prediction==0):
    print('Bad Quality Wine')
else:
    print('Good Quality Wine')

