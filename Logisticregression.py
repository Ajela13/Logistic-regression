#We need to predict if an employee will get promoted or not

import pandas as pd
import io

import numpy as np
#libraries for data visualization
import matplotlib.pyplot as plt
from plotly import subplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
#We will use sklearn for building logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score,auc
from sklearn.metrics import roc_curve
from warnings import simplefilter
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning
simplefilter('ignore',category=ConvergenceWarning)


data=pd.read_csv("HR_Data.csv") 

#Building Data understanding 

#shape of dataset
print('shape of dataframe is:', data.shape)
data.info()
data.describe()

#Data cleaning

#Drop employee_id column as it is just a unique id
data.drop('employee_id',inplace=True,axis=1)

#checking number of null values
data.isnull().sum()

#checking null percentage
data.isnull().mean()*100

#fill missing value
#We will fill previous user rating with "0" if not present (One can fill with anything else as well based on their understanding of data)
data['previous_year_rating']=data['previous_year_rating'].fillna(0)
#change type of int
data['previous_year_rating']=data['previous_year_rating'].astype('int')

#find out mode value for education
data['education'].mode()
#fill missing value with mode
data['education']=data['education'].fillna("Bachelor's")

#Chart for distribution of target variable
#(I can use any visualization or chart type I'm is comfortable with)
fig=plt.figure(figsize=(10,3))
fig.add_subplot(1,2,1)
a=data["is_promoted"].value_counts(normalize=True).plot.pie()
fig.add_subplot(1,2,2)
churnchart=sns.countplot(x=data['is_promoted'])
plt.tight_layout()
plt.show()


#visualiza relationship between promoted and other features
fig=plt.figure(figsize=(10,5))
fig.add_subplot(1,3,1)
ar_6=sns.boxplot(x=data['is_promoted'],y=data['length_of_service'])
fig.add_subplot(1,3,2)
ar_6=sns.boxplot(x=data['is_promoted'],y=data['avg_training_score'])
fig.add_subplot(1,3,3)
ar_6=sns.boxplot(x=data['is_promoted'],y=data['previous_year_rating'])
plt.tight_layout()
plt.show()


#correlation between features 
corr_plot=sns.heatmap(data.corr().round(2),annot=True,linewidths=3)
plt.title('Correlation plot')
plt.show()

#Age and length of service are correlated (though not a very strong correlation)

#List of other categorical variables 
categorical_cols=data.select_dtypes(['object']).columns
print(categorical_cols)

#create dummy variables
ds=pd.get_dummies(data[categorical_cols],drop_first=True)
ds.head


#concat newly created columns with original dataframe
data=pd.concat([data,ds],axis=1)
#Drop original columns
data.drop(categorical_cols,axis=1,inplace=True)



#Splitting Data into test and train 
from sklearn.model_selection import train_test_split
# split data into dependent variables x and independent variables y that we would predict 
Y=data.pop("is_promoted")
X=data

#Let's split X and Y using train test split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)
#Get shape of train and test data
print("train size X:",X_train.shape)
print("train size Y:",Y_train.shape)
print("test size X:",X_test.shape)
print("test size Y:",Y_test.shape)

#Check for distribution of labels
#our 91% of data is "not promoted" while "9%" of data is of employees whoe are promoted 
Y_train.value_counts(normalize=True)
Y_train.value_counts()
#unbalance calassification problem

#import library
from sklearn.linear_model import LogisticRegression
#make instance of model with default parameters except class weight as we will add class weights due to class unbalance problem
lr_basemodel=LogisticRegression()
#train to learn relationships between input and output variables
lr_basemodel.fit(X_train,Y_train)

y_pred_test=lr_basemodel.predict(X_test)
confusionMatrix=confusion_matrix(Y_test,y_pred_test)
print(confusionMatrix)
disp=ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=lr_basemodel.classes_)
disp.plot()
plt.show()

#visuailization with ploty
# fig = px.imshow(confusionMatrix, text_auto=True, aspect="auto")
# fig.update_layout(title= 'Confusion Matrix',
#     xaxis_title='Predicted label',
#     yaxis_title='True label',)
# fig.show()

y_pred_test=lr_basemodel.predict(X_test)
print("f1 score for base model is:",f1_score(Y_test,y_pred_test))

#Accuracy score 
print("accuracy score test dataset: t", accuracy_score(Y_test,y_pred_test))
#precision score
print("precision score test dataset: t", precision_score(Y_test,y_pred_test))
#recall score
print("recall score test dataset: t", recall_score(Y_test,y_pred_test))

fpr,tpr,threshold=roc_curve(Y_test,lr_basemodel.predict_proba(X_test)[:,1])
auc_var=auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,label='logistic Regression (area=%0.2f)'%auc_var)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()

#Giving different weight to each class 

from sklearn.linear_model import LogisticRegression
lr_diff_weight_model=LogisticRegression(class_weight={0:1,1:10})
lr_diff_weight_model.fit(X_train,Y_train)


y_pred_test=lr_diff_weight_model.predict(X_test)
confusionMatrix=confusion_matrix(Y_test,y_pred_test)
print(confusionMatrix)
disp=ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=lr_basemodel.classes_)
disp.plot()
plt.show()

#visualization with ploty
# fig = px.imshow(confusionMatrix, text_auto=True, aspect="auto")
# fig.update_layout(title= 'Confusion Matrix',
#     xaxis_title='Predicted label',
#     yaxis_title='True label',)
# fig.show()

#Accuracy score 
print("accuracy score test dataset: t", accuracy_score(Y_test,y_pred_test))
#precision score
print("precision score test dataset: t", precision_score(Y_test,y_pred_test))
#recall score
print("recall score test dataset: t", recall_score(Y_test,y_pred_test))


fpr_2,tpr_2,threshold_2=roc_curve(Y_test,lr_diff_weight_model.predict_proba(X_test)[:,1])
auc_var_2=auc(fpr_2,tpr_2)


plt.figure()
plt.plot(fpr,tpr,label='logistic Regression 1 (area=%0.2f)'%auc_var)
plt.plot(fpr_2,tpr_2,label='logistic Regression 2 (area=%0.2f)'%auc_var_2)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

#visualization with ploty
# fig = go.Figure()
# fig.add_traces([go.Line(x=fpr, y=tpr,name='logistic Regression 1 (area=%0.2f)'%auc_var),
#                 go.Line(x=fpr_2, y=tpr_2,name='logistic Regression 2 (area=%0.2f)'%auc_var_2),
# ])
# fig.add_shape(
#     type='line', line=dict(dash='dash'),
#     x0=0, x1=1, y0=0, y1=1
# )
# fig.update_layout(
#     title= 'ROC Curve',
#     xaxis_title='False Positive Rate',
#     yaxis_title='True Positive Rate',
#     width=700, height=500
# )
# fig.show()

#bayesian optimization 
from skopt import BayesSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


lr_diff_bayesian_model= LogisticRegression(max_iter=2000)
param_space = {'C': (1e-6, 1e+6, 'log-uniform'),
               'class_weight': [None, 'balanced']}


scorer = make_scorer(recall_score, greater_is_better=True)
opt = BayesSearchCV(lr_diff_bayesian_model, param_space, n_iter=50, cv=5, scoring=scorer, n_jobs=-1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

opt.fit(X_train_scaled, Y_train)
print("Best hyperparameters:", opt.best_params_)
X_test_scaled = scaler.transform(X_test)

recall = opt.score(X_test_scaled, Y_test)
print("Recall on the test set:", recall)

y_pred_test = opt.predict(X_test_scaled)
confusionMatrix=confusion_matrix(Y_test,y_pred_test)
print(confusionMatrix)
disp=ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=lr_basemodel.classes_)
disp.plot()
plt.show()


#Accuracy score 
print("accuracy score test dataset: t", accuracy_score(Y_test,y_pred_test))
#precision score
print("precision score test dataset: t", precision_score(Y_test,y_pred_test))
#recall score
print("recall score test dataset: t", recall_score(Y_test,y_pred_test))


fpr_3,tpr_3,threshold_3=roc_curve(Y_test,opt.predict_proba(X_test_scaled)[:,1])
auc_var_3=auc(fpr_3,tpr_3)
plt.figure()
plt.plot(fpr,tpr,label='logistic Regression 1 (area=%0.2f)'%auc_var)
plt.plot(fpr_2,tpr_2,label='logistic Regression 2 (area=%0.2f)'%auc_var_2)
plt.plot(fpr_3,tpr_3,label='logistic Regression 3 (area=%0.2f)'%auc_var_3)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


#visualization with ploty
# fig = go.Figure()
# fig.add_traces([go.Line(x=fpr, y=tpr,name='logistic Regression 1 (area=%0.2f)'%auc_var),
#                 go.Line(x=fpr_2, y=tpr_2,name='logistic Regression 2 (area=%0.2f)'%auc_var_2),
#                 go.Line(x=fpr_3,y=tpr_3,name='logistic Regression 3 (area=%0.2f)'%auc_var_3)])

# fig.add_shape(
#     type='line', line=dict(dash='dash'),
#     x0=0, x1=1, y0=0, y1=1
# )
# fig.update_layout(
#     title= 'ROC Curve',
#     xaxis_title='False Positive Rate',
#     yaxis_title='True Positive Rate',
#     width=700, height=500
# )
# fig.show()

