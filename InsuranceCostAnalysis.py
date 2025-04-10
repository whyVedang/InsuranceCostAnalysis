import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
df = pd.read_csv(filepath,header=None)
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 100)   
df.columns=[['age','gender','bmi','noOfchildren','smoker','region','charges']]
df.replace("?",np.nan,inplace=True)
# print(df.head(10))
# print(df.info())
smoker=df['smoker'].value_counts().idxmax()
# print(type(smoker[0]))
freqsmoker=(int) (smoker[0])
# print(freqsmoker)
df['smoker']=df[['smoker']].fillna(freqsmoker)
# print(df['smoker'].head(30))
# print(df['smoker'].value_counts())
# print(df['age'].dtypes)
mean_age = df['age'].astype('float').mean(axis=0)
df['age']=df[['age']].fillna(mean_age)
# print(df['age'].tail(55))
# print(df['age'].info())
df[["charges"]] = np.round(df[["charges"]],2)
xdata=df['smoker']
ydata=df['charges']
# sns.regplot(x=xdata,y=ydata,data=df,line_kws={"color": "red"})
# sns.boxplot(x='smoker',y='charges',data=df)
# plt.ylim(0,)
# plt.show()
# print(df.corr())

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
lr=LinearRegression()
lr.fit(xdata,ydata)
lr.predict(xdata)
lr1=LinearRegression()
bigdata=df[["age", "gender", "bmi", "noOfchildren", "smoker", "region"]]
lr1.fit(bigdata,ydata)
lr1.predict(bigdata)
# print(lr1.score(bigdata,ydata))
Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(bigdata,ydata)
pred=pipe.predict(bigdata)
# print(r2_score(ydata,pred))

xtrain,xtest,ytrain,ytest=train_test_split(bigdata,ydata,test_size=0.20,random_state=0)

from sklearn.linear_model import Ridge
RR=Ridge(alpha=0.1)
RR.fit(xtrain,ytrain)
predRR=RR.predict(xtest)
# print('predicted:', pred[0:4])
# print('test set :', ytest[0:4].values)

pr=PolynomialFeatures(degree=2)
xtrainpr=pr.fit_transform(xtrain)
xtestpr=pr.fit_transform(xtest)
RR.fit(xtrainpr,ytrain)
predPR=RR.predict(xtestpr)
print('predicted:', pred[0:4])
print(r2_score(ytest,predPR))