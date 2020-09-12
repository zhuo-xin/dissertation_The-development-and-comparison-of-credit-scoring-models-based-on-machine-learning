import numpy as np
import pandas as pd
data = pd.read_csv('credit_risk_dataset.csv')
data.head()

data.shape
data.describe()
data.info()

#change function: swap variable from string to integer
def change_homevalue(x):
    if x=='MORTGAGE':
       return 1
    elif x=='OTHER':
       return 2
    elif x=='OWN':
       return 3
    else:
       return 4   
data['person_home_ownership']=data['person_home_ownership'].apply(lambda x: change_homevalue(x) )

data['loan_intent'].unique()
def change_intentvalue(x):
    if x=='DEBTCONSOLIDATION':
       return 1
    elif x=='HOMEIMPROVEMENT':
       return 2
    elif x=='VENTURE':
       return 3
    elif x =='MEDICAL':
       return 4   
    elif x =='EDUCATION':
       return 5   
    elif x =='PERSONAL':
       return 6      
data['loan_intent']=data['loan_intent'].apply(lambda x: change_intentvalue(x) )

data['loan_grade'].unique()
def change_gradevalue(x):
    if x=='A':
       return 1
    elif x=='B':
       return 2
    elif x=='C':
       return 3
    elif x =='D':
       return 4   
    elif x =='E':
       return 5   
    elif x =='F':
       return 6      
    elif x =='G':
       return 7   
data['loan_grade']=data['loan_grade'].apply(lambda x: change_gradevalue(x) )

data['cb_person_default_on_file'].unique()
def change_defaultvalue(x):
    if x=='Y':
       return 1                   #yes-----1
    elif x=='N':
       return 0                   #no------0
data['cb_person_default_on_file']=data['cb_person_default_on_file'].apply(lambda x: change_defaultvalue(x) )   

#check which column has missing value
data.isnull().any()  

#delete data without figure about 'person_emp_length'
data = data.dropna(subset=["person_emp_length"])

#use random forest to impute missing value
import matplotlib.pyplot as plt
from  sklearn.ensemble import RandomForestClassifier

data=data.multiply(100)
#complete data set
known_rate_data = data[(data['loan_int_rate'].notna())] 
known_rata_X=known_rate_data.drop(columns=['loan_int_rate'])
known_rate_Y=known_rate_data['loan_int_rate']

known_rata_X=known_rata_X.astype(float)
known_rate_Y=known_rate_Y.astype(float)

#missing value data set
alldata_withmissing=data[data['loan_int_rate'].isnull()]     
train_x=alldata_withmissing.drop(columns=['loan_int_rate']).astype(float)
train_y=alldata_withmissing['loan_int_rate'].astype(float)

#use random forest to predict missing value 
fc=RandomForestClassifier()
fc.fit(known_rata_X,known_rate_Y.astype('int'))
pr=fc.predict(train_x)*0.01
data=data.multiply(0.01)

#transfer data type from ndarray to series
pr = pd.Series(pr.tolist()) 

#data without missing value
known_rate_data = data[(data['loan_int_rate'].notna())] 
known_rate_data=known_rate_data.reset_index(drop=True)

#data imputated missing value
alldata_withmissing=data[data['loan_int_rate'].isnull()]  
alldata_withmissing = alldata_withmissing.reset_index(drop=True)
alldata_withmissing['loan_int_rate']=pr

#concat all data to create a complete data set
data=pd.concat([known_rate_data,alldata_withmissing])
data.isnull().any()  

#detect outliers
data.columns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,20))#设置画布的尺寸
plt.subplot(5,2,1)
data.boxplot(column="person_age",ax=None,fontsize=None,grid=True,sym="*" )
plt.subplot(5,2,2)
data.boxplot(column="person_income",ax=None,fontsize=None,grid=True,sym="*")
plt.subplot(5,2,3)
data.boxplot(column="person_emp_length",ax=None,fontsize=None,grid=True,sym="*")
plt.subplot(5,2,4)
data.boxplot(column="loan_amnt",ax=None,fontsize=None,grid=True,sym="*")
plt.subplot(5,2,5)
data.boxplot(column="loan_int_rate",ax=None,fontsize=None,grid=True,sym="*")
plt.subplot(5,2,6)
data.boxplot(column="loan_percent_income",ax=None,fontsize=None,grid=True,sym="*")
plt.subplot(5,2,7)
data.boxplot(column="cb_person_cred_hist_length",ax=None,fontsize=None,grid=True,sym="*")

###  deal with outliers
#age: delete age >90
data = data.drop(data[data['person_age']>100].index)
#income: delete income>100000
data = data.drop(data[data['person_income']>1000000].index)
#lenth:delete lenth>30
E=data.person_emp_length.quantile(0.75)+1.5*(data.person_emp_length.quantile(0.75)
                                             -data.person_emp_length.quantile(0.25))
data = data.drop(data[data['person_emp_length']>E].index)
#amount: 
E=data.loan_amnt.quantile(0.75)+1.5*(data.loan_amnt.quantile(0.75)
                                     -data.loan_amnt.quantile(0.25))
data = data.drop(data[data['loan_amnt']>E].index)
#rate: delete >22.5
E=data.loan_int_rate.quantile(0.75)+1.5*(data.loan_int_rate.quantile(0.75)
                                         -data.loan_int_rate.quantile(0.25))
data = data.drop(data[data['loan_int_rate']>E].index)
#loan percent:delete >0.8
E=data.loan_percent_income.quantile(0.75)+1.5*(
    data.loan_percent_income.quantile(0.75)-data.loan_percent_income.quantile(0.25))
data = data.drop(data[data['loan_percent_income']>E].index)
#hist lenth:delete >20
E=data.cb_person_cred_hist_length.quantile(0.75)+1.5*(
    data.cb_person_cred_hist_length.quantile(0.75)
    -data.cb_person_cred_hist_length.quantile(0.25))
data = data.drop(data[data['cb_person_cred_hist_length']>E].index)


data.to_csv(r'data_after_preprocess.csv', index=False)


