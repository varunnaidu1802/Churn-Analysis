#!/usr/bin/env python
# coding: utf-8



# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 


# In[2]:


churn = pd.read_csv("churn_data.csv")
churn


# In[3]:


cust = pd.read_csv("customer_data.csv")
cust


# In[4]:


internet = pd.read_csv("internet_data.csv")
internet


# In[5]:


data = cust.merge(churn, on='customerID', how='right').merge(internet, on='customerID', how='right')


# In[6]:


data


# In[7]:


data.info()


# In[8]:


data.shape


# In[9]:


data.describe()


# In[10]:


data['TotalCharges'].describe()


# In[11]:


get_ipython().system('pip install numpy')


# In[12]:


data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data


# In[13]:


value = (data['TotalCharges']/data['MonthlyCharges']).median()*data['MonthlyCharges']


# In[14]:


value


# In[15]:


data['TotalCharges'] = value.where(data['TotalCharges'] == np.nan, other =data['TotalCharges'])
data


# In[16]:


fig, axs = plt.subplots(1,2, figsize = (15,5))
plt1 = sns.countplot(data['Churn'], ax = axs[0])

pie_churn = pd.DataFrame(data['Churn'].value_counts())
pie_churn.plot.pie( subplots=True,labels = pie_churn.index.values, autopct='%1.1f%%', figsize = (15,5), startangle= 50, ax = axs[1])


plt.gca().set_aspect('equal')

plt.show()


# In[17]:


import math 


# In[18]:


def percentage_stacked_plot(columns_to_plot, super_title):
    
    
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot)/2)

    fig = plt.figure(figsize=(22, 7 * number_of_rows)) 
    fig.suptitle(super_title, fontsize=22,  y=.95)
 
    for index, column in enumerate(columns_to_plot, 1):
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        prop_by_independent = pd.crosstab(data[column], data['Churn']).apply(lambda x: x/x.sum()*100, axis=1)

        prop_by_independent.plot(kind='bar', ax=ax, stacked=True,
                                 rot=0, color=['Blue','Yellow'])

        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                  title='Churn', fancybox=True)

        ax.set_title('Proportion of observations by ' + column,
                     fontsize=16, loc='left')

        ax.tick_params(rotation='auto')

        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)


# In[19]:


demographic_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

percentage_stacked_plot(demographic_columns, 'Demographic Information')


# In[20]:


account_columns = ['Contract', 'PaperlessBilling', 'PaymentMethod']


percentage_stacked_plot(account_columns, 'Customer Account Information')


# In[21]:


def histogram_plots(columns_to_plot, super_title):
    
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot)/2)

    fig = plt.figure(figsize=(12, 5 * number_of_rows)) 
    fig.suptitle(super_title, fontsize=22,  y=.95)
 

 
    for index, column in enumerate(columns_to_plot, 1):

        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        data[data['Churn']=='No'][column].plot(kind='hist', ax=ax, density=True, 
                                                       alpha=0.5, color='Blue', label='No')
        data[data['Churn']=='Yes'][column].plot(kind='hist', ax=ax, density=True,
                                                        alpha=0.5, color='Yellow', label='Yes')
        
        ax.legend(loc="upper right", bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),
                  title='Churn', fancybox=True)

        ax.set_title('Distribution of ' + column + ' by churn',
                     fontsize=16, loc='left')

        ax.tick_params(rotation='auto')

        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)
            

account_columns_numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']

histogram_plots(account_columns_numeric, 'Customer Account Information')


# In[22]:


services_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

percentage_stacked_plot(services_columns, 'Services Information')


# In[23]:


varlist =  ['PhoneService', 'PaperlessBilling', 'Churn', 'Partner', 'Dependents']


def binary_map(x):
    return x.map({'Yes': 1, "No": 0})


data[varlist] = data[varlist].apply(binary_map)


# In[24]:


data


# In[25]:


dummy1 = pd.get_dummies(data[['Contract', 'PaymentMethod', 'gender', 'InternetService']], drop_first=True)

data = pd.concat([data, dummy1], axis=1)

ml = pd.get_dummies(data['MultipleLines'], prefix='MultipleLines')

ml1 = ml.drop(['MultipleLines_No phone service'], 1)

data = pd.concat([data,ml1], axis=1)

os = pd.get_dummies(data['OnlineSecurity'], prefix='OnlineSecurity')
os1 = os.drop(['OnlineSecurity_No internet service'], 1)
data = pd.concat([data,os1], axis=1)

ob = pd.get_dummies(data['OnlineBackup'], prefix='OnlineBackup')
ob1 = ob.drop(['OnlineBackup_No internet service'], 1)
data = pd.concat([data,ob1], axis=1)

dp = pd.get_dummies(data['DeviceProtection'], prefix='DeviceProtection')
dp1 = dp.drop(['DeviceProtection_No internet service'], 1)
data = pd.concat([data,dp1], axis=1)

ts = pd.get_dummies(data['TechSupport'], prefix='TechSupport')
ts1 = ts.drop(['TechSupport_No internet service'], 1)
data = pd.concat([data,ts1], axis=1)

st =pd.get_dummies(data['StreamingTV'], prefix='StreamingTV')
st1 = st.drop(['StreamingTV_No internet service'], 1)
data = pd.concat([data,st1], axis=1)

sm = pd.get_dummies(data['StreamingMovies'], prefix='StreamingMovies')
sm1 = sm.drop(['StreamingMovies_No internet service'], 1)
data = pd.concat([data,sm1], axis=1)

data


# In[26]:


data = data.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies'], 1)


# In[27]:


data


# In[28]:


num_data = data[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]


# In[29]:


num_data.describe(percentiles=[.25, .5, .75, .90, .95, .99])


# In[30]:


data.isnull().sum()


# In[31]:


data = data[~np.isnan(data['TotalCharges'])]


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X = data.drop(['Churn','customerID'], axis=1)

X.head()


# In[34]:


y = data['Churn']

y.head()


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[36]:


from sklearn.preprocessing import StandardScaler


# In[37]:


scaler = StandardScaler()

X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])

X_train.head()


# In[38]:


plt.figure(figsize = (30,15))        
sns.heatmap(data.corr(),annot = True)
plt.show()


# In[39]:


X_test = X_test.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',
                       'StreamingTV_No','StreamingMovies_No'], 1)
X_train = X_train.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',
                         'StreamingTV_No','StreamingMovies_No'], 1)


# In[40]:


plt.figure(figsize = (20,10))
sns.heatmap(X_train.corr(),annot = True)
plt.show()


# In[41]:


import statsmodels.api as sm


# In[42]:


logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[43]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[44]:


from sklearn.feature_selection import RFE


# In[45]:


rfe = RFE(logreg)            
rfe = rfe.fit(X_train, y_train)


# In[46]:


rfe.support_


# In[47]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[48]:


col = X_train.columns[rfe.support_]


# In[49]:


X_train.columns[~rfe.support_]


# In[50]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[51]:


y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[52]:


y_train_pred_final = pd.DataFrame({'Churn':y_train.values, 'Churn_Prob':y_train_pred})
y_train_pred_final['CustID'] = y_train.index
y_train_pred_final.head()


# In[53]:


y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)


y_train_pred_final.head()


# In[54]:


from sklearn import metrics


# In[55]:


confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
print(confusion)


# In[56]:


print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))


# In[57]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[58]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[59]:


X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[60]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[61]:


y_train_pred[:10]


# In[62]:


y_train_pred_final['Churn_Prob'] = y_train_pred


# In[63]:


y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[64]:


print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))


# In[65]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[66]:


confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
confusion


# In[67]:


metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted)


# In[68]:


TP = confusion[1,1] 
TN = confusion[0,0] 
FP = confusion[0,1] 
FN = confusion[1,0]


# In[69]:


TP / float(TP+FN)


# In[70]:


TN / float(TN+FP)


# In[71]:


print(FP/ float(TN+FP))


# In[72]:


print (TP / float(TP+FP))


# In[73]:


print (TN / float(TN+ FN))


# In[74]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[75]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Churn, y_train_pred_final.Churn_Prob, drop_intermediate = False )


# In[76]:


draw_roc(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)


# In[77]:


numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[78]:


cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[79]:


cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[80]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Churn_Prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[81]:


metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.final_predicted)


# In[82]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.final_predicted )
confusion2


# In[83]:


TP = confusion2[1,1] 
TN = confusion2[0,0] 
FP = confusion2[0,1]
FN = confusion2[1,0]


# In[84]:


TP / float(TP+FN)


# In[85]:


TN / float(TN+FP)


# In[86]:


print(FP/ float(TN+FP))


# In[87]:


print (TP / float(TP+FP))


# In[88]:


print (TN / float(TN+ FN))


# In[89]:


confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
confusion


# In[90]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[91]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[92]:


from sklearn.metrics import precision_score, recall_score


# In[93]:


get_ipython().run_line_magic('pinfo', 'precision_score')


# In[94]:


precision_score(y_train_pred_final.Churn, y_train_pred_final.predicted)


# In[95]:


recall_score(y_train_pred_final.Churn, y_train_pred_final.predicted)


# In[96]:


from sklearn.metrics import precision_recall_curve


# In[97]:


y_train_pred_final.Churn, y_train_pred_final.predicted


# In[98]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)


# In[99]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[100]:


X_test[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(X_test[['tenure','MonthlyCharges','TotalCharges']])


# In[101]:


X_test = X_test[col]
X_test.head()


# In[102]:


X_test_sm = sm.add_constant(X_test)


# In[103]:


y_test_pred = res.predict(X_test_sm)


# In[104]:


y_test_pred[:10]


# In[105]:


y_pred_1 = pd.DataFrame(y_test_pred)


# In[106]:


y_pred_1.head()


# In[107]:


y_test_df = pd.DataFrame(y_test)


# In[108]:


y_test_df['CustID'] = y_test_df.index


# In[109]:


y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[110]:


y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[111]:


y_pred_final= y_pred_final.rename(columns={ 0 : 'Churn_Prob'})


# In[113]:


y_pred_final = y_pred_final.reindex(['CustID','Churn','Churn_Prob'], axis=1)


# In[114]:


y_pred_final.head()


# In[115]:


y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.42 else 0)


# In[116]:


y_pred_final.head()


# In[117]:


metrics.accuracy_score(y_pred_final.Churn, y_pred_final.final_predicted)


# In[118]:


confusion2 = metrics.confusion_matrix(y_pred_final.Churn, y_pred_final.final_predicted )
confusion2


# In[119]:


TP = confusion2[1,1] 
TN = confusion2[0,0] 
FP = confusion2[0,1] 
FN = confusion2[1,0] 


# In[120]:


TP / float(TP+FN)


# In[121]:


TN / float(TN+FP)


# In[ ]:




