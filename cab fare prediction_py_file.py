#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date
import os
import seaborn as sns


# In[29]:


os.chdir('C://Users//DELL//Desktop//assignments for adwisor//cab  fare prediction')


# In[30]:


train_data=pd.read_csv('C://Users//DELL//Desktop//assignments for adwisor//cab  fare prediction//train_cab.csv')


# In[31]:


test_data=pd.read_csv('C://Users//DELL//Desktop//assignments for adwisor//cab  fare prediction//test.csv')


# In[32]:


test_data.head()


# In[33]:


train_data.head()


# In[34]:


#we will be checking each column of data for noise, missing values and outliers


# In[35]:


##checking the  passenger count we can have max 6 passengers if we consider a SUV passeneger count having more than 
##6 will be noise


# In[36]:


train_data['passenger_count'].describe()
## here we can see that the max is showing 5345 which is not possible 


# In[37]:


test_data.passenger_count.describe()


# In[38]:


train_data=train_data.drop(train_data[train_data['passenger_count']>6].index, axis=0)


# In[39]:


train_data=train_data.drop(train_data[train_data['passenger_count']==0].index, axis=0)


# In[40]:


train_data.passenger_count.describe()
## now we have successfully removed the noise values.


# In[41]:


##checking for missing values and removing it
train_data.passenger_count.isnull().sum()


# In[42]:


train_data=train_data.drop(train_data[train_data['passenger_count'].isnull()].index, axis=0)


# In[43]:


train_data.passenger_count.isnull().sum()
train_data = train_data.drop(train_data[train_data["passenger_count"] == 1.3 ].index, axis=0)
train_data = train_data.drop(train_data[train_data["passenger_count"] == 0.12 ].index, axis=0)
## we removed noise and missing values from passenger_count


# In[44]:


## checking fare amount
train_data.fare_amount=pd.to_numeric(train_data['fare_amount'],errors='coerce')


# In[45]:


train_data.fare_amount.describe()


# In[46]:


train_data.fare_amount.sort_values(ascending=False)


# In[47]:


##from above we can see that fare amount cannot be this high so these are the outliers which are to be removed


# In[48]:


from collections import Counter
Counter(train_data.fare_amount<0)


# In[49]:


train_data=train_data.drop(train_data[train_data['fare_amount']<0].index, axis=0)


# In[50]:


train_data['fare_amount'].min()


# In[51]:


train_data=train_data.drop(train_data[train_data['fare_amount']<1].index, axis=0)


# In[52]:


## cab fare can go maximun upto to 2000 but it cannot go upto 50000 or more so here i am considering a threshold value of 2000


# In[53]:


train_data=train_data.drop(train_data[train_data['fare_amount']>2000].index, axis=0)


# In[54]:


train_data.fare_amount.describe()


# In[55]:


train_data.fare_amount.isnull().sum()


# In[56]:


## removing nan values
train_data=train_data.drop(train_data[train_data['fare_amount'].isnull()].index, axis=0)


# In[57]:


train_data.fare_amount.isnull().sum()
## fare amount is cleared from missing values and noise


# In[58]:


## now converting time data
train_data.info()


# In[59]:


train_data=train_data.drop(train_data[train_data['pickup_datetime'].isnull()].index, axis=0)


# In[60]:


test_data=test_data.drop(test_data[test_data['pickup_datetime'].isnull()].index, axis=0)


# In[61]:


train_data['pickup_datetime']=pd.to_datetime(train_data['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC',errors='coerce')


# In[62]:


test_data['pickup_datetime']=pd.to_datetime(test_data['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC',errors='coerce')


# In[63]:


train_data['year']=train_data['pickup_datetime'].dt.year
train_data['month']=train_data['pickup_datetime'].dt.month
train_data['date']=train_data['pickup_datetime'].dt.day
train_data['day']=train_data['pickup_datetime'].dt.dayofweek
train_data['hour']=train_data['pickup_datetime'].dt.hour
train_data['min']=train_data['pickup_datetime'].dt.minute


# In[64]:


train_data.drop('pickup_datetime', axis=1, inplace=True)


# In[65]:


train_data.head()


# In[66]:


test_data['year']=test_data['pickup_datetime'].dt.year
test_data['month']=test_data['pickup_datetime'].dt.month
test_data['date']=test_data['pickup_datetime'].dt.day
test_data['day']=test_data['pickup_datetime'].dt.dayofweek
test_data['hour']=test_data['pickup_datetime'].dt.hour
test_data['min']=test_data['pickup_datetime'].dt.minute


# In[67]:


test_data.drop('pickup_datetime', axis=1, inplace=True)


# In[68]:


## now checking locations 
## the lattitude should be between -90 to 90
## longitude should be between -180 to 180


# In[69]:


##train data pick up
train_data=train_data.drop(train_data[train_data['pickup_latitude']<-90].index, axis=0)
train_data=train_data.drop(train_data[train_data['pickup_latitude']>90].index, axis=0)
train_data=train_data.drop(train_data[train_data['pickup_longitude']<-180].index, axis=0)
train_data=train_data.drop(train_data[train_data['pickup_longitude']>180].index, axis=0)


# In[70]:


##train data drop off
train_data=train_data.drop(train_data[train_data['dropoff_latitude']<-90].index, axis=0)
train_data=train_data.drop(train_data[train_data['dropoff_latitude']>90].index, axis=0)
train_data=train_data.drop(train_data[train_data['dropoff_longitude']<-180].index, axis=0)
train_data=train_data.drop(train_data[train_data['dropoff_longitude']>180].index, axis=0)


# In[71]:


##test data pick up
test_data=test_data.drop(test_data[test_data['pickup_latitude']<-90].index, axis=0)
test_data=test_data.drop(test_data[test_data['pickup_latitude']>90].index, axis=0)
test_data=test_data.drop(test_data[test_data['pickup_longitude']<-180].index, axis=0)
test_data=test_data.drop(test_data[test_data['pickup_longitude']>180].index, axis=0)


# In[72]:


##test data drop off
test_data=test_data.drop(test_data[test_data['dropoff_latitude']<-90].index, axis=0)
test_data=test_data.drop(test_data[test_data['dropoff_latitude']>90].index, axis=0)
test_data=test_data.drop(test_data[test_data['dropoff_longitude']<-180].index, axis=0)
test_data=test_data.drop(test_data[test_data['dropoff_longitude']>180].index, axis=0)


# In[73]:


## now we need to calculate the distance between the locations
train_data.isnull().sum()


# In[74]:


train_data=train_data.drop(train_data[train_data['year'].isnull()].index, axis=0)
train_data=train_data.drop(train_data[train_data['month'].isnull()].index, axis=0)
train_data=train_data.drop(train_data[train_data['date'].isnull()].index, axis=0)
train_data=train_data.drop(train_data[train_data['day'].isnull()].index, axis=0)
train_data=train_data.drop(train_data[train_data['hour'].isnull()].index, axis=0)
train_data=train_data.drop(train_data[train_data['min'].isnull()].index, axis=0)


# In[75]:


test_data.isnull().sum()


# In[76]:


#we would need to use haversine function to get the distance bet two point
from math import radians, cos, sin, asin, sqrt

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km


# In[77]:


train_data['distance']=train_data[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine, axis=1)


# In[78]:


test_data['distance']=test_data[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine, axis=1)


# In[79]:


train_data.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'], axis=1, inplace=True)


# In[80]:


train_data.head()


# In[81]:


test_data.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'], axis=1, inplace=True)


# In[82]:


##we have successfully cleaned the data and now we will be moving to the visualisation part


# In[84]:


sns.countplot(data=train_data,x='passenger_count')


# In[85]:


sns.lmplot(data=train_data,x='passenger_count', y='fare_amount')
plt.xlabel('no. of passenger')
plt.ylabel('Fare')
plt.show()
## here we can see that max fare is earned by single or doulble occupancy


# In[86]:


sns.lineplot(data=train_data, x='day',y='fare_amount')
plt.xlabel('time of day')
plt.ylabel('fare')
plt.show()


# In[87]:


sns.lineplot(data=train_data, x='month',y='fare_amount')
plt.xlabel('Month')
plt.ylabel('fare')
plt.show()


# In[88]:


sns.lineplot(data=train_data, x='year',y='fare_amount')
plt.xlabel('year')
plt.ylabel('fare')
plt.show()


# In[89]:


sns.relplot(data=train_data,x='distance', y='fare_amount')
plt.xlabel('distance')
plt.ylabel('fare')
plt.show()


# In[90]:


sns.lineplot(data=train_data, x='passenger_count', y='fare_amount')
plt.xlabel('No. of passengers')
plt.ylabel('fare')
plt.show()
## from here we can see that earnings were high when 2 passengers were travelling 


# In[91]:


sns.distplot(train_data['fare_amount'], bins='auto',color='red')
plt.xlabel('fare')
plt.ylabel('density')
plt.show()


# In[92]:


sns.distplot(train_data['distance'], color='green')


# In[93]:


train=train_data.copy()


# In[95]:


## alos tried to use standarization but the results were not as expected


# In[96]:


## using normalization
train['fare_amount']=np.log1p(train['fare_amount'])
sns.distplot(train['fare_amount'])


# In[97]:


## using normalization
train['distance']=np.log1p(train_data['distance'])
sns.distplot(train['distance'])


# In[98]:


x_n=train.drop('fare_amount', axis=1)
y_n=train['fare_amount']


# In[99]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_n,y_n, test_size=0.2, random_state=44)


# In[100]:


## using linear regression
from sklearn.linear_model import LinearRegression
LR=LinearRegression()


# In[101]:


LR.fit(x_train,y_train)


# In[102]:


predictions_LR_N=LR.predict(x_test)


# In[103]:


##calculating RMSE for test data
from sklearn.metrics import mean_squared_error, r2_score
RMSE_test_LR_n= np.sqrt(mean_squared_error(y_test, predictions_LR_N))


# In[104]:


RMSE_test_LR_n


# In[105]:


print(r2_score(y_test, predictions_LR_N))


# In[106]:


## using decision tree
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)


# In[107]:


predict_dtr=dtr.predict(x_test)


# In[108]:


rsme_dtr=np.sqrt(mean_squared_error(y_test, predict_dtr))


# In[109]:


rsme_dtr


# In[110]:


print(r2_score(y_test, predict_dtr))


# In[111]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)


# In[112]:


predict_rfr=rfr.predict(x_test)


# In[113]:


print(r2_score(y_test,predict_rfr))


# In[114]:


rmse=np.sqrt(mean_squared_error(y_test, predict_rfr))


# In[115]:


rmse


# In[116]:


from sklearn.linear_model import Ridge
r=Ridge(alpha=0.05)
r.fit(x_train,y_train,)


# In[117]:


predicted_r=r.predict(x_test)


# In[118]:


print(r2_score(y_test,predicted_r))


# In[119]:


from sklearn.linear_model import Lasso
la=Lasso()


# In[120]:


la.fit(x_train,y_train)


# In[121]:


prediction=la.predict(x_test)


# In[122]:


rsme_r=np.sqrt(mean_squared_error(y_test, prediction))


# In[123]:


rsme_r


# In[124]:


print(r2_score(y_test,prediction))


# In[125]:


## using gradient boosting 
from sklearn.ensemble import GradientBoostingRegressor
grad=GradientBoostingRegressor()
grad.fit(x_train,y_train)


# In[126]:


prediction_grad=grad.predict(x_test)


# In[127]:


rsme_grad=np.sqrt(mean_squared_error(y_test,prediction_grad))


# In[128]:


print(rsme_grad)


# In[129]:


print(r2_score(y_test,prediction_grad))


# In[130]:


## using hyper parameter tuning for randon forest
from sklearn.model_selection import GridSearchCV
param_grid = {  'bootstrap': [True], 'max_depth': [5, 10, None], 'max_features': ['auto', 'log2'], 'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]}


# In[131]:


rfr = RandomForestRegressor(random_state = 1)


# In[132]:


grid_search=GridSearchCV(estimator = rfr, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 0,return_train_score=True)


# In[133]:


grid_search.fit(x_train,y_train)


# In[134]:


prediction_grid=grid_search.predict(x_test)


# In[135]:


rmse=np.sqrt(mean_squared_error(y_test, prediction_grid))


# In[136]:


print(rmse)


# In[137]:


print(r2_score(y_test, prediction_grid))


# In[138]:


## using grid search on gradient boost
from sklearn.model_selection import GridSearchCV
param_grid=param_grid = { 'max_depth': [5, 10, 2], 'max_features': ['auto', 'log2'],
                         'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15],'learning_rate':[0.01,0.05,0.1,0.2]}


# In[139]:


grid_grad=GridSearchCV(estimator=grad,param_grid=param_grid, n_jobs=1,return_train_score=True,cv=3,verbose=0)


# In[140]:


grid_grad.fit(x_train,y_train)


# In[141]:


prediction_grid_grad=grid_grad.predict(x_test)


# In[142]:


rsme_grid_grad=np.sqrt(mean_squared_error(y_test,prediction_grid_grad))


# In[143]:


print(rsme_grid_grad)


# In[144]:


print(r2_score(y_test,prediction_grid_grad))


# In[145]:


grid_grad.best_params_


# In[146]:


grid_grad.best_estimator_


# In[147]:


grid_grad.best_score_


# In[148]:


##From above we can see that  Gradient boosting  regressor with grid search has best r2_score and RSME


# In[149]:


##Now getting the fare amount for test data


# In[150]:


test_data.head()


# In[151]:


final_prediction=grad.predict(test_data)


# In[152]:


test_data['predicted fare']=final_prediction


# In[153]:


test_data.head()


# In[154]:


test_data.to_csv('Predicted test data')


# In[155]:


import pickle
pickle_out=open('predictor.pkl', mode='wb')
pickle.dump(grad, pickle_out)
pickle_out.close()


# In[156]:


get_ipython().run_cell_magic('writefile', 'cab_fare_predictor.py', 'import pickle\nimport streamlit as st\nimport pandas as pd\nimport numpy as np\n\npickle_in=open(\'predictor.pkl\', mode=\'rb\')\npredictor=pickle.load(pickle_in)\n\ndef run():       \n    add_selectbox=st.sidebar.selectbox(\n      "How would like to get the predictions?",\n      (\'Realtime\',\'Batch\'))\n    st.sidebar.info(\'This application helps to predict cab fare\')\n    if add_selectbox==\'Realtime\':\n        passenger=st.number_input(\'passengers\', min_value=1, max_value=6, value=1)\n        hour=st.number_input(\'hour\', min_value=0, max_value=23, value=0)\n        mins=st.number_input(\'mins\', min_value=0, max_value=59, value=0)\n        day=st.number_input(\'day\', min_value=0, max_value=6, value=0)\n        distance=st.number_input(\'distance\',min_value=0, max_value=1000, value=0)\n        year=st.number_input(\'year\', min_value=2009, max_value=2015, value=2009)\n        date=st.number_input(\'date\', min_value=1, max_value=31, value=1)\n        output=\'\'\n        input_dict={\'passenger\':passenger, \'hour\':hour,\'mins\':mins,\'day\':day,\n                \'distance\':distance,\'year\':year,\'date\':date}\n        input_df=pd.DataFrame([input_dict])\n        if st.button("predict"):\n            output=predictor(input_df=input_df)\n            output=float(output)\n    if add_selectbox==\'Batch\':\n        file_upload=st.file_uploader("upload the csv file", type=[\'csv\'])\n        if file_upload is not None:\n            data=pd.read_csv(file_upload)\n            predictions=predict_model(estimator=predictor, data=data)\n            st.write(predictions)\nrun()')


# In[157]:


get_ipython().system('pip install -q streamlit')


# In[ ]:




