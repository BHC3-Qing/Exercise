#!/usr/bin/env python
# coding: utf-8

# Step 0. Import Python libraries and main dataset

# In[2]:


import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


from sklearn.model_selection import train_test_split


# In[3]:


DSNY = pd.read_csv('311-DSNY-20151017.csv')
print(DSNY.shape,DSNY['Unique Key'].nunique())


# In[4]:


DSNY_df = DSNY.copy()
DSNY_df.dtypes


# Step 1. Hand-picked categotical variables and valuate them

# In[5]:


dict_df = pd.read_csv('311_SR_Data_Dictionary_2018.csv')
ca_list = list(dict_df['Column Name'])
print(len(ca_list))
rm_list = [0,1,2,8,9,12,13,14,15,16,17,18,19,24,26,27,29,30,31]
for i in sorted(rm_list, reverse = True):
    del ca_list[i]
print(len(ca_list))
ca_list = np.array(ca_list)
ca_list = np.where(ca_list == 'Open_Data_Channel_Type', 'Open Data Channel Type', ca_list)
ca_list


# In[6]:


list(DSNY_df.shape)[0]


# In[7]:


for i in ca_list:
    print(i,DSNY_df[i].isnull().sum()/list(DSNY_df.shape)[0])
    print(DSNY_df[i].value_counts())
    print('***')


# Observations:
# 1. Data quality is good - no duplicates, misspelling, and little missing in key infomation
# 2. Some of the features are completely missing, although they are less important

# Step 2. Check numeric & time variables

# In[8]:


print(DSNY_df['Created Date'].isnull().sum())
print(DSNY_df['Closed Date'].isnull().sum())
print(DSNY_df['Due Date'].isnull().sum())


# In[9]:


DSNY_df['Closed Date'] = DSNY_df['Closed Date'].fillna('12/31/2022')
DSNY_df['Due Date'] = DSNY_df['Due Date'].fillna('12/31/2022')


# In[10]:


print(DSNY_df['Created Date'].isnull().sum())
print(DSNY_df['Closed Date'].isnull().sum())
print(DSNY_df['Due Date'].isnull().sum())


# In[11]:


DSNY_df['Created_Date'] = DSNY_df['Created Date'].str[:10]
DSNY_df['Closed_Date'] = DSNY_df['Closed Date'].str[:10]
DSNY_df['Due_Date'] = DSNY_df['Due Date'].str[:10]
DSNY_df['Created_Year'] = DSNY_df['Created Date'].str[6:10]
DSNY_df['Closed_Year'] = DSNY_df['Closed Date'].str[6:10]
DSNY_df['Due_Year'] = DSNY_df['Due Date'].str[6:10]


# In[12]:


print(DSNY_df['Created_Date'].min(),DSNY_df['Created_Date'].max())
print(DSNY_df['Closed_Date'].min(),DSNY_df['Closed_Date'].max())
print(DSNY_df['Due_Date'].min(),DSNY_df['Due_Date'].max())
print(DSNY_df['Created_Year'].min(),DSNY_df['Created_Year'].max())
print(DSNY_df['Closed_Year'].min(),DSNY_df['Closed_Year'].max())
print(DSNY_df['Due_Year'].min(),DSNY_df['Due_Year'].max())


# In[13]:


DSNY_df[DSNY_df['Closed_Date'].str.contains('3027')]


# In[14]:


DSNY_df[DSNY_df['Closed_Date'].str.contains('1900')]


# In[15]:


DSNY_df_clean = DSNY_df[DSNY_df['Closed_Year'].astype(int)<2022]
DSNY_df_clean = DSNY_df_clean[DSNY_df_clean['Closed_Year'].astype(int)>=2015]
DSNY_df_clean.shape


# In[16]:


DSNY_df_clean['Created_DT'] = pd.to_datetime(DSNY_df_clean['Created_Date'])
DSNY_df_clean['Closed_DT'] = pd.to_datetime(DSNY_df_clean['Closed_Date'])
DSNY_df_clean['Due_DT'] = pd.to_datetime(DSNY_df_clean['Due_Date'])

DSNY_df_clean['Created_Year_Month'] = DSNY_df_clean['Created_Date'].str[6:10]+'-'+DSNY_df_clean['Created_Date'].str[:2]
DSNY_df_clean['Closed_Year_Month'] = DSNY_df_clean['Closed_Date'].str[6:10] +'-'+ DSNY_df_clean['Closed_Date'].str[:2]
DSNY_df_clean['Due_Year_Month'] = DSNY_df_clean['Due_Date'].str[6:10] +'-'+ DSNY_df_clean['Due_Date'].str[:2]
DSNY_df_clean.head()


# In[91]:


# DSNY_df_clean['Created_Year_Month'] = DSNY_df_clean['Created_DT'].dt.year.astype(str) + '-' + DSNY_df_clean['Created_DT'].dt.month.astype(str)
# DSNY_df_clean['Closed_Year_Month'] = DSNY_df_clean['Closed_DT'].dt.year.astype(str) + '-' + DSNY_df_clean['Closed_DT'].dt.month.astype(str)
# DSNY_df_clean['Due_Year_Month'] = DSNY_df_clean['Due_DT'].dt.year.astype(str) + '-' + DSNY_df_clean['Due_DT'].dt.month.astype(str)
# DSNY_df_clean.head()


# In[17]:


Created_DT_Bar = pd.DataFrame(DSNY_df_clean['Created_DT'].value_counts()).reset_index()

fig, ax = plt.subplots(figsize=(12, 8))

ax.bar(Created_DT_Bar['index'],
       Created_DT_Bar['Created_DT'],
       color='orange')

# Set title and labels for axes
ax.set(xlabel="SR Create Date",
       ylabel="Frequency",
       title="Check SR Create Date")


# In[18]:


Closed_DT_Bar = pd.DataFrame(DSNY_df_clean['Closed_DT'].value_counts()).reset_index()

fig, ax = plt.subplots(figsize=(12, 8))

ax.bar(Closed_DT_Bar['index'],
       Closed_DT_Bar['Closed_DT'],
       color='green')

# Set title and labels for axes
ax.set(xlabel="SR Close Date",
       ylabel="Frequency",
       title="Check SR Close Date")


# In[19]:


Closed_DT_Bar2 =  Closed_DT_Bar[Closed_DT_Bar['index']<'2018-01-01']
fig, ax = plt.subplots(figsize=(12, 8))

ax.bar(Closed_DT_Bar2['index'],
       Closed_DT_Bar2['Closed_DT'],
       color='green')

# Set title and labels for axes
ax.set(xlabel="SR Close Date",
       ylabel="Frequency",
       title="Check SR Close Date - Closer Look")


# In[20]:


DSNY_df_clean['Processing_Days'] = (DSNY_df_clean['Closed_DT'] - DSNY_df_clean['Created_DT']).dt.days
print(DSNY_df_clean['Processing_Days'].min(),DSNY_df_clean['Processing_Days'].max(),DSNY_df_clean['Processing_Days'].isnull().sum())


# In[21]:


# n, bins, patches = plt.hist(DSNY_df_clean['Processing_Days'], 50)
fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(DSNY_df_clean['Processing_Days'],100,color='green')

# Set title and labels for axes
ax.set(xlabel="Processing Days",
       ylabel="Frequency",
       title="Processing Days")


# In[22]:


PD_filtered = list(filter(lambda days: (days >= 0) & (days<200), DSNY_df_clean['Processing_Days']))
print(len(PD_filtered),len(PD_filtered)/len(DSNY_df_clean['Processing_Days']))

fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(PD_filtered,100,color='green')

# Set title and labels for axes
ax.set(xlabel="Processing Days",
       ylabel="Frequency",
       title="Processing Days - Closer Look")


# In[23]:


PD_filtered = list(filter(lambda days: (days >= 0) & (days<100), DSNY_df_clean['Processing_Days']))
print(len(PD_filtered),len(PD_filtered)/len(DSNY_df_clean['Processing_Days']))

fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(PD_filtered,100,color='green')

# Set title and labels for axes
ax.set(xlabel="Processing Days",
       ylabel="Frequency",
       title="Processing Days - Even Closer Look")


# In[24]:


fig, ax = plt.subplots(figsize=(12, 12))

ax.scatter(DSNY_df_clean['Latitude'], DSNY_df_clean['Longitude'], s=1,c=(DSNY_df_clean['Processing_Days'])/10, alpha=0.5)

ax.set(title="Latitude*Longitude")


# In[25]:


fig, ax = plt.subplots(figsize=(12, 12))

ax.scatter(DSNY_df_clean['X Coordinate (State Plane)'], DSNY_df_clean['Y Coordinate (State Plane)'], s=1,c=DSNY_df_clean['Processing_Days']/10, alpha=0.5)

ax.set(title="X Coordinate * Y Coordinate")


# In[26]:


DSNY_df_clean['Processing_Days_Flag'] = np.where(DSNY_df_clean['Processing_Days']>7,'red','green')
DSNY_df_clean['Processing_Days_Flag'].value_counts()


# In[27]:


fig, ax = plt.subplots(figsize=(12, 12))

ax.scatter(DSNY_df_clean['X Coordinate (State Plane)'], DSNY_df_clean['Y Coordinate (State Plane)'], s=1,c=DSNY_df_clean['Processing_Days_Flag'], alpha=0.5)

ax.set(title="X Coordinate * Y Coordinate - Processing Time Comparison")


# Observations
# 1. Some cleanse is needed for close date
# 2. Majority of processing time is legit and can be used for modeling
# 3. Geo location seems asscocited with processing time 

# Step 3. Predict processing time
# 
# Hand-picked features for baseline model (easy to interpret): Agency Name, Complaint Type, Location Type, City, Borough, Community Board, Open Data Channel Type

# Step 3.1 Model data prep

# In[28]:


DSNY_df_model = DSNY_df_clean[['Unique Key','Agency Name','Complaint Type','Location Type','City','Borough','Community Board','Open Data Channel Type','Processing_Days']]
DSNY_df_model = DSNY_df_model[(DSNY_df_model['Processing_Days']>=0) & (DSNY_df_model['Processing_Days']<=100)]
print(DSNY_df_model.shape,DSNY_df_model['Processing_Days'].min(),DSNY_df_model['Processing_Days'].max(),list(DSNY_df_model.shape)[0]/list(DSNY_df.shape)[0])


# In[29]:


DSNY_df_model = pd.get_dummies(DSNY_df_model)
DSNY_df_model.head()


# In[32]:


labels = np.array(DSNY_df_model['Processing_Days'])
DSNY_df_model_feature = DSNY_df_model.drop(['Processing_Days','Unique Key'], axis = 1)
feature_list = list(DSNY_df_model_feature.columns)
DSNY_df_model_feature = np.array(DSNY_df_model_feature)


# In[35]:


train_features, test_features, train_labels, test_labels = train_test_split(DSNY_df_model_feature, labels, test_size = 0.30, random_state = 1017)


# In[36]:


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# Step 3.2 Random forest regression

# In[37]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 1017)
rf.fit(train_features, train_labels);


# In[39]:


predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Random Forest Mean Absolute Error:', round(np.mean(errors), 2))


# Step 3.3 Gradient boosting regression

# In[ ]:


params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'squared_error'}
gbr = GradientBoostingRegressor(**params)
gbr.fit(train_features, train_labels)
predictions = gbr.predict(test_features)
errors = abs(predictions - test_labels)
print('DT Mean Absolute Error:', round(np.mean(errors), 2))


# Step 3.4 Decision tree regression

# In[ ]:


dtr = DecisionTreeRegressor(max_depth=5)
dtr.fit(train_features, train_labels)
predictions = dtr.predict(test_features)
errors = abs(predictions - test_labels)
print('DT Mean Absolute Error:', round(np.mean(errors), 2))

