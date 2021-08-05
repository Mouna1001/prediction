#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis

# In[1]:


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set(font_scale=1.5, style="whitegrid")
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('all_pdfs.csv', parse_dates=['Date'])


# In[3]:


df.head(2)


# Replace None value 

# In[4]:


df.Fill_rate = df.Fill_rate.replace({None:np.nan})


# In[5]:


df.Fill_rate = df.Fill_rate.astype(float)
df.Fill_rate.fillna(df.Fill_rate.mean(), inplace=True)


# In[6]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(corr, annot=True, ax=ax)
ax.set_title('Correlation of features')
plt.show()


# In[6]:


date_feats = ['year']


# In[7]:


for feat in date_feats:
    df[feat] = getattr(df['Date'].dt, feat)


# In[8]:


df['day_name'] = df.Date.dt.day_name()
df['month_name'] = df.Date.dt.month_name()


# 

# In[9]:


def get_data(df, name='all', date_sort=True, day_filter='all', month_filter='all', year_filter='all'):
    if name == 'all':
        data = df.copy()
    else:
        data = df[df['Name'].isin(name) ]
    if date_sort:
        data = data.sort_values(by='Date')
    if day_filter != 'all':
        data = data[data['day_name'].isin(day_filter) ]
    if month_filter != 'all':
        data = data[data['month_name'].isin(month_filter) ]
    if year_filter != 'all':
        data = data[data['year'].isin(year_filter) ]
    data.drop_duplicates(inplace=True)
        
    return data.reset_index(drop=True)


# ### FIltering data for the first three months in 2021

# In[14]:


data_viz = get_data(df, name=['Idriss 1er', 'Oued El Makhazine'], day_filter='all',
                   month_filter=['January', 'February', 'March'], year_filter=[2021])


# In[15]:


data_viz


# In[16]:


def visualize_data(data, x='Date', y='Fill_rate', hue='Name'):
    fig, ax = plt.subplots(figsize=(10,8))
    sns.lineplot(data=data, x=x, y=y, hue=hue, ax=ax)
    if x=='Date':
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
    ax.set_title('{} vs {}'.format(x,y))
    plt.show()


# In[17]:


visualize_data(data_viz)


# From the chart above, we notice a regular month pattern for the fill_rate, where the values peak at the beginning of the month, then a sharp fall occurs.

# In[15]:


visualize_data(data_viz, x='Reserve')


# From the chart above we can see that the there's a postive linear correlation between Reserve and Fill_rate. <p>Also the reserve values for the <q>Oued El Makhazine </q> barrage have smaller standard deviation and are less spread than the reserve values for the <q>Idriss 1er</q> barrage. </p>

# 

# ### FIltering data  all the months in 2020 on only Weekends 

# In[16]:


data_viz1 = get_data(df, name=['Idriss 1er', 'Oued El Makhazine'], day_filter=['Friday','Saturday','Sunday'],
                   month_filter='all', year_filter=[2020])


# In[17]:


visualize_data(data_viz1)


# From the plot above, we see that for weekends, there's a regular monthly pattern where there's a peak at the beginning weekends of the month and a fall at the last weekends of the month.

# In[18]:


visualize_data(data_viz1, x='Reserve')


# ## Predictive Modelling

# In[18]:


df.head(3)


# In[10]:


df.sort_values(by=['Date'], inplace=True)


# Label Encode Name and month_name column

# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


barrage_encoder = LabelEncoder()
month_encoder = LabelEncoder()


# In[13]:


barrage_encoder.fit(df.Name)
month_encoder.fit(df.month_name)


# In[14]:


df['Name'] = barrage_encoder.transform(df['Name'])
df['month_name'] = month_encoder.transform(df['month_name'])


# In[15]:


after_2021 = df[df['Date']>=datetime(2021, 1, 1)]
before_2021 = df[df['Date']<datetime(2021, 1, 1)]


# In[16]:


features = ['Name', 'Reserve', 'month_name',]
TARGET = ['Fill_rate']


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# In[18]:


X = before_2021[features]
y = before_2021[TARGET]


# In[19]:


random_state=2021


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)


# In[21]:


rfc = RandomForestRegressor(random_state=random_state)


# In[22]:


rfc.fit(x_train, y_train)


# In[23]:


def EvaluateModel(model, test_features, test_targets):    
    '''
    function to evaluate the performance of a model on the test data
    
    args:
        model: The model to evaluate
        
        test_features: the independent feature values to be evaluated
        
        test_targets: the dependent target values to be evaluated
        
    returns:
        eval_df: A pandas datafame that contains metric results 
    '''
    
    eval_df = pd.DataFrame()
    preds = model.predict(test_features)
    
    eval_df['R2_score'] = pd.Series(r2_score(test_targets, preds), name='r2')
    eval_df['Mean_absolute_error'] = pd.Series(mean_absolute_error(test_targets, preds), name='mae')
    eval_df['Mean_squared_error'] = pd.Series(mean_squared_error(test_targets, preds), name='mse')
    eval_df['Root_mean_squared_error'] = pd.Series(np.sqrt(mean_squared_error(test_targets, preds)), name='rmse')
    
    return eval_df


# In[24]:


train_eval_df = EvaluateModel(rfc, x_train, y_train)


# In[25]:


train_eval_df


# 

# In[26]:


test_eval_df = EvaluateModel(rfc, x_test, y_test)


# In[27]:


test_eval_df


# 

# In[28]:


x_val = after_2021[features]
y_val = after_2021[TARGET]


# In[29]:


val_eval_df = EvaluateModel(rfc, x_val, y_val)


# In[30]:


val_eval_df


# 

# Pickle Useful Files

# In[32]:


import pickle


# In[33]:


modelname = 'rfc_model.pkl'
pickle.dump(rfc, open(modelname, 'wb'))


# In[34]:


barrage_encoder_name = 'barrage_encoder.pkl'
pickle.dump(barrage_encoder, open(barrage_encoder_name, 'wb'))


# In[35]:


month_encoder_name = 'month_encoder.pkl'
pickle.dump(month_encoder, open(month_encoder_name, 'wb'))


# 
