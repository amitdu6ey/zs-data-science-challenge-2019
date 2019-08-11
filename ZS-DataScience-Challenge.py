#!/usr/bin/env python
# coding: utf-8

# In[24]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


from fastai.imports import *
from fastai.structured import *
import numpy as np
import pandas as pd
from pandas_summary import DataFrameSummary
import sklearn.model_selection
from IPython.display import display
import math
import random
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import collections


# In[26]:


PATH = "data/Cristano_Ronaldo_Final_v1/"


# In[27]:


get_ipython().system('ls {PATH}')


# # Required Functions

# In[28]:


def imae(x,y):
    return 1/(1+(abs(x-y)).mean())


# In[29]:


def print_score(m):
    res = [
        imae(m.predict(X_train.drop(['Unnamed: 0'],axis=1)), y_train),
        imae(m.predict(X_valid.drop(['Unnamed: 0'],axis=1)), y_valid),
        m.score(X_train.drop(['Unnamed: 0'],axis=1), y_train), 
        m.score(X_valid.drop(['Unnamed: 0'],axis=1), y_valid),
          ]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[30]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


# # Data Pre-processing

# In[31]:


df_i = pd.read_csv(f'{PATH}sample_submission.csv')
df_i.shot_id_number = df_i.shot_id_number-1
df_i=df_i.drop(['is_goal'], axis=1)


# In[32]:


df_raw = pd.read_csv(f'{PATH}data.csv', low_memory=False, parse_dates=['date_of_game'])


# In[33]:


df_raw.is_goal.value_counts()


# In[34]:


display_all(df_raw.T)


# In[35]:


df_raw['date_of_game'] = pd.to_datetime(df_raw.date_of_game)
df_raw=df_raw.sort_values('date_of_game')
display_all(df_raw.T)


# In[72]:


"""
I tried this but it lead to worse r^2 score :/ so its commented now
lst = [
    'is_goal',
    'knockout_match',
    'game_season',
    'shot_basics',
    'team_name',
    'home/away',
    'lat/lng',
    'type_of_combined_shot',
    'match_id',
    'team_id',
    'knockout_match.1',
]
for col in lst:
    df_raw[col].interpolate(method='nearest',inplace=True)
    """;


# ## train_cats
# It change any columns of strings in a panda's dataframe to a column of
# categorical values. This applies the changes inplace.

# In[37]:


train_cats(df_raw)


# In[38]:


cols = ['knockout_match','match_event_id', 'game_season', 'area_of_shot','shot_basics', 'range_of_shot', 'lat/lng', 'team_name',
       'home/away', 'shot_id_number',  'type_of_shot', 'type_of_combined_shot','match_id','team_id']


# In[39]:


for col in cols:
    df_raw[col] = df_raw[col].astype('category').cat.codes


# In[40]:


df_raw['year'] = df_raw['date_of_game'].dt.year
df_raw['month']=df_raw['date_of_game'].dt.month
df_raw=df_raw.drop(['date_of_game'],axis=1)


# In[41]:


df_raw, __, ___ = proc_df(df_raw)


# ## proc_df
# It takes a data frame df and splits off the response variable, and
# changes the df into an entirely numeric dataframe. For each column of df 
# which is not in skip_flds nor in ignore_flds, na values are replaced by the
# median value of the column.

# In[42]:


df_raw.match_event_id.value_counts()


# In[43]:


df_tst = df_raw[df_raw['Unnamed: 0'].isin(df_i['shot_id_number'])]
df_trn = df_raw[~df_raw['Unnamed: 0'].isin(df_i['shot_id_number'])]


# In[44]:


display_all(df_trn.T)


# In[45]:


df_trn.describe()


# In[46]:


X_train, X_valid, y_train, y_valid  = sklearn.model_selection.train_test_split(df_trn.drop(['is_goal'],axis=1), df_trn['is_goal'], test_size=0.20, random_state=42)


# In[47]:


X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# # Model Selection & Analysis

# ### set_rf_samples(n)
# Changes Scikit learn's random forests to give each tree a random sample of n random rows.

# In[67]:


set_rf_samples(100)


# In[68]:


clf = RandomForestClassifier(n_estimators=1000, max_depth=10, max_features=0.5, min_samples_leaf=5)
clf.fit(X_train.drop(['Unnamed: 0'],axis=1), y_train)
print_score(clf)


# In[50]:


draw_tree(clf.estimators_[0], X_train.drop(['Unnamed: 0'],axis=1), precision=3)


# In[51]:


pred_valid =  clf.predict(X_train.drop(['Unnamed: 0'],axis=1))
collections.Counter(pred_valid)


# In[52]:


preds = np.stack([t.predict(X_valid.drop(['Unnamed: 0'],axis=1)) for t in clf.estimators_])
#preds[:,0], np.mean(preds[:,0]), y_valid[0]


# In[65]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(100)]);


# # Feature Importance

# In[54]:


fi = rf_feat_importance(clf, X_valid.drop(['Unnamed: 0'],axis=1))
fi[:]


# # Final model

# In[55]:


clf.fit(df_trn.drop(['Unnamed: 0','is_goal'],axis=1), df_trn['is_goal'])


# In[56]:


df_tst = df_tst.drop(['is_goal'],axis=1)


# In[57]:


df_trn.T


# In[58]:


pred_tst = clf.predict(df_tst.drop(['Unnamed: 0'],axis=1))


# In[59]:


pred_tst


# In[60]:


df_ans=pd.DataFrame()
df_ans['id']=df_tst['Unnamed: 0']+1


# In[61]:


df_ans['prediction']=pred_tst.astype('int')


# In[62]:


df_ans=df_ans.sort_values('id')


# In[63]:


#df_ans.prediction = df_ans.prediction.astype('int')
df_ans.to_csv(f'{PATH}/amit_dubey_190199_code_5.csv', index=False)


# In[64]:


df_ans.T

