#######################################################################################
#    Pulled some of the preprocessing and analysis from kaggle                        #
#    https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-zillow-prize  #
#                                                                                     #
#######################################################################################


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

from subprocess import check_output

train_df = pd.read_csv("D:/Documents/kaggle/Zillow/train_2016/train_2016.csv", parse_dates=["transactiondate"])
train_df.shape

#plt.figure(figsize=(8,6))
#plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
#plt.xlabel('index', fontsize=12)
#plt.ylabel('logerror', fontsize=12)
#plt.show()

ulimit = np.percentile(train_df.logerror.values, 99)
llimit = np.percentile(train_df.logerror.values, 1)
train_df['logerror'].ix[train_df['logerror']>ulimit] = ulimit
train_df['logerror'].ix[train_df['logerror']<llimit] = llimit

#plt.figure(figsize=(12,8))
#sns.distplot(train_df.logerror.values, bins=50, kde=False)
#plt.xlabel('logerror', fontsize=12)
#plt.show()

prop_df = pd.read_csv("D:/Documents/kaggle/Zillow/properties_2016/properties_2016.csv")
#prop_df.head()

#missing_df = prop_df.isnull().sum(axis=0).reset_index()
#missing_df.columns = ['column_name', 'missing_count']
#missing_df = missing_df.ix[missing_df['missing_count']>0]
#missing_df = missing_df.sort_values(by='missing_count')

#ind = np.arange(missing_df.shape[0])
#width = 0.9
#fig, ax = plt.subplots(figsize=(12,18))
#rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
#ax.set_yticks(ind)
#ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
#ax.set_xlabel("Count of missing values")
#ax.set_title("Number of missing values in each column")
#plt.show()

#plt.figure(figsize=(12,12))
#sns.jointplot(x=prop_df.latitude.values, y=prop_df.longitude.values, size=10)
#plt.ylabel('Longitude', fontsize=12)
#plt.xlabel('Latitude', fontsize=12)
#plt.show()

train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
#train_df.head()

pd.options.display.max_rows = 65

dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
#dtype_df

dtype_df.groupby("Column Type").aggregate('count').reset_index()

missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] / train_df.shape[0]
missing_df.ix[missing_df['missing_ratio']>0.999]
