#!/usr/bin/env python
# coding: utf-8

# # Medical Frauds PartB

# In[1]:


import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import copy
from collections import Counter
from numpy import where
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import random
import itertools
#import sagemaker, boto3, os
from interpret.glassbox import ExplainableBoostingClassifier 
import xgboost as xgb
from interpret.perf import ROC  
from imblearn import over_sampling
from imblearn import under_sampling
from imblearn.pipeline import Pipeline
import os              # for directory and file manipulation
import numpy as np     # for basic array manipulation
import pandas as pd    # for dataframe manipulation
import datetime        # for timestamp

# for model eval
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_squared_error, roc_auc_score, roc_curve

# S3 bucket
#bucket = sagemaker.Session().default_bucket()
#prefix = "sagemaker-xgboost-fraud-prediction"

# global constants 
ROUND = 3              # generally, insane precision is not needed 
SEED = 12345           # seed for better reproducibility

# set global random seed for better reproducibility
np.random.seed(SEED)
SEED = 1234
seed = 1234
NTHREAD = 4

import warnings
warnings.filterwarnings('ignore')

#! aws s3 ls {bucket}/{prefix}/data --recursive


# # Load Datasets

# In[2]:


tic = time.time()
path = r'/Users/alex/Desktop/Master/BA_Practicum_6217_10/Project/dataset/PartB' 
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0, encoding="ISO-8859-1")
    li.append(df)

partB = pd.concat(li, axis=0, ignore_index=True)
toc = time.time()
print(toc - tic)


# In[44]:


get_ipython().run_line_magic('cd', '/Users/alex/Desktop/Master/BA_Practicum_6217_10/Project/dataset')
# Load LEIE dataset
leie = pd.read_csv("LEIE.csv")
leie_plus = pd.read_excel("leie_nppes_matches.xlsx")


# In[37]:


# Load Reinstatement LEIE dataset(2020 - 2021)
path = r'/Users/alex/Desktop/Master/BA_Practicum_6217_10/Project/dataset/REIN' # use your path
all_REIN = glob.glob(path + "/*.csv")

li = []

for filename in all_REIN:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

rein_df = pd.concat(li, axis=0, ignore_index=True)


# # 1. Data Cleaning & Exploration

# In[5]:


partB.head(3)


# In[6]:


list(partB.columns)


# In[5]:


# Include only the attributes used for data cleaning process
partB = partB[["Rndrng_NPI", "Rndrng_Prvdr_Type", "HCPCS_Cd", 
               "Place_Of_Srvc", "Tot_Benes", "Tot_Srvcs", "Tot_Bene_Day_Srvcs", 
               "Avg_Sbmtd_Chrg", 'Avg_Mdcr_Alowd_Amt', "Avg_Mdcr_Pymt_Amt", 'Avg_Mdcr_Stdzd_Amt']]
partB.columns


# In[8]:


# Brief information on the partB dataset
partB.info()


# In[6]:


# Check the invalid NPI
# outliers can be invlid NPI 
plt.figure(figsize=(12,3))

plt.subplot(121)
partB["Rndrng_NPI"].hist();

plt.subplot(122)
sns.boxplot('Rndrng_NPI', data=partB)
plt.title('Rndrng_NPI', fontsize=13);


# * No invlid NPIs are detected

# ### Missing Value Detection

# In[7]:


# Detecting missing values
partB.isnull().sum(axis=0)


# * Only gender column has null values

# In[8]:


# List up the unique NPIs whose gender data values are Null
NPI_missGndr = list(partB.loc[partB["Rndrng_Prvdr_Gndr"].isnull()]["Rndrng_NPI"].unique())

# Find the gender data from the other rows that have the equal NPI matching to the elements of the list above
nullGndr_df = partB.loc[partB["Rndrng_NPI"].isin(NPI_missGndr)]
partB["Rndrng_Prvdr_Gndr"].isnull().sum() == nullGndr_df["Rndrng_Prvdr_Gndr"].isnull().sum()


# * Every row matching to the NPIs with Null gender values has missing gender values
# * Remove the corresponding instances from the whole dataset

# In[ ]:


# Drop the "Rndrng_Prvdr_Gndr" variable to preserve all the data
partB = partB.drop("Rndrng_Prvdr_Gndr", axis = 1)


# ### Remove Non-Procedure HCPCS

# In[10]:


get_ipython().run_line_magic('cd', '/Users/alex/Desktop/Master/BA_Practicum_6217_10/Project/dataset')
# Load the Oct 2019 Drug Average Sales Price excel file
asp = pd.read_excel("asp_2019Oct.xls", usecols=[0], skiprows=[0,1,2,3,4,5,6,7,8])
asp.head(3)


# In[11]:


# Drop the rows that have the HCPCS codes recorded in the ASP excel file
drop_idx = partB[partB["HCPCS_Cd"].isin(asp.iloc[:,0])].index
partB = partB.drop(drop_idx)


# In[13]:


# Rename the columns 
partB.rename(columns = {"Rndrng_NPI":"NPI", "Rndrng_Prvdr_Type":"Type"}, inplace = True)


# ### Labeling

# In[46]:


# Rename NPI, DATE, TYPE columns of the LEIE Plus dataset to combine with the other two LEIE datasets 
leie_plus.rename(columns = {"NPPES_NPI":"NPI", "EXCLUDED_DATE":"EXCLDATE", "EXCLUDED_TYPE":"EXCLTYPE"}, inplace = True)


# In[48]:


# Combine the three LEIE datasets
leie_comb = pd.concat([leie[["NPI", "EXCLTYPE", "EXCLDATE"]], 
                       rein_df[["NPI", "EXCLTYPE", "EXCLDATE"]],
                       leie_plus[["NPI", "EXCLTYPE", "EXCLDATE"]]], 
                      ignore_index = True)


# In[55]:


# Convert the integer type of EXCLDATE to date format
leie_comb['EXCLDATE'] = pd.to_datetime(leie_comb['EXCLDATE'], format = '%Y%m%d')

# Convert the upper-case letters to lower case
leie_comb['EXCLTYPE'] = leie_comb[['EXCLTYPE']].applymap(str.lower)


# In[64]:


# Check the Exclusion types (OIG Acts)
leie_comb["EXCLTYPE"].value_counts()


# In[65]:


# Make a list with the OIG rules that correspond to fradulent behaviors
label_list = ["1128a1", "1128a2", "1128a3", "1128a4", 
              "1128b1", "1128b6", "1128b7", "1128b2", 
              "1128b4", "1128b5", "1128b8", "1128b15", 
              "1128b16"]

# Filter the column "NPI" and the rows matching to valid NPI, fraud-related exclusion types, and exclusion date before 2020
end_date = "2019-12-31"
leie_new = leie_comb["NPI"][(leie_comb["EXCLTYPE"].isin(label_list)) 
                       & (leie_comb["NPI"] != 0 ) 
                       & (leie_comb["EXCLDATE"] <= end_date)]


# In[66]:


# Label each row with fraud = 1/ non-fraud = 0
fraud_npi_list = leie_new.tolist()
partB_new['Fraud'] = np.where(partB_new['NPI'].isin(fraud_npi_list), 1, 0)


# In[ ]:


# Ratio of the two classes
partB["Fraud"].value_counts()


# ### Under-sampling the majority group to reduce the size of the dataset

# In[ ]:


partB_copy = partB.copy()


# In[ ]:


# Assign X and y features
feature = list(partB_copy.columns)

feature.remove("Fraud")

target = "Fraud"


# In[ ]:


undersample = under_sampling.RandomUnderSampler(sampling_strategy=0.1)
X, y = undersample.fit_resample(partB_copy[feature], partB_copy[[target]])


# In[ ]:


under_partB = pd.concat([X, y], axis = 1)
under_partB.head()


# In[ ]:


plt.figure(figsize=(14, 6))
plt.subplot(121)

# Plot the bar graph for the original train
Y_fraud = list(partB_copy[target].value_counts())
Labels = ["0", "1"]

plot = plt.bar(Labels, Y_fraud, 
               width = 0.4,
              color = "blue",
              alpha = 0.6)
 
# Add the data value on head of the bar
for value in plot:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom', 
             fontsize = 13)

# Add labels and title
plt.title("Original", fontsize = 20)
plt.xlabel("Label", fontsize = 18)
plt.ylabel("Frequency", fontsize = 18)


plt.subplot(122)

# Plot the bar graph for the over-sampled train
Y_fraud_under = list(under_partB[target].value_counts())

plot_under = plt.bar(Labels, Y_fraud_under, 
               width = 0.4,
              color = "orange",
              alpha = 0.6)
 
# Add the data value on head of the bar
for value in plot_under:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom', 
             fontsize = 13)

# Add labels and title
plt.title("Under-Sampling", fontsize = 20)
plt.xlabel("Label", fontsize = 18)

# Display the graph on the screen
plt.show()


# In[ ]:


# Drop the HCPCS_Cd feature
under_partB = under_partB.drop("HCPCS_Cd", axis = 1)
under_partB.head()


# In[70]:


# Export the cleaned dataset to a csv file

#under_partB.to_csv("/Users/alex/Desktop/Master/BA_Practicum_6217_10/Project/dataset/partB_new4.csv", index=False, header = True)


# ### -------------------------------

# In[2]:


# import the cleaned dataset 

#%cd /Users/alex/Desktop/Master/BA_Practicum_6217_10/Project/dataset
partB = pd.read_csv("partB_new4.csv")


# ### --------------------------------

# In[3]:


partB.head()


# In[4]:


partB.info()


# ### Ratio of Fraud and Non-fraud classes 

# In[5]:


fig = plt.figure(figsize=(8, 6))

# Plot the bar graph
Y_fraud = list(partB["Fraud"].value_counts())
X_fraud = ["0", "1"]

plot = plt.bar(X_fraud, Y_fraud, 
               width = 0.4,
              color = "orange",
              alpha = 0.6)
 
# Add the data value on head of the bar
for value in plot:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom', 
             fontsize = 13)
 
# Add labels and title
plt.title("Labels", fontsize = 20)
plt.xlabel("Label", fontsize = 18)
plt.ylabel("Count", fontsize = 18)
 
# Display the graph on the screen
plt.show()


# In[6]:


ratio = partB["Fraud"].value_counts(normalize = True) 
print("Non-Fraud: ", round(ratio[0]*100, 2), "%")
print("Fraud: ", round(ratio[1]*100, 2), "%")


# ### One-Hot Encoding

# In[7]:


partB.Type.unique().tolist()


# In[8]:


# One-Hot Encoding 

# Convert the Fraud variable to object datatype
partB["Fraud"] = partB["Fraud"].astype(object)

# Encoding
encoded_partB = pd.get_dummies(partB, drop_first = True)

# Rename some of the changed variable names
encoded_partB.rename(columns = {"Gender_M":"Gender", "Fraud_1":"Fraud", "Place_Of_Srvc_O":"Place_Of_Srvc"}, inplace = True)


# In[9]:


# Feature size of the encoded dataset 

fig, ax = plt.subplots(figsize=(6,7))

xs = ["Before", "After"]
ys = [len(partB.columns), len(encoded_partB.columns)]
ax.plot(xs, ys, "bo-")

for x,y in zip(xs, ys):
    label = "{:d}".format(y)
    plt.annotate(label, 
                 (x,y), 
                 textcoords="offset points", 
                 xytext=(25,4), 
                 ha='left',
                fontsize = 20) 
    
ax.set_xticklabels(xs, fontsize=18)

plt.show()


# ### Data Spliting

# In[10]:


# Assign X and y features

X_var = list(encoded_partB.columns)

for var in ["NPI", "Fraud"]:
    X_var.remove(var)

y_var = "Fraud"


# In[11]:


# Split the whole dataset into train and test dataset
# Using a stratified random sampling so that the Fraud-class (1) data are evenly split into train & test sets
x_train, x_test, y_train, y_test = train_test_split(encoded_partB[X_var], 
                                                    encoded_partB[y_var], 
                                                    test_size=0.2, 
                                                    stratify=encoded_partB["Fraud"])

# Also concatenate the split x & y dataframes 
tr_df = pd.concat([x_train, y_train], axis = 1)
te_df = pd.concat([x_test, y_test], axis = 1)


# In[12]:


# Calculate the odds ratio of Fraud & Non-fraud labels for train & test sets

train_0 = len(tr_df[tr_df["Fraud"] == 0])
train_1 = len(tr_df[tr_df["Fraud"] == 1])

test_0 = len(te_df[te_df["Fraud"] == 0])
test_1 = len(te_df[te_df["Fraud"] == 1])

split_df = pd.DataFrame({"x_axis":["Train", "Test"], 
                         "Ratio":[train_1/train_0, test_1/test_0]})


# In[13]:


fig, ax = plt.subplots(figsize=(8,6))

# Plot the bar graph
plot = plt.bar(split_df["x_axis"], split_df["Ratio"],
               width = 0.4,
              color = "pink",
              alpha = 0.6)
 
# Add the data value on head of the bar
for value in plot:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%.5f' % height, ha='center', va='bottom', 
             fontsize = 13)
 
# Add labels and title
plt.title("Odds Ratio of Fraud", fontsize = 20)
plt.xlabel("Set", fontsize = 18)
plt.ylabel("Count", fontsize = 18)

ax.set_xticklabels(split_df["x_axis"], fontsize=18)

# Display the graph on the screen
plt.show()


# ### Histogram 

# In[14]:


cont_features = partB.columns.tolist()
for var in ["NPI", "Type", 'Place_Of_Srvc', "Fraud"]:
    cont_features.remove(var)


# In[15]:


# Histograms

_ = tr_df[cont_features].hist(bins=60, figsize=(30, 30))


# ### Heat map for correlations

# In[16]:


# Check Multicollinearity 
plt.figure(figsize = (20,20))

corr = tr_df[cont_features + [y_var]].corr()
_ = sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, annot = True)


# ### creating interaction features

# ### box plots

# In[17]:


# in order to draw the box plots of all the features in the same y-axis, scale the variables
scaled_tr = pd.DataFrame(scale(tr_df[cont_features]), columns = cont_features)

# box plots for the continous variables 
plt.figure(figsize=(16, 10))

scaled_tr.boxplot()
plt.xticks(rotation=45)
plt.show()


# ### Outlier Detection

# In[18]:


# Outlier = |z-score| > 3
def detect_outlier(var_list):
    
    threshold=3
    outliers_idx = {}
    outliers_data = {}
    
    
    for col in var_list:
        
        mean1 = np.mean(tr_df[col])
        std1 =np.std(tr_df[col])

        z_score = np.abs((tr_df[col] - mean1)/std1)
        outliers_idx[col] =  z_score.index[z_score > threshold].tolist()
        outliers_data[col] =  z_score[z_score > threshold].tolist()
       
    return outliers_idx, outliers_data


# In[19]:


outliers_idx_dict, outliers_data_dict = detect_outlier(cont_features)


# In[20]:


# Outliers of the outliers
outlier_df = pd.DataFrame({key:pd.Series(value) for key, value in outliers_data_dict.items()})

# To draw the box plots of all the features in the same y-axis, scale the variables
scaled_outlier = pd.DataFrame(scale(outlier_df[cont_features]), columns = cont_features)

# Plot
plt.figure(figsize=(16, 10))

scaled_outlier.boxplot()
plt.xticks(rotation=45)
plt.show()


# In[21]:


# Find the unique indeces of the detected outliers

outliers_list = outliers_idx_dict.get("Tot_Benes")
for i in np.arange(1, len(cont_features)):
    outliers_list.extend(outliers_idx_dict.get(cont_features[i]))
    
outlier_uniq_idx = np.unique(outliers_list).tolist()


# delete the ourlier rows from the train set

orgl_len = len(tr_df)

tr_df = tr_df.drop(index = outlier_uniq_idx, axis = 0)

deleted_rows = orgl_len - len(tr_df)
print("the number of deleted rows: %d" % deleted_rows)


# In[22]:


# Box plots for Outlier-free train set

# To draw the box plots of all the features in the same y-axis, scale the variables
scaled_tr_outlierFree = pd.DataFrame(scale(tr_df[cont_features]), columns = cont_features)

# box plots for the continous variables 
plt.figure(figsize=(16, 10))

scaled_tr_outlierFree.boxplot()
plt.xticks(rotation=45)
plt.show()


# * data are extremely skewed to the right 

# ### -----------------SMOTE---------------------

# In[23]:


tr_df.head()


# ### Scatter Plot of The Imbalanced Data

# In[24]:


# scatter plot of the imbalanced data
# x-axis = "Tot_Benes_mean"
# y-axis = "Tot_Bene_Day_Srvcs_mean"

plt.figure(figsize=(10, 10))

counter = Counter(tr_df[y_var])
for label, _ in counter.items():
    row_ix = where(tr_df[y_var] == label)[0]
    plt.scatter(tr_df[X_var].iloc[row_ix, 3], tr_df[X_var].iloc[row_ix, 4], label=str(label))

plt.title("Trainset", fontsize = 18)
plt.xlabel("Avg_Sbmtd_Chrg", fontsize = 15)
plt.ylabel("Avg_Mdcr_Alowd_Amt", fontsize = 15)
plt.legend()
plt.show()


# ### Over-Sampling the imbalnaced data through SMOTE

# In[25]:


# transform the dataset
oversample = over_sampling.SMOTE()
tr_X, tr_y = oversample.fit_resample(tr_df[X_var], tr_df[y_var])


# In[26]:


plt.figure(figsize=(14, 6))
plt.subplot(121)

# Plot the bar graph for the original train
Y_fraud = list(y_train.value_counts())
Labels = ["0", "1"]

plot = plt.bar(Labels, Y_fraud, 
               width = 0.4,
              color = "blue",
              alpha = 0.6)
 
# Add the data value on head of the bar
for value in plot:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom', 
             fontsize = 13)

# Add labels and title
plt.title("Original", fontsize = 20)
plt.xlabel("Label", fontsize = 18)
plt.ylabel("Frequency", fontsize = 18)


plt.subplot(122)

# Plot the bar graph for the over-sampled train
Y_fraud_smote = list(tr_y.value_counts())

plot_smote = plt.bar(Labels, Y_fraud_smote, 
               width = 0.4,
              color = "orange",
              alpha = 0.6)
 
# Add the data value on head of the bar
for value in plot_smote:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom', 
             fontsize = 13)

# Add labels and title
plt.title("SMOTE", fontsize = 20)
plt.xlabel("Label", fontsize = 18)

# Display the graph on the screen
plt.show()


# In[27]:


# scatter plot of the data transformed by SMOTE 
# x-axis = "Tot_Benes_mean"
# y-axis = "Tot_Bene_Day_Srvcs_mean"

plt.figure(figsize=(10, 10))

counter = Counter(tr_y)
for label, _ in counter.items():
    row_ix = where(tr_y == label)[0]
    plt.scatter(tr_X.iloc[row_ix, 3], tr_X.iloc[row_ix, 4], label=str(label))

plt.title("Oversampled Trainset", fontsize = 18)
plt.xlabel("Avg_Sbmtd_Chrg", fontsize = 15)
plt.ylabel("Avg_Mdcr_Alowd_Amt", fontsize = 15)
plt.legend()
plt.show()


# ### ------------------SMOTE & Under-Sampling-------------------------

# In[28]:


# SMOTE the Fraud data increasing its size to a 10% of the number of Non-fraud data 
# and then reduce the number of Non-fraud data to have 50 percent more than the Fraud data using random under-sampling
over = over_sampling.SMOTE(sampling_strategy=0.3)
under = under_sampling.RandomUnderSampler(sampling_strategy=0.5)

steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)


# In[29]:


# transform the dataset
#tr_X, tr_y = pipeline.fit_resample(x_train, y_train)

tr_X, tr_y = pipeline.fit_resample(tr_df[X_var], tr_df[y_var])


# In[30]:


plt.figure(figsize=(14, 6))
plt.subplot(121)

# Plot the bar graph for the original train
Y_fraud = list(tr_df[y_var].value_counts())
Labels = ["0", "1"]

plot = plt.bar(Labels, Y_fraud, 
               width = 0.4,
              color = "blue",
              alpha = 0.6)
 
# Add the data value on head of the bar
for value in plot:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom', 
             fontsize = 13)

# Add labels and title
plt.title("Original", fontsize = 20)
plt.xlabel("Label", fontsize = 18)
plt.ylabel("Frequency", fontsize = 18)


# After SMOTE & Random Undersampling transformation 
plt.subplot(122)

# Plot the bar graph for SMOTE & Undersampled train
Y_fraud = list(tr_y.value_counts())
X_fraud = ["0", "1"]

plot = plt.bar(X_fraud, Y_fraud, 
               width = 0.4,
              color = "orange",
              alpha = 0.6)
 
# Add the data value on head of the bar
for value in plot:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom', 
             fontsize = 13)
 
# Add labels and title
plt.title("SMOTE & Under-Sampling", fontsize = 20)
plt.xlabel("Label", fontsize = 18)
 
# Display the graph on the screen
plt.show()


# In[31]:


# scatter plot of the data transformed by SMOTE 
# x-axis = "Tot_Benes_mean"
# y-axis = "Tot_Bene_Day_Srvcs_mean"

plt.figure(figsize=(10, 10))

counter = Counter(tr_y)
for label, _ in counter.items():
    row_ix = where(tr_y == label)[0]
    plt.scatter(tr_X.iloc[row_ix, 3], tr_X.iloc[row_ix, 4], label=str(label))
plt.legend()
plt.show()


# ## Model Training 

# ### Data Partitioning (Train & Valid)

# In[32]:


trans_tr_df = pd.concat([tr_X, tr_y], axis = 1)


# In[33]:


# Export the cleaned dataset to a csv file
#trans_tr_df.to_csv("/Users/yuwenluo/Desktop/IntegrityM/IntegrityM_v3/train.csv", index=False, header = True)


# In[34]:


# Export the cleaned dataset to a csv file
#te_df.to_csv("/Users/yuwenluo/Desktop/IntegrityM/IntegrityM_v3/test.csv", index=False, header = True)


# In[35]:


# Split train and validation sets 
np.random.seed(SEED)

ratio = 0.7 # split train & validation sets with 7:3 ratio 

split = np.random.rand(len(trans_tr_df)) < ratio # define indices of 70% corresponding to the training set

train = trans_tr_df[split]
valid = trans_tr_df[~split]

# summarize split
print('Train data rows = %d, columns = %d' % (train.shape[0], train.shape[1]))
print('Validation data rows = %d, columns = %d' % (valid.shape[0], valid.shape[1]))


# In[36]:


# Using a stratified random sampling so that the Fraud-class (1) data are evenly split into train & test sets
x_train = train[X_var]
x_valid = valid[X_var]
y_train = train[y_var]
y_valid = valid[y_var]


# ## ---------Random Forest_v3--------

# In[ ]:


# tarin: tr_df
# test: te_df
# train of train: train
# valid of train: valid


# In[136]:


SEED                    = 12345   # global random seed for better reproducibility
GLM_SELECTION_THRESHOLD = 0.001   # threshold above which a GLM coefficient is considered "selected"
MONO_THRESHOLD          = 6       # lower == more monotone constraints
TRUE_POSITIVE_AMOUNT    = 0       # revenue for rejecting a defaulting customer
TRUE_NEGATIVE_AMOUNT    = 20000   # revenue for accepting a paying customer, ~ customer LTV
FALSE_POSITIVE_AMOUNT   = -20000  # revenue for rejecting a paying customer, ~ -customer LTV 
FALSE_NEGATIVE_AMOUNT   = -100000 # revenue for accepting a defaulting customer, ~ -mean(LIMIT_BAL)


# In[137]:


import auto_ph                                                    # simple module for training and eval
import h2o                                                        # import h2o python bindings to java server
import numpy as np                                                # array, vector, matrix calculations
import operator                                                   # for sorting dictionaries
import pandas as pd                                               # DataFrame handling
import time                                                       # for timers

import matplotlib.pyplot as plt      # plotting
pd.options.display.max_columns = 999 # enable display of all columns in notebook

# enables display of plots in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(SEED)                     # set random seed for better reproducibility

h2o.init(max_mem_size='24G', nthreads=4) # start h2o with plenty of memory and threads
h2o.remove_all()                         # clears h2o memory
h2o.no_progress()                        # turn off h2o progress indicators


# In[138]:


big_tic = time.time()


# In[139]:


# Pearson correlation between inputs and target
# is last column of correlation matrix
corr = pd.DataFrame(tr_df[X_var + [y_var]].corr()[y_var]).iloc[:-1]
corr.columns = ['Pearson Correlation Coefficient']
corr


# ### Elastic Net Forward Step-wise Training for Initial Feature Selection

# In[140]:


# Split train and valid sets 
np.random.seed(SEED)

ratio = 0.7 # split train & validation sets with 7:3 ratio 

split = np.random.rand(len(tr_df)) < ratio # define indices of 70% corresponding to the training set

train1 = tr_df[split]
valid = tr_df[~split]

# summarize split
print('Train data rows = %d, columns = %d' % (train1.shape[0], train1.shape[1]))
print('Validation data rows = %d, columns = %d' % (valid.shape[0], valid.shape[1]))


# In[141]:


# train penalized GLM w/ alpha and lambda grid search
best_glm = auto_ph.glm_grid(X_var, y_var, h2o.H2OFrame(train1),
                            h2o.H2OFrame(valid), SEED)

# output results
print('Best penalized GLM AUC: %.2f' % 
      best_glm.auc(valid=True))

# print selected coefficients
print('Best penalized GLM coefficients:')
for c_name, c_val in sorted(best_glm.coef().items(), key=operator.itemgetter(1)):
    if abs(c_val) > GLM_SELECTION_THRESHOLD:
        print('%s %s' % (str(c_name + ':').ljust(25), c_val))


# In[142]:


# collect regularization paths from dict in DataFrame
reg_path_dict = best_glm.getGLMRegularizationPath(best_glm)
reg_path_frame = pd.DataFrame(columns=reg_path_dict['coefficients'][0].keys())
for i in range(0, len(reg_path_dict['coefficients'])): 
    reg_path_frame = reg_path_frame.append(reg_path_dict['coefficients'][i], 
                                           ignore_index=True)

###########################################    
# establish benchmark feature selection:  #
#           glm_selected                  #
# used frequently in further calculations #
###########################################

glm_selected = list(reg_path_frame.iloc[-1, :][reg_path_frame.iloc[-1, :] > GLM_SELECTION_THRESHOLD].index)

# plot regularization paths
fig, ax_ = plt.subplots(figsize=(8, 6))
_ = reg_path_frame[glm_selected].plot(kind='line', ax=ax_, title='Penalized GLM Regularization Paths',
                                      colormap='gnuplot')
_ = ax_.set_xlabel('Iteration')
_ = ax_.set_ylabel('Coefficient Value')
_ = ax_.axhline(c='k', lw=1, xmin=0.045, xmax=0.955)
_ = plt.legend(bbox_to_anchor=(1.05, 0),
               loc=3, 
               borderaxespad=0.)


# In[143]:


glm_selected


# In[147]:


best_glm


# ### Train on Ramdom Forest Model_v1

# #### Model on Whole

# In[38]:


from sklearn.ensemble import RandomForestClassifier

# Fit Random Forest Model
rf_model = RandomForestClassifier(n_estimators=20)

#rf_model.fit(train, valid)
rf_model.fit(x_train, y_train)


# In[39]:


# Predict on Valid set
y_pred_rf = rf_model.predict(x_valid)
rf_binary_pred = [round(value) for value in y_pred_rf]


# In[40]:


#%pip install sklearn
import sklearn


# In[41]:


# Confusion Matrix
rf_confusion_matrix = sklearn.metrics.confusion_matrix(y_valid, rf_binary_pred)
rf_confusion_matrix = pd.DataFrame(rf_confusion_matrix)
rf_confusion_matrix.rename(index = {0:"non_fraud", 1:"fraud"},columns = {0:"non_fraud", 1:"fraud"}, inplace = True)
rf_confusion_matrix


# In[42]:


# Estimate quality of classifier
rf_precision = sklearn.metrics.precision_score(y_valid, rf_binary_pred, average='macro')
rf_roc_auc = sklearn.metrics.roc_auc_score(y_valid, rf_binary_pred)
rf_accuracy = sklearn.metrics.accuracy_score(y_valid, rf_binary_pred)
rf_f1 = sklearn.metrics.f1_score(y_valid, rf_binary_pred)
rf_logloss = sklearn.metrics.log_loss(y_valid, rf_binary_pred)
rf_mse = sklearn.metrics.mean_squared_error(y_valid, rf_binary_pred)
#print('rf_precision', rf_precision, 'rf_accuracy',rf_accuracy,'rf_roc_auc',rf_roc_auc,'rf_f1',rf_f1,'rf_logloss',rf_logloss,'mse',rf_mse)
rf_result1 = [rf_precision,rf_roc_auc,rf_accuracy,rf_f1,rf_logloss,rf_mse]
rf_result1 = pd.DataFrame(rf_result1)
rf_result1.rename(index = {0:"rf_precision", 2:"rf_accuracy", 1:"rf_roc_auc",3:"rf_f1",4:"rf_logloss",5:"rf_mse"},columns = {0:"Metrix"}, inplace = True)
rf_result1


# #### Model on Selected Features

# In[70]:


features = ['Avg_Mdcr_Alowd_Amt',
 'Avg_Mdcr_Stdzd_Amt',
 'Type_Allergy/Immunology',
 'Type_Ambulance Service Provider',
 'Type_Ambulance Service Supplier',
 'Type_Anesthesiology',
 'Type_Audiologist',
 'Type_Audiologist (billing independently)',
 'Type_CRNA',
 'Type_Cardiology',
 'Type_Cardiovascular Disease (Cardiology)',
 'Type_Chiropractic',
 'Type_Clinical Laboratory',
 'Type_Clinical Psychologist',
 'Type_Emergency Medicine',
 'Type_Family Practice',
 'Type_General Practice',
 'Type_General Surgery',
 'Type_Geriatric Medicine',
 'Type_Hematology-Oncology',
 'Type_Hematology/Oncology',
 'Type_Independent Diagnostic Testing Facility',
 'Type_Independent Diagnostic Testing Facility (IDTF)',
 'Type_Internal Medicine',
 'Type_Interventional Pain Management',
 'Type_Medical Oncology',
 'Type_Multispecialty Clinic/Group Practice',
 'Type_Neurology',
 'Type_Neuropsychiatry',
 'Type_Neurosurgery',
 'Type_Nuclear Medicine',
 'Type_Obstetrics & Gynecology',
 'Type_Obstetrics/Gynecology',
 'Type_Osteopathic Manipulative Medicine',
 'Type_Otolaryngology',
 'Type_Pain Management',
 'Type_Pediatric Medicine',
 'Type_Physical Medicine and Rehabilitation',
 'Type_Plastic and Reconstructive Surgery',
 'Type_Podiatry',
 'Type_Portable X-ray',
 'Type_Psychiatry',
 'Type_Radiation Oncology',
 'Type_Sports Medicine',
 'Type_Surgical Oncology',
 'Type_Urology',
 'Place_Of_Srvc']


# In[44]:


# Using a stratified random sampling so that the Fraud-class (1) data are evenly split into train & test sets
x_train1 = train[features]
x_valid1 = valid[features]
y_train = train[y_var]
y_valid = valid[y_var]


# In[45]:


from sklearn.ensemble import RandomForestClassifier

# Fit Random Forest Model
rf_model1 = RandomForestClassifier(n_estimators=20)

#rf_model.fit(train, valid)
rf_model1.fit(x_train1, y_train)


# In[46]:


# Predict on Valid set
y_pred_rf1 = rf_model1.predict(x_valid1)
rf_binary_pred1 = [round(value) for value in y_pred_rf1]


# In[47]:


# Confusion Matrix
rf_confusion_matrix1 = sklearn.metrics.confusion_matrix(y_valid, rf_binary_pred1)
rf_confusion_matrix1 = pd.DataFrame(rf_confusion_matrix)
rf_confusion_matrix1.rename(index = {0:"non_fraud", 1:"fraud"},columns = {0:"non_fraud", 1:"fraud"}, inplace = True)
rf_confusion_matrix1


# In[48]:


# Estimate quality of classifier
rf_precision1 = sklearn.metrics.precision_score(y_valid, rf_binary_pred1, average='macro')
rf_roc_auc1 = sklearn.metrics.roc_auc_score(y_valid, rf_binary_pred1)
rf_accuracy1 = sklearn.metrics.accuracy_score(y_valid, rf_binary_pred1)
rf_f11 = sklearn.metrics.f1_score(y_valid, rf_binary_pred1)
rf_logloss1 = sklearn.metrics.log_loss(y_valid, rf_binary_pred1)
rf_mse1 = sklearn.metrics.mean_squared_error(y_valid, rf_binary_pred1)
#print('rf_precision', rf_precision, 'rf_accuracy',rf_accuracy,'rf_roc_auc',rf_roc_auc,'rf_f1',rf_f1,'rf_logloss',rf_logloss,'mse',rf_mse)
rf_result2 = [rf_precision1,rf_roc_auc1,rf_accuracy1,rf_f11,rf_logloss1,rf_mse1]
rf_result2 = pd.DataFrame(rf_result1)
rf_result2.rename(index = {0:"rf_precision", 2:"rf_accuracy", 1:"rf_roc_auc",3:"rf_f1",4:"rf_logloss",5:"rf_mse"},columns = {0:"Metrix"}, inplace = True)
rf_result2


# #### Model on Valid Set

# In[49]:


from sklearn.ensemble import RandomForestClassifier

# Fit Random Forest Model
rf_model1 = RandomForestClassifier(n_estimators=20)

#rf_model.fit(train, valid)
rf_model1.fit(x_valid1, y_valid)


# In[50]:


# Predict on Valid set
y_pred_rf1 = rf_model1.predict(x_valid1)
rf_binary_pred1 = [round(value) for value in y_pred_rf1]


# In[51]:


# Confusion Matrix
rf_confusion_matrix2 = sklearn.metrics.confusion_matrix(y_valid, rf_binary_pred1)
rf_confusion_matrix2 = pd.DataFrame(rf_confusion_matrix)
rf_confusion_matrix2.rename(index = {0:"non_fraud", 1:"fraud"},columns = {0:"non_fraud", 1:"fraud"}, inplace = True)
rf_confusion_matrix2


# In[53]:


# Estimate quality of classifier
rf_precision2 = sklearn.metrics.precision_score(y_valid, rf_binary_pred1, average='macro')
rf_roc_auc2 = sklearn.metrics.roc_auc_score(y_valid, rf_binary_pred1)
rf_accuracy2 = sklearn.metrics.accuracy_score(y_valid, rf_binary_pred1)
rf_f12 = sklearn.metrics.f1_score(y_valid, rf_binary_pred1)
rf_logloss2 = sklearn.metrics.log_loss(y_valid, rf_binary_pred1)
rf_mse1 = sklearn.metrics.mean_squared_error(y_valid, rf_binary_pred1)
#print('rf_precision', rf_precision, 'rf_accuracy',rf_accuracy,'rf_roc_auc',rf_roc_auc,'rf_f1',rf_f1,'rf_logloss',rf_logloss,'mse',rf_mse)
rf_result2 = [rf_precision1,rf_roc_auc1,rf_accuracy1,rf_f11,rf_logloss1,rf_mse1]
rf_result2 = pd.DataFrame(rf_result1)
rf_result2.rename(index = {0:"rf_precision", 2:"rf_accuracy", 1:"rf_roc_auc",3:"rf_f1",4:"rf_logloss",5:"rf_mse"},columns = {0:"Matrix"}, inplace = True)
rf_result2


# In[56]:


from sklearn import tree


# In[96]:


fn=X_var
cn=y_var

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,20), dpi=800)
tree.plot_tree(rf_model.estimators_[0],
               max_depth=(3),
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('rf_individualtree.png')


# #### Model on Test set

# In[102]:


tr_df.count()


# In[103]:


te_df.count()


# In[98]:


train = tr_df
test = te_df # you can change your testing part
x_train = train[features]
y_train = train[y_var]
x_test = test[features]
y_test = test[y_var]


# In[ ]:





# In[99]:


from sklearn.ensemble import RandomForestClassifier

# Fit Random Forest Model
rf_model3 = RandomForestClassifier(n_estimators=20)

#rf_model.fit(train, valid)
rf_model3.fit(x_train, y_train)


# In[100]:


# Predict on Valid set
y_pred_rf3 = rf_model3.predict(x_test)
rf_binary_pred3 = [round(value) for value in y_pred_rf3]


# In[101]:


# Confusion Matrix
rf_confusion_matrix3 = sklearn.metrics.confusion_matrix(y_test, rf_binary_pred3)
rf_confusion_matrix3 = pd.DataFrame(rf_confusion_matrix3)
rf_confusion_matrix3.rename(index = {0:"non_fraud", 1:"fraud"},columns = {0:"non_fraud", 1:"fraud"}, inplace = True)
rf_confusion_matrix3


# In[106]:


# Estimate quality of classifier
rf_precision3 = sklearn.metrics.precision_score(y_test, rf_binary_pred3, average='macro')
rf_roc_auc3 = sklearn.metrics.roc_auc_score(y_test, rf_binary_pred3)
rf_accuracy3 = sklearn.metrics.accuracy_score(y_test, rf_binary_pred3)
rf_f13 = sklearn.metrics.f1_score(y_test, rf_binary_pred3)
rf_logloss3 = sklearn.metrics.log_loss(y_test, rf_binary_pred3)
rf_mse3 = sklearn.metrics.mean_squared_error(y_test, rf_binary_pred3)
#print('rf_precision', rf_precision, 'rf_accuracy',rf_accuracy,'rf_roc_auc',rf_roc_auc,'rf_f1',rf_f1,'rf_logloss',rf_logloss,'mse',rf_mse)
rf_result3 = [rf_precision3,rf_roc_auc3,rf_accuracy3,rf_f13,rf_logloss3,rf_mse3]
rf_result3 = pd.DataFrame(rf_result3)
rf_result3.rename(index = {0:"rf_precision", 2:"rf_accuracy", 1:"rf_roc_auc",3:"rf_f1",4:"rf_logloss",5:"rf_mse"},columns = {0:"Matrix"}, inplace = True)
rf_result3


# In[ ]:





# In[ ]:





# In[ ]:




