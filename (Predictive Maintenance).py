#!/usr/bin/env python
# coding: utf-8

# # AI Predictive Maintenance for Machines
# 
# ## Introduction
# 
# Artificial intelligence is commonly used in various spheres to automate processes, gather insights, and speed up processes. I used Python to study an interesting use of artificial intelligence in a real-life scenario - how AI can use sensor data from an aircraft to predict which sensor needs maintenance.
# 
# 
# ## Context
# 
# I will be working with aircraft sensor data, obtained from [Github](https://github.com/Samimust/predictive-maintenance/tree/master/data). The text files contain simulated aircraft engine run-to-failure events, operational settings, and 21 sensors measurements, which are provided by Microsoft. It is assumed that the engine progressing degradation pattern is reflected in its sensor measurements.
# 
# 
# ## Use Python to open csv files
# 
# I will use the [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/) and [scikit-learn](https://scikit-learn.org/stable/) libraries to work with our dataset. Pandas is a popular Python library for data science. It offers powerful and flexible data structures to make data manipulation and analysis easier. Scikit-learn is a very useful machine learning library that provides efficient tools for predictive data analysis. Matplotlib is a Python 2D plotting library that we can use to produce high quality data visualization. It is highly usable (as you will soon find out), you can create simple and complex graphs with just a few lines of codes!

# In[1]:


#Importing Libraries
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

from sklearn import model_selection 
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn import metrics 
from sklearn.feature_selection import SelectFromModel, RFECV


# ## Dataset Description

# In[2]:


# loading the training data 
df_train = pd.read_csv(r'[Dataset]_Train_(Maintenance).csv')


# The dataset contains simulated aircraft engine run-to-failure events, operational settings, and 21 sensors measurements. Now, let us actually see those records.

# ## Exploring the training and test data by printing the first 15 rows

# In[16]:


df_train.head(15)


# In[3]:


#loading test data and exploring  
df_test = pd.read_csv(r'[Dataset]_Test_(Maintenance).csv')
df_test.head(15)


# ## Getting statistics about the training and test set by using describe function

# In[4]:


df_train.describe()


# In[5]:


df_test.describe()


# In[2]:


#Check for missing values
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

from sklearn import model_selection 
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn import metrics 
from sklearn.feature_selection import SelectFromModel, RFECV

df_train = pd.read_csv(r'[Dataset]_Module11_Train_(Maintenance).csv')
df_train.isnull().sum()


# ## Data Visualization
# 
# Visualizing the data often gives us an idea about how various features are distributed and thus helps us in the analysis.

# In[17]:


features = []
for col in df_train.columns: 
    features.append(col)


# In[4]:


#plot and compare the standard deviation of input features:
df_train[features].std().plot(kind='bar', figsize=(10,6), title="Features Standard Deviation")


# Seeing correlation between features.

# In[6]:


#For better visualization let's take feat, which is a subset of features
feat= ['s12',
 's7',
 's21',
 's20',
 's6',
 's14',
 's9',
 's13',
 's8',
 's3',
 's17',
 's2',
 's15',
 's4',
 's11',
 'ttf']

import seaborn as sns
cm = np.corrcoef(df_train[feat].values.T)
sns.set(font_scale=1.0)
fig = plt.figure(figsize=(10, 8))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=feat, xticklabels=feat)
plt.title('Features Correlation Heatmap')
plt.show()


# ## Data preparation for linear regression

# In[13]:


print("type of dataset is",type(feat))


# In[18]:


#Let's prepare data for regression model

#These are original features
features_orig = ['setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

#These are features with low or no correlation with regression label
features_lowcr = ['setting3', 's1', 's10', 's18','s19','s16','s5', 'setting1', 'setting2']

#These are features that have correlation with regression label
features_corrl = ['s2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20','s21']

#features is a variable to hold the set of features to experiment with
features = features_orig


# In[ ]:


X_train = df_train[features]
y_train = df_train['ttf']

X_test = df_test[features]
y_test = df_test['ttf']


# ## Helper Functions

# In[20]:


def get_regression_metrics(model, actual, predicted):
    
    """Calculate main regression metrics.
    
    Args:
        model (str): The model name identifier
        actual (series): Contains the test label values
        predicted (series): Contains the predicted values
        
    Returns:
        dataframe: The combined metrics in single dataframe
    
    
    """
    regr_metrics = {
                        'Root Mean Squared Error' : metrics.mean_squared_error(actual, predicted)**0.5,
                        'Mean Absolute Error' : metrics.mean_absolute_error(actual, predicted),
                        'R^2' : metrics.r2_score(actual, predicted),
                        'Explained Variance' : metrics.explained_variance_score(actual, predicted)
                   }

    #return reg_metrics
    df_regr_metrics = pd.DataFrame.from_dict(regr_metrics, orient='index')
    df_regr_metrics.columns = [model]
    return df_regr_metrics


# In[23]:


def plot_features_weights(model, weights, feature_names, weights_type='c'):
    
    """Plot regression coefficients weights or feature importance.
    
    Args:
        model (str): The model name identifier
        weights (array): Contains the regression coefficients weights or feature importance
        feature_names (list): Contains the corresponding features names
        weights_type (str): 'c' for 'coefficients weights', otherwise is 'feature importance'
        
    Returns:
        plot of either regression coefficients weights or feature importance
        
    
    """
    (px, py) = (8, 10) if len(weights) > 30 else (8, 5)
    W = pd.DataFrame({'Weights':weights}, feature_names)
    W.sort_values(by='Weights', ascending=True).plot(kind='barh', color='r', figsize=(px,py))
    label = ' Coefficients' if weights_type =='c' else ' Features Importance'
    plt.xlabel(model + label)
    plt.gca().legend_ = None


# In[22]:


def plot_residual(model, y_train, y_train_pred, y_test, y_test_pred):
    
    """Print the regression residuals.
    
    Args:
        model (str): The model name identifier
        y_train (series): The training labels
        y_train_pred (series): Predictions on training data
        y_test (series): The test labels
        y_test_pred (series): Predictions on test data
        
    Returns:
        Plot of regression residuals
    
    """
    
    plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-50, xmax=400, color='red', lw=2)
    plt.title(model + ' Residuals')
    plt.show()


# ## Linear Regression Model
# 
# Linear regression is a simple model that models the relationship between a dependent variable (output) and one or more independent variables (input).
# 
# ![linear regression](https://static.javatpoint.com/tutorial/machine-learning/images/linear-regression-in-machine-learning.png)
# 
# RMSE
# 
# RMSE stands for root mean squared error. When we are doing predictions using our machine learning models, we need to find out if our predictions are correct. RMSE is a way of measuring the error in our predictions - if our RMSE is high, our predictions are bad and vice versa.
# 
# MAE
# 
# MAE stands for mean absolute error. Just like RMSE, MAE is a way of measuring the error in our predictions - if our MAE is high, our predictions are bad and vice versa.

# In[ ]:


linreg = linear_model.LinearRegression()
linreg.fit(X_train, y_train)

y_test_predict = linreg.predict(X_test)
y_train_predict = linreg.predict(X_train)

print('R^2 training: %.3f, R^2 test: %.3f' % (
      (metrics.r2_score(y_train, y_train_predict)), 
      (metrics.r2_score(y_test, y_test_predict))))

linreg_metrics = get_regression_metrics('Linear Regression', y_test, y_test_predict)
linreg_metrics


# ## Task 3: Plot the weights of features obtained by linear regression

# In[ ]:





# The parameters of the model setting2, setting1, s1, s2 etc. are plotted here. You can see that red lines for setting2 and s6 are the longest. So, they are the most important.
