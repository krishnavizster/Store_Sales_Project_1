
Stores Sales Prediction Project
The objective of this notebook is to follow a step-by-step workflow, explaining each step and rationale for every decision we take during solution development.
# Store_Sales_Project_1
This Model will predict the sales of each store 
prediction deployement link 
https://salespredictionkk.herokuapp.com/

...............................................................................
Problem Statement:
Nowadays, shopping malls and Big Marts keep track of individual item sales data in order to forecast future client demand and adjust inventory management. In a data warehouse, these data stores hold a significant amount of consumer information and particular item details. By mining the data store from the data warehouse, more anomalies and common patterns can be discovered.
...............................................................................

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Approach: The classical machine learning tasks like Data Exploration, Data Cleaning, Feature Engineering, Model Building and Model Testing. Try out different machine learning algorithms that’s best fit for the above case.
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

...............................................................................
Imports 
# Data manupulation and analysis
import pandas as pd
import numpy as np
# import pycaret
# import klib
# import dtale
# from pandas_profiling import ProfileReport
# from pycaret.regression import *

# Data visualization 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import scipy

# Sklearn preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score
import sklearn.metrics as metrics
from math import sqrt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# Ignore Warnings
import warnings 
warnings.filterwarnings('ignore')
...............................................................................
Workflow stages
The solution workflow goes through seven stages described in the Data Science Solutions book.

1.Question or problem definition.
2.Acquire training and testing data.
3.Wrangle, prepare, cleanse the data.
4.Analyze, identify patterns, and explore the data.
5.Model, predict and solve the problem.
6.Visualize, report, and present the problem solving steps and final solution.
7.Supply or submit the results.
The workflow indicates general sequence of how each stage may follow the other. However there are use cases with exceptions.

1.We may combine mulitple workflow stages. We may analyze by visualizing data.
2.Perform a stage earlier than indicated. We may analyze data before and after wrangling.
3.Perform a stage multiple times in our workflow. Visualize stage may be used multiple times.
4.Drop a stage altogether. We may not need supply stage to productize or service enable our dataset.
