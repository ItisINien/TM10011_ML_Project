# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!


#%% Data loading functions. Uncomment the one you want to use
import pandas as pd
import numpy as np
from worcliver.load_data import load_data
import matplotlib.pyplot as plt

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

#Je wilt even kijken of het aantal malignant en benign tumoren een beetje vergelijkbaar is
#als je bijv 80% van een label hebt kun je met een model wat altijd dat zegt al een hoge accuracy bereiken
print(data['label'].value_counts())
print(data['label'].value_counts(normalize=True))

#check the missing values
#print(data.isnull().sum())
#no missing values!

#because p > N, we have a risk of overfitting, so we need to do feature selection
# univariate feature selection using t-test
from scipy.stats import ttest_ind

#1) make y
y = data['label'].map({'benign':0, 'malignant':1})

#2) drop columns 1 and 2
x = data.drop(columns=['label']) #ID is index and therefore not in data

#3) split per class
x_benign = x[y == 0]
x_malign = x[y == 1]

#4) T-test per feature
t_values = [] 
p_values = []

for col in x.columns: #check for each feature
    t, p = ttest_ind( #perform t-test, compare mean between two groups
        x_benign[col].values,
        x_malign[col].values,
        equal_var=False, #welch's t-test, for medical data more realistic
        nan_policy='omit' #when dealing with NaN
    )
    t_values.append(t)
    p_values.append(p)

results = pd.DataFrame({
    'feature': x.columns, #name of feature
    't_statistics': t_values, #t-test values
    'p_value': p_values, #p values
    'abs_t': np.abs(t_values), #absolute t-test values
}).sort_values(['p_value', 'abs_t'], ascending=[True, False]) 
#sort p-values ascending (smallest pvalue first), if p-values are similar sort on t-value (biggest value first)

top50 = results.head(50) #for example select the 50 'best' features
top50_features = top50['feature'].tolist() #add them to a list 
print(top50[["feature", "t_statistics", "p_value"]].to_string(index=False)) #print the list

