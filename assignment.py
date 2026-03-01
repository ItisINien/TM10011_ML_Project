# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!


#%% Data loading functions. Uncomment the one you want to use
#from worcgist.load_data import load_data
#from worclipo.load_data import load_data
from worcliver.load_data import load_data
#from hn.load_data import load_data
#from ecg.load_data import load_data
data = load_data()


# %%
import numpy as np
numeric_cols = data.select_dtypes(include=[np.number])

# Bereken Z-scores
z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()

# Zoek outliers: rijen met minstens één Z-score > 3 of < -3
outliers = numeric_cols[(np.abs(z_scores) > 3).any(axis=1)]

print(f"Aantal rijen met outliers: {len(outliers)}")


# %%
