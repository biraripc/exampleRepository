import numpy as np
import pandas as pd
import statsmodels.api as sm


data = pd.read_csv("dataset01.csv")

#Number of data entries in column 'y'
print(f"Number of entries in 'y': {data['y'].count()}")

#Mean of column 'y'
print(f"Mean of 'y': {data['y'].mean()}")

#Standard deviation of column 'y'
print(f"Standard deviation of 'y': {data['y'].std()}")

#Variance of column 'y'
print(f"Variance of 'y': {data['y'].var()}")

#Min and max of column 'y'
print(f"Min of 'y': {data['y'].min()}, Max of 'y': {data['y'].max()}")

#OLS Model: x -> y
x = sm.add_constant(data['x'])  # Add constant for intercept
y = data['y']
model = sm.OLS(y, x).fit()

# Print model summary
print(model.summary())

# Save OLS model to a file
with open("OLS_model", "w") as f:
    f.write(model.summary().as_text())
