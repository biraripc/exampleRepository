import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore
import matplotlib.pyplot as plt

data = pd.read_csv("dataset02.csv")
data_numeric = data.apply(pd.to_numeric, errors='coerce') 

data_numeric.isnull().sum()

clean_data = data_numeric.dropna()

z_scores = np.abs(zscore(clean_data))
final_data = clean_data[(z_scores < 2).all(axis=1)]

Q1 = final_data.quantile(0.25)
Q3 = final_data.quantile(0.75)
IQR = Q3 - Q1
final_data = final_data[~((final_data < (Q1 - 1.5 * IQR)) | (final_data > (Q3 + 1.5 * IQR))).any(axis=1)]

final_data = (final_data - final_data.min()) / (final_data.max() - final_data.min())

training_data = final_data.sample(frac=0.8, random_state=42)
training_data.to_csv('/tmp/dataset02_training.csv', index=False)

testing_data = final_data.drop(training_data.index)
testing_data.to_csv('/tmp/dataset02_testing.csv', index=False)

plt.scatter(training_data['x'], training_data['y'], color='blue', label='Training Data')
plt.scatter(testing_data['x'], testing_data['y'], color='orange', label='Testing Data')
plt.plot(training_data['x'], sm.OLS(training_data['y'], sm.add_constant(training_data[['x']])).fit().predict(sm.add_constant(training_data[['x']])), color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of x vs y')
plt.legend()
plt.savefig('UE_04_App2_ScatterVisualizationAndOLSModel.pdf')
plt.show()
plt.close()

final_data.boxplot()
plt.title('Box Plot of All Dimensions')
plt.savefig('UE_04_App2_BoxPlot.pdf')
plt.show()
plt.close()

# Import required libraries and the diagnostic class
import pandas as pd
import statsmodels.formula.api as smf
from UE_04_LinearRegDiagnostic import LinearRegDiagnostic

model = smf.ols('y ~ x', data=final_data).fit()

# Initialize the LinearRegDiagnostic class
diagnostic = LinearRegDiagnostic(model)

vif_table, fig, ax = diagnostic(plot_context='seaborn-talk')

fig.savefig('UE_04_App2_DiagnosticPlots.pdf')

print(vif_table)



