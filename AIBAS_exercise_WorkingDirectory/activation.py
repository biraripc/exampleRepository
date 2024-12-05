import pandas as pd

df = pd.read_csv("cleaned_data.csv")

single_entry = df.iloc[:1]
print(single_entry)

single_entry.to_csv("currentActivation.csv", index=False)