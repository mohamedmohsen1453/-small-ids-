import pandas as pd
import json

with open(r'test.json', 'r') as file:
    data = json.load(file)

# If the JSON is a list of records and not a dictionary with a 'Records' key, normalize directly on the list
df = pd.json_normalize(data)

# Save the DataFrame to CSV
df.to_csv('output_json.csv', index=False)

# Show the first few rows of the DataFrame
print(df.head())
