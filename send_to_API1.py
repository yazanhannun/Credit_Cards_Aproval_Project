import requests
import pandas as pd


# Specify the URL
url = 'http://127.0.0.1:5000/get_data'

# Asgine the the file name to a varaible
file_name ='cc_apps_data.csv'

# Convert the file into a DataFrame
data = pd.read_csv(f'{file_name}', index_col=0)

# Convert the file into a Json object
files = data.to_json()

# print it to check the data before send it
print(files)

# Send a get request 
x = requests.get(url, params={'files':files})