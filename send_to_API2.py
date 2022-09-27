import requests
import pandas as pd


# Specify the URL
url = 'http://127.0.0.1:5000/get_report'

# Specify the request to be a report that contains the data for a specific customer using CustomerID and date
req ='report'
date = '2022-09-01'
CustomerID = 72

# Send a get request 
x = requests.get(url, params={'req':req, 'date': date, 'CustomerID':CustomerID})

# Send
