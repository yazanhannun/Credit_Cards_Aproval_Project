# Import important pacakges
import os
import re
from flask import Flask, request
import numpy as np
import pandas as pd
import json 
import sqlite3


app = Flask(__name__)
APP_ROUTE = os.path.dirname(os.path.abspath(__file__))

'''
API will be created to retrieve applications from the client for example (a bank) and then send back a .xlsx 
report to the client contains the predictions of the status of their customers credit card applications. 
'''

# API
@app.route("/get_report", methods= ['GET'])
def get_data():

    # Break the request into variables req, date and CustomerID
    req= request.args.get("req")
    date = request.args.get("date")
    CustomerID = request.args.get("CustomerID")

   

    if req == 'report':
        # Connect to the sqlite database
        conn = sqlite3.connect("info.db")
        cur = conn.cursor()
        query=f"select * from CCP WHERE ID = {CustomerID} and DATE >= '{date}'"
        data=pd.read_sql(query,conn)
        # Take the data that have been predicted recently after date1 
        data.to_excel('report.xlsx')
        print('Report generated successfully')
    return 'ok'


if __name__ == "__main__":
    app.run(debug=False, host= '0.0.0.0', port= 5000)