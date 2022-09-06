
# Import important pacakges
from datetime import datetime
import os
from flask import Flask, request
import numpy as np
import pandas as pd
import json 
import sqlite3
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


app = Flask(__name__)
APP_ROUTE = os.path.dirname(os.path.abspath(__file__))

'''
API will be created to retrieve applications from the client for example (a bank), handle missing values using 
prepro() function in preprocessing.py file and predict the statuts of their application using the model()
function in file model.py. Then create a sqlite data base contains a table named CCP insert the the data into it.
Moreover, the sqlite database should also be auto synchronized to a cloud database (Firebase)'''

# API
@app.route("/get_data", methods= ['GET'])
def get_data():

    # 2.7.1 Retrive the data from the client That require handling, preprocessing and prediction
    data0 = json.loads(request.args.get("files"))
    data_pre = pd.DataFrame(data0)
    
    # --------------------------------2.7.2 Handle and preprocess the data for prediction-------------------------------#

    from preprocessing import prepro
    from model import model

    # Preprocess the file and handle the missing values using the function prepro() in preprocessing.py file
    data_p = prepro(data_pre)

    # Call the model funtion in model.py file and predict the data sent by client 
    data_pred = model(data_p)

    # -------------------------2.7.3 Create SQLite database ------------------------------------#

    # Create a Table named CCP in the database
    try:
        # Connect to the sqlite database
        conn = sqlite3.connect('info.db')
        cur = conn.cursor()
        query1 = '''CREATE TABLE CCP(
                    ID             INT          NOT NULL,
                    DATE           timestamp    NOT NULL,
                    STATUS         TEXT         NOT NULL
                    );'''
        
        cur.execute(query1)

    # Insert predicted data into the CCP table 
    except:
        print ("Table already exists")

        # Close previous connection to the database
        conn.commit()
        conn.close()

    # This date will be used to select the data that have been inserted recently into the database to 
    # create an Excel report
    date1 = datetime.now()

    # Connect to the sqlite database
    conn = sqlite3.connect('info.db')
    cur = conn.cursor()      
    s = data_pre.shape


    # Loop over the data_pred and insert the data into the local sqlite 
    for i in range(0, s[0]):
        id =int(data_pre.iloc[i, 1])
        status_binary = data_pred[i]
        date = datetime.now() 
        if int(status_binary) == 1:
            status = 'Denied' 
        else:
            status = 'Approved'

        cur.execute(f"INSERT INTO CCP(ID,DATE,STATUS) \
        VALUES ('{id}', '{date}', '{status}')"); 

    print ("Data Inserted Successfully")

    # Close previous connection to the database
    conn.commit()        
    conn.close()

    #-----------2.7.4 Synchronize the data into a cloud database and Create Excel report-------------#

    # Connect to the sqlite database
    conn = sqlite3.connect("info.db")
    cur = conn.cursor()
    query="select * from CCP"
    data=pd.read_sql(query,conn)

    # Take the data that have been predicted recently after date1 
    data = data.query(f"DATE>= '{date1}'")    
  
    # logging in using private key
    cred = credentials.Certificate('project-ba02d-firebase-adminsdk-7uwsq-7d185bc609.json')

    firebase_admin.initialize_app(cred, {'databaseURL' : 'https://project-ba02d-default-rtdb.firebaseio.com/', 
    'httpTimeout' : 30})

    print('logged in to firebase')

    # set the location where you want to write or read and add the data to the database 
    for i in range(0, 228):
        
        # DATE column 
        ref1 = f"{data.iloc[i, 0]}/DATE"  
        root = db.reference(ref1)

        # writing on the database
        x = {'DATE': f'{data.iloc[i, 1]}'}
        root.set(x)

        # STATUS column 
        ref2 = f"{data.iloc[i, 0]}/STATUS"  
        root = db.reference(ref2)

        # writing on the database
        x = {'STATUS': f'{data.iloc[i, 2]}'}
        root.set(x)

    conn.commit()
    conn.close()
    return 'ok'


if __name__ == "__main__":
    app.run(debug=False, host= '0.0.0.0', port= 5000)