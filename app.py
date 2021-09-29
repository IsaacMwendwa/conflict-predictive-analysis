from flask import Flask, render_template, request, redirect, url_for, current_app, send_from_directory
import os
from os.path import join, dirname, realpath

from datetime import datetime
from fbprophet import Prophet
#from fbprophet.plot import plot_plotly
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import plotly.offline as py
import pickle
#py.init_notebook_mode()
#%matplotlib inline

# function for making predictions
def return_prediction(model, periods):
    # prediction window: future
    future = model.make_future_dataframe(periods=periods)

    print(future.head(2))
    print(future.tail(2))
    
    # make a prediction
    forecast = model.predict(future)
    predictions = forecast[2814:]

    return predictions

# function for getting prediction labels
def return_pred_values(predictions, scaler):
    #get predictions
    pred = predictions['yhat']
    #print(pred)

    # Round off pred values to whole no.s
    rounded_pred = round(pred)
    #print(np.unique(rounded_pred))

    # Type cast to int
    int_pred = rounded_pred.astype('int')
    #print(np.unique(int_pred))

    # Reverse transform predicted values to get labels
    pred_labels = scaler.inverse_transform(int_pred)

    # display labels
    unique_pred_labels = np.unique(pred_labels)
    #print(unique_pred_labels)

    return unique_pred_labels


    values = dict.values()
    total= 0
    for v in values:
        total += v 
    
    return total



app = Flask(__name__, template_folder="templates")

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

# LOADING THE MODEL AND THE SCALER: COUNIES PREDICTION
with open("counties_model.bin", 'rb') as f_in:
    counties_model = pickle.load(f_in)

with open("le_admin1_scaler.pkl", 'rb') as p_in:
    counties_scaler = pickle.load(p_in)

# LOADING THE MODEL AND THE SCALER: EVENT_TYPE PREDICTION
with open("event_type_model.bin", 'rb') as e_in:
    event_type_model = pickle.load(e_in)

with open("event_scaler.pkl", 'rb') as g_in:
    event_scaler = pickle.load(g_in)

@app.route('/')
def Homepage():
    return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        pred_date = request.form['date']
        pred_date_obj = datetime.strptime(pred_date, '%d/%m/%Y')
        #print(type(pred_date_obj))
        #print(pred_date_obj)

        #Read CSV file
        df = pd.read_csv(R'C:\Users\Admin\Desktop\Nebula\conflict_prediction\static\files\KENYA_ACLED_DATASET_COMBINED_FINAL.csv', parse_dates=['EVENT_DATE'])

        last_record = df.tail(1)
        model_date = last_record.EVENT_DATE.values[0]
        model_date_obj = datetime.utcfromtimestamp(model_date.tolist()/1e9)
        #print(type(model_date_obj))
        #print(model_date_obj)

        # difference in days
        diff = pred_date_obj - model_date_obj
        periods = diff.days

        #print(periods)

        counties_predictions = return_prediction(counties_model, periods)
        event_predictions = return_prediction(event_type_model, periods)
        #print(event_type_predictions)

        #return_pred_values(predictions, counties_scaler)
        counties_pred_values = return_pred_values(counties_predictions, counties_scaler)
        #print(counties_pred_values)

        events_pred_values = return_pred_values(event_predictions, event_scaler)
        #print(events_pred_values)
        #print(type(events_pred_values))
     
        #rendering results
        return render_template("prediction.html", events_pred_values = events_pred_values,
        counties_pred_values = counties_pred_values)
    
    

if __name__ == "__main__":
    #port = int(os.environ.get("PORT", 5000))
    #app.run(host='0.0.0.0', port=port)
    app.run()