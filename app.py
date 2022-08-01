import flask 
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import os

app=Flask(__name__)
model=pickle.load(open('modelml.pkl','rb'))
@app.route('/')
def home():
    
    return render_template('sample.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():	

    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    output=model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():

    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    
    output=model.predict(final_features)[0]
    print(output)
    #output = round(prediction[0], 2)
    return render_template('sample.html', prediction_text="DMC Duff Moisture Code value is  {}".format(output))



if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True,port=os.environ['PORT'])