from flask import Flask, render_template, request, app, Response
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open("models/standardScalar.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("models/modelForPrediction.pkl", "rb") as file:
    model = pickle.load(file)

print(scaler)

@app.route('/', methods= ['GET', 'POST'])
def home():
    
    if request.method == 'GET':
        return render_template('index.html')
    else:
        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'
            
        return render_template('single_prediction.html',result=result)

if __name__ == '__main__':
    app.run(debug=True)