# Dependencies
from flask import Flask, json, request, jsonify
#from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import joblib 

# Your API definition
app = Flask(__name__)

@app.route("/")
def hello():
    return "Bienvenido a este servicio rest para la aplicacion de ciencia de datos aplicadas"

@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(clf.predict(query))
            print(prediction)
            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

@app.route('/predict2', methods=['POST'])
def predict2():
    if model:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns2, fill_value=0)

            prediction = list(model.predict(query))

            print(prediction)
            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
   
    clf = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    model = joblib.load("model2.pkl") # Load "model.pkl"
    print ('Model loaded2')
    model_columns2 = joblib.load("model_columns2.pkl") # Load "model_columns.pkl"
    print ('Model columns2 loaded')

    app.run(debug=True)