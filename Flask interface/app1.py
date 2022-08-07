import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

app=Flask(__name__)
model1=load_model("fruit.h5")
model2=load_model("vegetable.h5")

#Home Page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    return render_template('predict.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        f=request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img = image.load_img(filepath,target_size=(128,128))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        plant = request.form('plant')
        print(plant)
        if (plant == 'vegetable'):
            preds=model2.predict(x)
            preds=np.argmax(preds)
            print(preds)
            df = pd.read_excel('precausions-veg.xlsx')
            print(df.iloc[preds]['caution'])
        else:
            preds=model1.predict(x)
            preds=np.argmax(preds)
            df = pd.read_excel('precausions-fruits.xlsx')
            print(df.iloc[preds]['caution'])
    return df.iloc[preds]['caution']

if __name__ == '__main__':
    app.run(debug=False)
    