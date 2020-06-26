from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf 
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

from flask import url_for,session
from flask_mysqldb import MySQL
import bcrypt
# import pymysql

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH = 'C:/Users/Jeanette/Desktop/cursoP/Red_Neuronal_Convolucional_rnc/Cancer_cervical-web/model/modelo1.h5py'
# MODEL_PATH ='C:/Users/Jeanette/Desktop/cursoP/Red_Neuronal_Convolucional_rnc/Cancer_cervical-web/model/modelo2.h5'
MODEL_PATH = 'C:/Users/Jeanette/Desktop/cursoP/Red_Neuronal_Convolucional_rnc/Cancer_cervical-web/model/model_resnet.h5'
# Load your trained model
model =tf.keras.models.load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary ----compila la función predict
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

app.secret_key='appLogin'

app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']=''
app.config['MYSQL_DB']='cancer'

mysql = MySQL(app)

semilla = bcrypt.gensalt()


def model_predict(img_path, model):
    # img = image.load_img(img_path, target_size=(150, 150))
    img = image.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    print("model predict.................")
    return preds

# @app.route('/login')
# def indexlogin():
#     # Main page
#     return render_template('login.html')

# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index_login.html')

@app.route('/')
def main():
    if 'nombre' in session:
        return render_template('index.html')
    else:
        return render_template('index_login.html')

@app.route('/inicio')
def inicio():
    return render_template('index.html')
    # if 'nombre' in session:
    #     return render_template('index.html')
    # else:
    #     return render_template('index_login.html')

@app.route('/ingresar', methods=['GET','POST'])
def ingresar():
    if (request.method == 'GET'):
        if 'nombre' in session:
            return render_template('index.html')
        else:
            return render_template('index_login.html')
    else:
        nombre = request.form['username']
        contrasena = request.form['pass']
        # contrasena_encode = contrasena.encode("utf-8")
        print("Insertando:")
        print("Contraseña::::",contrasena)
        # print("contraseña encriptada::::", contrasena_encode)

        cur = mysql.connection.cursor()
        sQuery = "select nombre, contrasena from users where nombre =%s"
            
        cur.execute(sQuery,[nombre])

        usuario = cur.fetchone()
        print("usuario[0]::::",usuario[0])
        print("usuario[1]::::",usuario[1])
        cur.close()

        if (usuario != None):
            # return redirect(url_for('inicio'))
            # contrasena_encriptado_encode = usuario[1].encode()
            # if(bcrypt.checkpw(contrasena_encode,contrasena_encriptado_encode)):
            if (contrasena == usuario[1] ):
                # session['username']= usuario[0]
                print("usuario[1]::::",contrasena, "-",usuario[1])
                return redirect(url_for('inicio'))
            else:
                # Flask("La contraseña es incorrecta", "alert-warning")
                return render_template("index_login.html")


@app.route('/salir')
def salir():
    session.clear()
    return redirect(url_for('inicio'))

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request        
        f = request.files['file']
        print('file::::',f)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path, model)
        #print('PREDS::::',preds)
        print('PREDS.SHAPE::::',preds.shape)
        print('LEN(PREDS.SHAPE)::::',len(preds.shape))
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        print("PRED_CLASS: ",pred_class)
        result = str(pred_class[0][0][1])               # Convert to string
        print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFILEPATH:::: ",result)
        return result        
    return None  # GET

if __name__ == '__main__':
    app.run(debug=True)