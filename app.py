from flask import Flask, flash, request, jsonify, redirect, url_for, render_template
import os
import pandas as pd
import numpy as np
import data_kopi
import missing_value
import seleksi_fitur
import akurasi
import pickle
import imp
from pyexpat import features
from mlxtend.regressor import StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


UPLOAD_FOLDER = "/xampp/htdocs/sistem penilaian biji kopi"
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
   
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == 'kopi.csv':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template("upload.html")   

@app.route("/dataset")
def dataset():
    data_kopi.dataset()
    raw_data = pd.read_csv('dataset.csv')
    n_dataset = len(raw_data)
    print (n_dataset)

    return render_template("dataset.html", raw_data=raw_data, n_dataset=n_dataset)
    
@app.route("/preprocessing")
def preprocessing():
    missing_value.missing_value()
    seleksi_fitur.seleksi_fitur()

    data_seleksi_fitur = pd.read_csv('seleksi_fitur_data.csv')   
    data_clean = pd.read_csv('clean_data.csv')

    n_clean = len(data_clean)
    n_seleksi_fitur = len(data_seleksi_fitur)
    print(n_clean)
    print(n_seleksi_fitur)

    return render_template("preprocessing.html", data_clean=data_clean, n_clean=n_clean, data_seleksi_fitur=data_seleksi_fitur, n_seleksi_fitur=n_seleksi_fitur)

@app.route("/accuration")
def accuration():
    akurasi.akurasi()

    data_akurasi = pd.read_csv('akurasi.csv')

    n_akurasi = len(data_akurasi)
    print (n_akurasi)
    return render_template("accuration.html", data_akurasi=data_akurasi, n_akurasi=n_akurasi )

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        Flavor = request.form['Flavor']
        Balance = request.form['Balance']
        Acidity = request.form['Acidity']
        Cupper_Points = request.form['Cupper.Points']
        Aroma = request.form['Aroma']
        Body = request.form['Body']
        Uniformity = request.form['Uniformity']
        Clean_Cup = request.form['Clean.Cup']
        Sweetness = request.form['Sweetness']
        Category_Two_Defects = request.form['Category.Two.Defects']
        Aftertaste = request.form['Aftertaste']
        features = [float(Flavor),float(Balance),float(Acidity),float(Cupper_Points),float(Aroma),float(Body),float(Uniformity),float(Clean_Cup),float(Sweetness),float(Category_Two_Defects),float(Aftertaste)]
        model = pickle.load(open("model.pkl", "rb"))
        features = [np.array(features)]
       # scaler= StandardScaler()
        df = pd.read_csv("seleksi_fitur_data.csv")
        X = df[['Flavor', 'Balance', 'Acidity', 'Cupper.Points', 'Aroma','Body', 'Uniformity', 'Clean.Cup', 'Sweetness', 'Category.Two.Defects', 'Aftertaste']]
        y = df['Total.Cup.Points']
        X.shape
        y.shape
        # train test split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size= 0.1, random_state= 123)
        #scale data

        scaler= StandardScaler()
        X_train= scaler.fit_transform(X_train)
        prediction = model.predict(scaler.transform(features))
        return render_template("penilaian.html", prediction_text = "{}".format(prediction))
    return render_template("penilaian.html")

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(debug=True)