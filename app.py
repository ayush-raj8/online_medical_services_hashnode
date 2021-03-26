# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import joblib
import json
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf



app = Flask(__name__)

# Load the Random Forest CLassifier model
filename = 'Models/diabetes-model.pkl'

classifier = pickle.load(open(filename, 'rb'))



@app.route('/')
def home():
	return render_template('index.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if request.method == 'POST':
        preg = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bp = float(request.form['bloodpressure'])
        st = float(request.form['skinthickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])
        s=""
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = float(classifier.predict(data))
        if my_prediction==1:
            s="danger"
        else :
            s="safe"
        return render_template('d_result.html',a0=preg,a1=glucose,a2=bp,a3=st,a4=insulin,a5=bmi,a6=dpf,a7=age, prediction=my_prediction,prediction_text=s)



def getParameters1():
    parameters = []
    #parameters.append(request.form('name'))
    parameters.append(request.form['age'])
    parameters.append(request.form['sex'])
    parameters.append(request.form['cp'])
    parameters.append(request.form['trestbps'])
    parameters.append(request.form['chol'])
    parameters.append(request.form['fbs'])
    parameters.append(request.form['restecg'])
    parameters.append(request.form['thalach'])
    parameters.append(request.form['exang'])
    parameters.append(request.form['oldpeak'])
    parameters.append(request.form['slope'])
    parameters.append(request.form['ca'])
    parameters.append(request.form['thal'])
    return parameters


def ValuePredictor(to_predict_list, size):
    loaded_model = joblib.load('models/heart_model')
    to_predict = np.array(to_predict_list).reshape(1,size)
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        print(to_predict_list)
        a=[]
        a=to_predict_list
        result = ValuePredictor(to_predict_list, 11)
    s = ""
    if (int(result) == 1):
        prediction = 1
        s = "Danger"
    else:
        prediction = 0
        s="Healthy"

    a1=""
    if(int(a[1])==1):
        a1="Male"
    else:
        a1="Female"

    a2=""
    if(int(a[2])==1):
        a2="Typical Angina"
    elif (int(a[2])==1):
        a2="Atypical Angina"
    elif (int(a[2]) == 1):
        a2="Non-anginal Pain"
    elif (int(a[2]) == 1):
        a2="Asymptomatic"

    a5 = ""
    if (int(a[5]) == 0):
        a5 = "Normal"
    elif (int(a[2]) == 1):
        a5 = "Having ST-T wave abnormality "
    elif (int(a[2]) == 2):
        a5 = "Probable or definite left ventricular hypertrophy"

    a8 = ""
    if (int(a[8]) == 0):
        a8 = "No"
    elif (int(a[8]) == 1):
        a8 = "yes "

    a10 = ""
    if (int(a[10]) == 3):
        a10 = "Normal"
    elif (int(a[10]) == 6):
        a10 = "Fixed Defect "
    elif (int(a[10]) == 7):
        a10 = "Reversible Defect"


    a9 = ""
    if (int(a[9]) == 0):
        a9= "Upsloping"
    elif (int(a[2]) == 1):
        a9= "Flat"
    elif (int(a[2]) == 2):
        a9 = "Downsloping"



    return render_template('h_result.html', prediction=prediction,prediction_text=s,a0=a[0],a1=a1,a2=a2,a3=a[3],a4=a[4],a5=a5,a6=a[6],a7=a[7],a8=a8,a9=a[9],a10=a[10])







def getParametersL():
    parameters = []
    parameters.append(request.form['age'])
    parameters.append(request.form['sex'])
    parameters.append(request.form['Total_Bilirubin'])
    parameters.append(request.form['Direct_Bilirubin'])
    parameters.append(request.form['Alkaline_Phosphotase'])
    parameters.append(request.form['Alamine_Aminotransferase'])
    parameters.append(request.form['Aspartate_Aminotransferase'])
    parameters.append(request.form['Total_Protiens'])
    parameters.append(request.form['Albumin'])
    parameters.append(request.form['Albumin_and_Globulin_Ratio'])
    return parameters






@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/predict_liver', methods=['POST'])
def predict_liver():
    model = pickle.load(open('Models/liver.pkl', 'rb'))
    parameters = getParametersL()
    para=parameters
    inputFeature = np.asarray(parameters).reshape(1, -1)
    my_prediction = model.predict(inputFeature)
    s=""
    if int(parameters[1])==0:
        s=" Female"
    elif int(parameters[1])==1:
        s=" Male"


    print(inputFeature)
    print(my_prediction)
    print(parameters[1])

    output = round(float(my_prediction[0]), 2)
    if(output == 1):
        return render_template('l_result.html',prediction=output, a0=parameters[0],a1=s,a2=parameters[2],a3=parameters[3],a4=parameters[4],a5=parameters[5],a6=parameters[6],a7=parameters[7],a8=parameters[8],a9=parameters[9],prediction_text='High chances of Liver disease.Consult doctor!')
    if (output == 0):
        return render_template('l_result.html',a0=parameters[0],a1=s,a2=parameters[2],a3=parameters[3],a4=parameters[4],a5=parameters[5],a6=parameters[6],a7=parameters[7],a8=parameters[8],a9=parameters[9], prediction_text='Low chances of Liver disease. Chill !!')






def get_list():
    test=pd.read_csv("Datasets/test_data.csv",error_bad_lines=False)
    x_test=test.drop('prognosis',axis=1)
    col=x_test.columns
    print(col)
    col=col.str.replace("_"," ")
    print(type(col))
    col=col.tolist()
    print(type(col))
    return col

@app.route("/symptoms", methods=['GET', 'POST'])
def symptoms():
    col=get_list()
    return render_template('symptoms.html',values1= (col))

@app.route('/symppredict',methods=['POST','GET'])
def symppredict():
    if request.method=='POST':
        model = pickle.load(open('Models/model.sav', 'rb'))
        inputt = []
        print(type(inputt))
        to_predict_list = request.form.to_dict()
        inputt = list(to_predict_list.values())
        print(type(inputt))
        print(type(inputt))
        b=[0]*132
        col=get_list()
        print(inputt)



        for x in range(0,132):
            for y in inputt:
                if(col[x]==y):
                    b[x]=1

        b=np.array(b)
        b=b.reshape(1,132)
        prediction = model.predict(b)
        prediction=prediction[0]
    return render_template('r_symptoms.html', pred="{}".format(prediction),a0=inputt[0],a1=inputt[1],a2=inputt[2])



















@app.route("/malaria", methods=['GET', 'POST'])
def malaria():
    return render_template('malaria.html')

@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredict():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,3))
                img = img.astype(np.float64)
                model = load_model("Models/malaria.h5")
                model_path="Models/malaria.h5"
                model = tf.keras.models.load_model(model_path)
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message = message)
    return render_template('malaria_predict.html', pred = pred)











# this function use to predict the output for Fetal Health from given data
def fetal_health_value_predictor(data):

    data = list(data.values())
    data = list(map(float, data))
    data = np.array(data).reshape(1,-1)
        # load the saved pre-trained model for new prediction
    model_path = 'Models/fetal.pkl'
    model = pickle.load(open(model_path, 'rb'))
    result = model.predict(data)
    result = int(result[0])

        # returns the predicted output value
    return (result)


# this route for prediction of Fetal Health
@app.route('/fetal_health', methods=['GET','POST'])
def fetal_health():
    return render_template('fetal_health.html')

@app.route('/fetal_health_res', methods=['GET','POST'])
def fetal_health_res():
    if request.method == 'POST':
        # geting the form data by POST method
        data = request.form.to_dict()
        print(data)
        model = pickle.load(open('Models/fetal.pkl', 'rb'))
        result = fetal_health_value_predictor(data)
        print(result)
        a = list(data.values())
        a = list(map(float, a))

        s = ""
        if int(result) == 1:
            s = " NORMAL"
        elif int(result) == 2:
            s = " SUSPECT"
        elif int(result) == 3:
            s = " SUSPECT"

    return render_template('f_result.html',result=result,a=a, prediction=result)



if __name__ == '__main__':
	app.run(debug=True)
