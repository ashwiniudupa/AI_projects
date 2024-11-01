from flask import Flask, render_template, redirect, request, url_for, send_from_directory, flash
import os
import numpy as np
import pickle

app = Flask(__name__)

REC_MODEL = pickle.load(open('naive_bayes_model.pkl', 'rb'))
FERT_MODEL = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
        return render_template("index.html")

@app.route('/croprecommendation/<res1>/<res2>')
def cropresult(res1, res2):
    print(res1)
    corrected_result1 = res1
    print(res2)
    corrected_result2 = res2
    return render_template('croprecresult.html', corrected_result1=corrected_result1, corrected_result2=corrected_result2)

@app.route('/croprecommendation', methods=['GET', 'POST'])
def cr():
    if request.method == 'POST':
        X = []
        Y=[]
        if request.form.get('nitrogen'):
            X.append(float(request.form.get('nitrogen')))
            Y.append(float(request.form.get('nitrogen')))
        if request.form.get('phosphorous'):
            X.append(float(request.form.get('phosphorous')))
            Y.append(float(request.form.get('phosphorous')))
        if request.form.get('potassium'):
            X.append(float(request.form.get('potassium')))
            Y.append(float(request.form.get('potassium')))
        if request.form.get('temperature'):
            X.append(float(request.form.get('temperature')))
        if request.form.get('humidity'):
            X.append(float(request.form.get('humidity')))
        if request.form.get('ph'):
            X.append(float(request.form.get('ph')))
        if request.form.get('rainfall'):
            X.append(float(request.form.get('rainfall')))
        X = np.array(X)
        X = X.reshape(1, -1)
        res1 = REC_MODEL.predict(X)[0]
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
        
        Y = np.array(Y)
        Y = Y.reshape(1, -1)
        res2 = FERT_MODEL.predict(Y)[0]
        fertilizer_dict = {
                0: 'Urea',
                1: 'DAP',
                2: 'Fourteen-Thirty Five-Fourteen',
                3: 'Twenty Eight-Twenty Eight',
                4: 'Seventeen-Seventeen-Seventeen',
                5: 'Twenty-Twenty',
                6: 'Ten-Twenty Six-Twenty Six'
        }

        if res1 in crop_dict and res2 in fertilizer_dict:
            crop = crop_dict[res1]
            print(res1)
            res1 = "{}".format(crop)

            fert = fertilizer_dict[res2]
            print(res2)
            res2 = "{}".format(fert)
            return redirect(url_for('cropresult', res1=res1, res2=res2))
    return render_template('croprec.html')

if __name__== "__main__":
    app.run(debug=True)