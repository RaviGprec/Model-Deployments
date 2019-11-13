import numpy as np
from flask import Flask, request, jsonify, render_template, redirect,url_for
import pickle
from sklearn.externals import joblib

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))
model = joblib.load('model.pkl') 
output = 1 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    
    final_features = [np.array(int_features)]
    final_features = np.insert(final_features, 0, 1., axis=1)
    prediction = model.predict(final_features)
    global output
    output = round(prediction[0], 2)

    #return render_template('index.html', prediction_text='The House Price for the selected options would be : Rs. {}'.format(output))
    return redirect(url_for('showresult'))

@app.route('/result')
def showresult():
    return render_template('showresult.html', prediction_text='The House Price for the selected options would be : Rs. {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)