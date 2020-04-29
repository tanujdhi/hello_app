
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('social.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 1:
        output = 'Yes, This ad is suitable for the particular age!!'
    elif prediction == 0:
        output = 'No, This ad is not suitable for the particular age!!'
    else:
        print("Invalid")

    return render_template('index1.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)