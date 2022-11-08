
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from mlmodel import prediksi

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/visual')
def visualize():
    return render_template('visual.html')


@app.route('/pred')
def pred():
    return render_template('index.html')


@app.route('/more')
def more():
    return render_template('more.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():

    # # int_features = [float(x) for x in request.form.values()]

    int_features = request.form['penyakit1'], request.form['penyakit2']

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Rekomendasi tanaman toga adalah : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
