import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_dump', 'rb'))

# %%


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    words = model.wv[int_features]  # checking the vocabulary

    #final_features = [np.array(int_features)]
    prediction = model.most_similar(words)

    #output = round(prediction[0], 2)

    print('\n')
    xy = render_template('index.html', prediction_text='The most similar words are : {}'.format(prediction))
    print('\n')
    print('The total number of similar words, ', len(prediction))
    print('\n')
    return xy

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.most_similar((list(data.values())))

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)