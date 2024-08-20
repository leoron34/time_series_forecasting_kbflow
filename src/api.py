from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('models/lstm_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, 1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': float(prediction[0, 0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
