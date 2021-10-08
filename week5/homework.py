import pickle

with open('./dv.bin', 'rb') as f_in:
	dv = pickle.load(f_in)
print(dv)

with open('./model1.bin', 'rb') as f_in:
	model = pickle.load(f_in)
print(model)


# Q3

customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}


X = dv.transform([customer])

score = model.predict_proba(X)[0, 1]
print("Q3:", score)


# Q4

from flask import Flask
from flask import request
from flask import jsonify

app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict():
	customer = request.get_json()

	X = dv.transform([customer])
	y_pred = model.predict_proba(X)[0, 1]
	
	result = { 'probability': float(y_pred) }
	return jsonify(result)


