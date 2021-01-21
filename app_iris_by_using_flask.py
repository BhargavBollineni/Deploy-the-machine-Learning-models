# We use App.py For Api Integrations 
from flask import Flask,render_template,url_for,request
from flask_material import Material


import joblib
import pandas as pd
import numpy as np


app = Flask(__name__)
Material(app)

#We have to Say Flask Where you have route

@app.route('/')
def index():
    return render_template("index.html")

# We Nead to Preview the Data
@app.route('/preview')
def preview():
    Path = "https://raw.githubusercontent.com/reddyprasade/Machine-Learning-Problems-DataSets/master/Classification/iris.csv"
    df = pd.read_csv(Path)
    return render_template("preview.html",df_view=df)


# Collect the Data From the User Sepal width, Sepal length, Petal width and Petal length
@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		petal_length = request.form['petal_length']
		sepal_length = request.form['sepal_length']
		petal_width = request.form['petal_width']
		sepal_width = request.form['sepal_width']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [sepal_length,sepal_width,petal_length,petal_width]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		if model_choice == 'logitmodel':
		    logit_model = joblib.load('Model/LogisticRegression_iris_dataSet.pkl')
		    result_prediction = logit_model.predict(ex1)
		elif model_choice == 'knnmodel':
			knn_model = joblib.load('data/knn_model_iris.pkl')
			result_prediction = knn_model.predict(ex1)
		elif model_choice == 'dtree':
		    logit_model = joblib.load('data/dtree_model_iris.pkl')
		    result_prediction = logit_model.predict(ex1)
		elif model_choice == 'svmmodel':
			knn_model = joblib.load('data/svm_model_iris.pkl')
			result_prediction = knn_model.predict(ex1)

	return render_template('index.html', petal_width=petal_width,
		sepal_width=sepal_width,
		sepal_length=sepal_length,
		petal_length=petal_length,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)


if __name__ == '__main__':
	app.run(debug=True)
