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
    Path = "https://raw.githubusercontent.com/reddyprasade/Machine-Learning-Problems-DataSets/master/Classification/Bank.csv"
    df = pd.read_csv(Path)
    return render_template("preview.html",df_view=df)


# Collect the Data From the User age,job,marital,education,default,balance,housing, loan, contact, day, month,duration, campaign, pdays,previews,poutcomet.
@app.route('/',methods=["POST"])
def analyze():
    if request.method == 'POST':
        age = request.form['age']
        job = request.form['job']
        marital = request.form['marital']
        education = request.form['education']
        default = request.form['default']
        balance =request.form['balance']
        housing = request.form['housing']
        loan = request.form['loan']
        contact =request.form['contact']
        day = request.form['day']
        month = request.form['month']
        duration =request.form['duration']
        campaign = request.form['campaign']
        pdays = request.form['pdays']
        previous =request.form['previous']
        poutcomet=request.form['poutcomet']


        # Clean the data by convert from unicode to float 
        sample_data=[age,job,marital,education,default,balance,housing,loan,contact,day,
                               month,duration,campaign,pdays,previous,poutcomet]
        clean_data = [float(i) for i in sample_data]

        # Reshape the Data as a Sample not Individual Features
        ex1 = np.array(clean_data).reshape(1,-1)

        # ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

        # Reloading the Model
        if model_choice == 'logitmodel':
        logit_model = joblib.load('model/LogisticRegression_Bank_dataSet1.pkl')
     result_prediction = logit_model.predict(ex1)
		elif model_choice == 'knnmodel':
			knn_model = joblib.load('model/KNNClassifier_BankDataset.pkl')
			result_prediction = knn_model.predict(ex1)
		elif model_choice == 'GNB':
			gnb_model = joblib.load('model/GNB_classification_BankDataset.pkl')
			result_prediction = gnb_model.predict(ex1)
		
		elif model_choice == 'dtree':
		    dt_model = joblib.load('model/DTclassifier_bank_dataset.pkl')
		    result_prediction = dt_model.predict(ex1)
		elif model_choice == 'svmmodel':
			svm_model = joblib.load('model/SVMclassfier_bank_dataset.pkl')
			result_prediction = svm_model.predict(ex1)
		elif model_choice == 'RandomForesttree':
		    rf_model = joblib.load('model/RFclassifier_bankdataser.pkl')
		    result_prediction = rf_model.predict(ex1)
		elif model_choice == 'AdaBoosting':
			ab_model = joblib.load('model/AdaBoostingclassifier_bankdataset.pkl')
			result_prediction = ab_model.predict(ex1)

	return render_template('index.html', age=age,job=job,marital=marital,education=education,
			                     default=default,balance=balance,housing=housing,loan=loan,
                                             contact=cotact,day=day,month=month,duration=duration,
					     campaign=campign,pdays=pdays,previous=previous,poutcomet=poutcomet,
					     clean_data=clean_data,
		                             result_prediction=result_prediction,
		                             model_selected=model_choice)


if __name__ == '__main__':
	app.run(debug=True)
