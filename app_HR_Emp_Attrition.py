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
    Path = "https://raw.githubusercontent.com/KarinkiManikanta/Data-Sets-For-Machine-Learnig-and-Data-Science/master/DataSets/Human_Resources_Employee_Attrition.csv"
    df = pd.read_csv(Path)
    return render_template("preview.html",df_view=df)


# Collect the Data From the User satisfaction_level,last_evaluation,number_of_projects
@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		satisfaction_level  = request.form['satisfaction_level']
		last_evaluation = request.form['last_evaluation']
		number_of_projects  = request.form['number_of_projects ']
		average_monthly_hours = request.form['average_monthly_hours']
		years_at_company  = request.form['years_at_company ']
		work_accident =request.form['work_accident']
		left = request.form['left']
		promotion_last_5years = request.form['promotion_last_5years']
		department =request.form['department']

        
# Clean the data by convert from unicode to float 
		sample_data = [satisfaction_level, last_evaluation, number_of_projects,
       average_monthly_hours, years_at_company, work_accident, left,
       promotion_last_5years, department]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		if model_choice == 'logitmodel':
		    logit_model = joblib.load('model/LGClass_HR_Employe_Attrition.pkl')
		    result_prediction = logit_model.predict(ex1)
		elif model_choice == 'knnmodel':
			knn_model = joblib.load('model/KNNClassifier_HR_Employe_Attrition.pkl')
			result_prediction = knn_model.predict(ex1)
		elif model_choice == 'GNB':
			gnb_model = joblib.load('model/GNBClassifier_HR_Employment_Attrition.pkl')
			result_prediction = gnb_model.predict(ex1)
		
		elif model_choice == 'dtree':
		    dt_model = joblib.load('model/DTClassifier_HR_Employe_Attrition.pkl')
		    result_prediction = dt_model.predict(ex1)
		elif model_choice == 'svmmodel':
			svm_model = joblib.load('model/SVMClassifier_HR_Employment_Attrition.pkl')
			result_prediction = svm_model.predict(ex1)
		elif model_choice == 'RandomForesttree':
		    rf_model = joblib.load('model/RFclassifier_HR_Employe_Attrition.pkl')
		    result_prediction = rf_model.predict(ex1)
		elif model_choice == 'AdaBoosting':
			ab_model = joblib.load('model/AdaBoostingClassifier_HR_Employment_Attrition.pkl')
			result_prediction = ab_model.predict(ex1)

	return render_template('index.html',satisfaction_level=satisfaction_level, last_evaluation=last_evaluation, 			                           number_of_projects=number_of_projects, average_monthly_hours=average_monthly_hours, 
                                           years_at_company=years_at_company, work_accident=work_accident,
                                           left=left,promotion_last_5years=promotion_last_5years,department= department,
					     clean_data=clean_data,
		                             result_prediction=result_prediction,
		                             model_selected=model_choice)


if __name__ == '__main__':
	app.run(debug=True)
