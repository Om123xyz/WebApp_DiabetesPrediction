Diabetes Prediction Streamlit App
The Diabetes Prediction App is a tool that predicts the probability of a patient having diabetes based on diagnostic measurements. This tool is intended for females above the age of 21 years, of Pima Indian heritage, and uses a dataset from the National Institute of Diabetes and Digestive and Kidney Diseases.

     

View Demo · Documentation · Report Bug · Request Feature

📔 Table of Contents
Dataset
Dependencies
Installation
Usage
Inputs
Outputs
Deployment and Notebook
License
Contact
📶 Dataset
The trained dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. The dataset can be found on Kaggle. It includes following health criteria:

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1)
Details
Number of Instances: 768
Number of Attributes: 8 plus class
Missing Attribute Values: Yes
Class Distribution: (class value 1 is interpreted as "tested positive for diabetes")
🧰 Dependecies
streamlit==0.88.0

pandas==1.3.3

numpy==1.21.2

matplotlib==3.4.3

plotly==5.3.1

seaborn==0.11.2

scikit-learn==0.24.2

joblib==1.1.0

scipy==1.7.3

torch==1.9.1

torchvision==0.10.1

⚙️ Installation
Clone the repository and install the required dependencies using the following commands:

git clone https://github.com/Priyanshu88/Diabetes-Prediction-Streamlit-App.git
cd Diabetes-Prediction-Streamlit-App
pip install -r requirements.txt
streamlit run app.py
⏯️ Usage
Open the app in your web browser.
Enter the required information in the input fields.
Click the 'Predict' button to generate the prediction.
🚧 Inputs
Click on the link and reboot the tool or run locally and enter your:

Name: Name of the patient
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
Blood Pressure: Diastolic blood pressure (mm Hg)
Skin Thickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
Diabetes Pedigree Function: Diabetes pedigree function
Age: Age (years)
🚀 Outputs
The app will display one of the following messages:

"Congratulations! [Name], you are not diabetic."
"[Name], we are really sorry to say but it seems like you are Diabetic. But don't lose hope, we have suggestions for you." along with a link to the Mayo Clinic's Diabetes Prevention page.
🚩 Deployment and Notebook
This tool has been deployed using Streamlit. Learn about streamlit deployment here. Checkout the notebook repository here from where the pickle file has been imployed in the tool.

⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contact

Project Link: https://webappdiabetesprediction-4evzguq48xdhlkdsnsmxzj.streamlit.app/
