import streamlit as st
import numpy as np
import pickle5 as pickle

# pickle_in = open('Diabetes.pkl', 'rb')
pickle_in = open('diabetes-prediction-rfc-model.pkl', 'rb')
classifier = pickle.load(pickle_in)

pickle_in = open('diabetes-prediction-rfc-model2.pkl', 'rb')
classifier2 = pickle.load(pickle_in)

# pickle_in = open('diabetes-prediction-rfc-model3.pkl', 'rb')
# classifier3 = pickle.load(pickle_in)

pickle_in = open('diabetes-prediction-rfc-model4.pkl', 'rb')
classifier4 = pickle.load(pickle_in)

pickle_in = open('diabetes-prediction-rfc-model5.pkl', 'rb')
classifier5 = pickle.load(pickle_in)


import streamlit as st
import plotly.express as px


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

def predict():

    st.sidebar.header('Predict Diabetes')
    # select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
    # if not st.sidebar.checkbox("Hide", True, key='2'):
    st.title('Diabetes Prediction(Only for Females Above 21 Years of Age)')
    st.markdown('This trained dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.')
    st.markdown('Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.')

    name = st.text_input("Name:")
    pregnancy = st.text_input("No. of times pregnant:")
    st.markdown('Pregnancies: Number of times pregnant')

    glucose = st.text_input("Plasma Glucose Concentration :")
    st.markdown('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test')

    bp =  st.text_input("Diastolic blood pressure (mm Hg):")
    st.markdown('BloodPressure: Diastolic blood pressure (mm Hg)')

    skin = st.text_input("Triceps skin fold thickness (mm):")
    st.markdown('SkinThickness: Triceps skin fold thickness (mm)')

    insulin = st.text_input("2-Hour serum insulin (mu U/ml):")
    st.markdown('Insulin: 2-Hour serum insulin (mu U/ml)')


    bmi = st.text_input("Body mass index (weight in kg/(height in m)^2):")
    st.markdown('BMI: Body mass index (weight in kg/(height in m)^2)')

    dpf = st.text_input("Diabetes Pedigree Function:")
    st.markdown('DiabetesPedigreeFunction: Diabetes pedigree function')


    age = st.text_input("Age:")
    st.markdown('Age: Age (years)')


    submit = st.button('Predict')
    st.markdown('Outcome: Class variable (0 or 1)')



    if submit:
        input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = classifier.predict(input_data_reshaped)
        print(prediction)
        prediction = classifier.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
        if prediction == 0:
            st.write('Congratulation!',name,'You are not diabetic')
        else:
            st.write(name,", we are really sorry to say but it seems like you are Diabetic. But don't lose hope, we have suggestions for you:")
            st.markdown('[Visit Here](https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/in-depth/diabetes-prevention/art-20047639#:~:text=Diabetes%20prevention%3A%205%20tips%20for%20taking%20control%201,Skip%20fad%20diets%20and%20make%20healthier%20choices%20)')

def predict2():

    st.sidebar.header('Predict Diabetes')
    # select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
    # if not st.sidebar.checkbox("Hide", True, key='2'):
    st.title('Diabetes Prediction(Only for Females Above 21 Years of Age)')
    st.markdown('This trained dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.')
    st.markdown('Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.')

    name = st.text_input("Name:")
    pregnancy = st.text_input("No. of times pregnant:")
    st.markdown('Pregnancies: Number of times pregnant')

    glucose = st.text_input("Plasma Glucose Concentration :")
    st.markdown('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test')

    bp =  st.text_input("Diastolic blood pressure (mm Hg):")
    st.markdown('BloodPressure: Diastolic blood pressure (mm Hg)')

    skin = st.text_input("Triceps skin fold thickness (mm):")
    st.markdown('SkinThickness: Triceps skin fold thickness (mm)')

    insulin = st.text_input("2-Hour serum insulin (mu U/ml):")
    st.markdown('Insulin: 2-Hour serum insulin (mu U/ml)')


    bmi = st.text_input("Body mass index (weight in kg/(height in m)^2):")
    st.markdown('BMI: Body mass index (weight in kg/(height in m)^2)')

    dpf = st.text_input("Diabetes Pedigree Function:")
    st.markdown('DiabetesPedigreeFunction: Diabetes pedigree function')


    age = st.text_input("Age:")
    st.markdown('Age: Age (years)')


    submit = st.button('Predict')
    st.markdown('Outcome: Class variable (0 or 1)')



    if submit:
        input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = classifier2.predict(input_data_reshaped)
        print(prediction)
        prediction = classifier2.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
        if prediction == 0:
            st.write('Congratulation!',name,'You are not diabetic')
        else:
            st.write(name,", we are really sorry to say but it seems like you are Diabetic. But don't lose hope, we have suggestions for you:")
            st.markdown('[Visit Here](https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/in-depth/diabetes-prevention/art-20047639#:~:text=Diabetes%20prevention%3A%205%20tips%20for%20taking%20control%201,Skip%20fad%20diets%20and%20make%20healthier%20choices%20)')

def predict3():

    st.sidebar.header('Predict Diabetes')
    # select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
    # if not st.sidebar.checkbox("Hide", True, key='2'):
    st.title('Diabetes Prediction(Only for Females Above 21 Years of Age)')
    st.markdown('This trained dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.')
    st.markdown('Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.')

    name = st.text_input("Name:")
    pregnancy = st.text_input("No. of times pregnant:")
    st.markdown('Pregnancies: Number of times pregnant')

    glucose = st.text_input("Plasma Glucose Concentration :")
    st.markdown('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test')

    bp =  st.text_input("Diastolic blood pressure (mm Hg):")
    st.markdown('BloodPressure: Diastolic blood pressure (mm Hg)')

    skin = st.text_input("Triceps skin fold thickness (mm):")
    st.markdown('SkinThickness: Triceps skin fold thickness (mm)')

    insulin = st.text_input("2-Hour serum insulin (mu U/ml):")
    st.markdown('Insulin: 2-Hour serum insulin (mu U/ml)')


    bmi = st.text_input("Body mass index (weight in kg/(height in m)^2):")
    st.markdown('BMI: Body mass index (weight in kg/(height in m)^2)')

    dpf = st.text_input("Diabetes Pedigree Function:")
    st.markdown('DiabetesPedigreeFunction: Diabetes pedigree function')


    age = st.text_input("Age:")
    st.markdown('Age: Age (years)')


    submit = st.button('Predict')
    st.markdown('Outcome: Class variable (0 or 1)')



    if submit:
        input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = classifier3.predict(input_data_reshaped)
        print(prediction)
        prediction = classifier3.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
        if prediction == 0:
            st.write('Congratulation!',name,'You are not diabetic')
        else:
            st.write(name,", we are really sorry to say but it seems like you are Diabetic. But don't lose hope, we have suggestions for you:")
            st.markdown('[Visit Here](https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/in-depth/diabetes-prevention/art-20047639#:~:text=Diabetes%20prevention%3A%205%20tips%20for%20taking%20control%201,Skip%20fad%20diets%20and%20make%20healthier%20choices%20)')

def predict4():

    st.sidebar.header('Predict Diabetes')
    # select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
    # if not st.sidebar.checkbox("Hide", True, key='2'):
    st.title('Diabetes Prediction(Only for Females Above 21 Years of Age)')
    st.markdown('This trained dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.')
    st.markdown('Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.')

    name = st.text_input("Name:")
    pregnancy = st.text_input("No. of times pregnant:")
    st.markdown('Pregnancies: Number of times pregnant')

    glucose = st.text_input("Plasma Glucose Concentration :")
    st.markdown('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test')

    bp =  st.text_input("Diastolic blood pressure (mm Hg):")
    st.markdown('BloodPressure: Diastolic blood pressure (mm Hg)')

    skin = st.text_input("Triceps skin fold thickness (mm):")
    st.markdown('SkinThickness: Triceps skin fold thickness (mm)')

    insulin = st.text_input("2-Hour serum insulin (mu U/ml):")
    st.markdown('Insulin: 2-Hour serum insulin (mu U/ml)')


    bmi = st.text_input("Body mass index (weight in kg/(height in m)^2):")
    st.markdown('BMI: Body mass index (weight in kg/(height in m)^2)')

    dpf = st.text_input("Diabetes Pedigree Function:")
    st.markdown('DiabetesPedigreeFunction: Diabetes pedigree function')


    age = st.text_input("Age:")
    st.markdown('Age: Age (years)')


    submit = st.button('Predict')
    st.markdown('Outcome: Class variable (0 or 1)')



    if submit:
        input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = classifier4.predict(input_data_reshaped)
        print(prediction)
        prediction = classifier4.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
        if prediction == 0:
            st.write('Congratulation!',name,'You are not diabetic')
        else:
            st.write(name,", we are really sorry to say but it seems like you are Diabetic. But don't lose hope, we have suggestions for you:")
            st.markdown('[Visit Here](https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/in-depth/diabetes-prevention/art-20047639#:~:text=Diabetes%20prevention%3A%205%20tips%20for%20taking%20control%201,Skip%20fad%20diets%20and%20make%20healthier%20choices%20)')

def predict5():

    st.sidebar.header('Predict Diabetes')
    # select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
    # if not st.sidebar.checkbox("Hide", True, key='2'):
    st.title('Diabetes Prediction(Only for Females Above 21 Years of Age)')
    st.markdown('This trained dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.')
    st.markdown('Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.')

    name = st.text_input("Name:")
    pregnancy = st.text_input("No. of times pregnant:")
    st.markdown('Pregnancies: Number of times pregnant')

    glucose = st.text_input("Plasma Glucose Concentration :")
    st.markdown('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test')

    bp =  st.text_input("Diastolic blood pressure (mm Hg):")
    st.markdown('BloodPressure: Diastolic blood pressure (mm Hg)')

    skin = st.text_input("Triceps skin fold thickness (mm):")
    st.markdown('SkinThickness: Triceps skin fold thickness (mm)')

    insulin = st.text_input("2-Hour serum insulin (mu U/ml):")
    st.markdown('Insulin: 2-Hour serum insulin (mu U/ml)')


    bmi = st.text_input("Body mass index (weight in kg/(height in m)^2):")
    st.markdown('BMI: Body mass index (weight in kg/(height in m)^2)')

    dpf = st.text_input("Diabetes Pedigree Function:")
    st.markdown('DiabetesPedigreeFunction: Diabetes pedigree function')


    age = st.text_input("Age:")
    st.markdown('Age: Age (years)')


    submit = st.button('Predict')
    st.markdown('Outcome: Class variable (0 or 1)')



    if submit:
        input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = classifier5.predict(input_data_reshaped)
        print(prediction)
        prediction = classifier5.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
        if prediction == 0:
            st.write('Congratulation!',name,'You are not diabetic')
        else:
            st.write(name,", we are really sorry to say but it seems like you are Diabetic. But don't lose hope, we have suggestions for you:")
            st.markdown('[Visit Here](https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/in-depth/diabetes-prevention/art-20047639#:~:text=Diabetes%20prevention%3A%205%20tips%20for%20taking%20control%201,Skip%20fad%20diets%20and%20make%20healthier%20choices%20)')



def main():
    new_title = '<p style="font-size: 42px;">Welcome The Diabetes Prediction App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
    read_me = st.markdown("""
    The application is built using Streamlit to demonstrate Diabetes Prediction. It performs prediction on multiple parameters
    Objective is to predict whether the person has Diabetes or not based on various features suach as Pregnancies, Insulin Level, Age, BMI.
                          
                          
    The motivation was to experiment with end to end machine learning project and get some idea about deployment platform like Streamlit and 
    "Diabetes is an increasingly growing health issue due to our inactive lifestyle. If it is detected in time then through proper medical treatment,
    adverse effects can be prevented
    [GIT HUB](https://github.com/Om123xyz).""")
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox(
        "Select ML Model from Dropdown", ("About", "Predict Diabetes Using KNN", "Predict Diabetes Using Random Forest", "Predict Diabetes Using Decision Tree","Predict Diabetes Using XgBoost", "Predict Diabetes Using Logistic Regression"))
    if choice == "Predict Diabetes Using KNN":
        read_me_0.empty()
        read_me.empty()
        predict4()
    elif choice == "Predict Diabetes Using Random Forest":
        read_me_0.empty()
        read_me.empty()
        predict()
    elif choice == "Predict Diabetes Using Decision Tree":
        read_me_0.empty()
        read_me.empty()
        predict2()
    elif choice == "Predict Diabetes Using XgBoost":
        read_me_0.empty()
        read_me.empty()
        predict3()
    elif choice == "Predict Diabetes Using Logistic Regression":
        read_me_0.empty()
        read_me.empty()
        predict5()   
    elif choice == "About":
        print()


if __name__ == '__main__':
    main()
