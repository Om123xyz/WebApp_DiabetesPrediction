# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Loading the dataset
diabetes_df = pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
#df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
diabetes_df_copy = diabetes_df.copy(deep = True)
diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution
diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].mean(), inplace = True)
diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].mean(), inplace = True)
diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace = True)
diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].median(), inplace = True)
diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace = True)

sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes_df_copy.drop(["Outcome"],axis = 1),), columns=['Pregnancies', 
'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()

# Model Building

X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=7)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc_train = rfc.predict(X_train)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)

from xgboost import XGBClassifier
xgb_model = XGBClassifier(gamma=0)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

knn = KNeighborsClassifier()
knn = KNeighborsClassifier(n_neighbors=5,weights='uniform', p=2)
knn.fit(X_train,y_train)
y_predict=knn.predict(X_test)

from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression(max_iter = 100)
# fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

# from sklearn import metrics
# print("Accuracy_Score =", format(metrics.accuracy_score(y_train, rfc_train)))
# predictions = rfc.predict(X_test)
# print("Accuracy_Score =", format(metrics.accuracy_score(y_test, predictions)))


# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(rfc, open(filename, 'wb'))

filename = 'diabetes-prediction-rfc-model2.pkl'
pickle.dump(dtree, open(filename, 'wb'))

filename = 'diabetes-prediction-rfc-model3.pkl'
pickle.dump(xgb_model, open(filename, 'wb'))

filename = 'diabetes-prediction-rfc-model4.pkl'
pickle.dump(knn, open(filename, 'wb'))

filename = 'diabetes-prediction-rfc-model5.pkl'
pickle.dump(logreg, open(filename, 'wb'))

# X_train_prediction = classifier.predict(X_train)
# training_data_accuracy = accuracy_score(X_train_prediction, y_train)
# print('Accuracy score of the training data : ', training_data_accuracy)