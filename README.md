# Ex.No.4b--MACHINE-LEARNING-MODEL-HEART-DISEASE-PREDICTION
## AIM
To develop a Heart Disease Prediction model using machine learning classification algorithms and compare the performance of different models using suitable evaluation metrics.
##  OBJECTIVES
•	To understand machine learning classification. 
•	To analyze a heart disease dataset. 
•	To identify the input features and target variable. 
•	To preprocess the dataset. 
•	To divide the dataset into training and testing data. 
•	To train different classification models. 
•	To predict whether a patient has heart disease. 
•	To evaluate and compare the models. 
## INTRODUCTION
•	Machine Learning enables computers to learn patterns from data and make predictions. 
•	Classification is a supervised learning technique used to predict categories or classes. 
•	In this experiment, classification algorithms are used to predict whether a patient is likely to have heart disease. 
•	The output generally contains two classes: 
o	0 – No Heart Disease 
o	1 – Heart Disease 
## DATASET
The dataset contains medical information about patients.
Typical attributes include:
Attribute	Description
Age	Age of the patient
Sex	Gender of the patient
Chest Pain	Type of chest pain
Resting BP	Resting blood pressure
Cholesterol	Cholesterol level
Fasting Blood Sugar	Blood sugar condition
Resting ECG	Resting electrocardiogram result
Maximum Heart Rate	Maximum heart rate achieved
Exercise Angina	Exercise-induced angina
Oldpeak	ST depression value
Target	Presence or absence of heart disease
## TARGET VARIABLE
•	Target is the dependent variable. 
•	It indicates whether the patient has heart disease. 
•	Usually: 
0 → No Heart Disease
1 → Heart Disease
6. DATA PREPROCESSING
### Steps
1.	Load the dataset. 
2.	Display the first few records. 
3.	Check dataset shape. 
4.	Check data types. 
5.	Check missing values. 
6.	Handle missing values if present. 
7.	Separate features and target. 
8.	Encode categorical variables if required. 
9.	Split the dataset into training and testing data. 
10.	Apply feature scaling where required. 
### Code
import pandas as pd
import numpy as np

df = pd.read_csv("heart_disease.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
7. SEPARATE INPUT AND OUTPUT
X = df.drop("target", axis=1)
y = df["target"]
•	X → Patient characteristics. 
•	y → Heart disease prediction. 
###  TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
•	80% → Training data 
•	20% → Testing data 
## FEATURE SCALING
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
## MACHINE LEARNING MODELS
The following models can be compared:
1.	Logistic Regression 
2.	K-Nearest Neighbors 
3.	Decision Tree 
4.	Random Forest 
5.	Support Vector Machine 
6.	Gradient Boosting 
## LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
##  K-NEAREST NEIGHBORS
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
## DECISION TREE
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
## RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
## SUPPORT VECTOR MACHINE
from sklearn.svm import SVC

svm = SVC(kernel="rbf")

svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)
## GRADIENT BOOSTING
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    random_state=42
)

gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
## MODEL EVALUATION
The models can be evaluated using:
•	Accuracy 
•	Precision 
•	Recall 
•	F1-score 
•	Confusion Matrix 
## Code
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

models = {
    "Logistic Regression": y_pred_lr,
    "KNN": y_pred_knn,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "SVM": y_pred_svm,
    "Gradient Boosting": y_pred_gb
}

for name, prediction in models.items():

    print(name)
    print("Accuracy :", accuracy_score(y_test, prediction))
    print("Precision:", precision_score(y_test, prediction))
    print("Recall   :", recall_score(y_test, prediction))
    print("F1 Score :", f1_score(y_test, prediction))
    print()
## MODEL COMPARISON
<img width="591" height="191" alt="image" src="https://github.com/user-attachments/assets/6bb058f3-4231-475d-92d1-2e64233ac6a0" />
Comparison
•	Higher Accuracy → Better overall classification. 
•	Higher Precision → Fewer false-positive predictions. 
•	Higher Recall → Better identification of patients with heart disease. 
•	Higher F1-score → Better balance between precision and recall. 
For a medical prediction task, recall is particularly important because missing a patient who actually has the condition can be more concerning than generating an extra positive prediction.
## CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")

plt.show()

## THEORY

Machine Learning Classification is a supervised learning technique used to predict a category or class based on given input data. Heart Disease Prediction is an important application of machine learning in healthcare, where a model is trained to predict whether a person is likely to have heart disease based on different health-related features.

A heart disease dataset may contain features such as age, blood pressure, cholesterol level, chest pain type, maximum heart rate, blood sugar level, and other clinical parameters. These features are provided as input to the machine learning model, while the target variable represents the presence or absence of heart disease.

In this experiment, different classification algorithms such as Logistic Regression, Decision Tree, and Random Forest are used to develop prediction models. The dataset is divided into training and testing data. The training data is used to learn patterns from the dataset, while the testing data is used to evaluate the trained models.

The performance of classification models can be evaluated using suitable metrics such as Accuracy, Precision, Recall, F1-Score, and Confusion Matrix. Accuracy measures the overall number of correct predictions. Precision measures how many predicted positive cases are actually positive. Recall measures how many actual positive cases are correctly identified. F1-Score provides a balance between precision and recall.

Comparing multiple classification models helps in identifying which model performs better for the given dataset.

## WORKING PRINCIPLE

The heart disease dataset is first loaded into Python and the required preprocessing steps are performed. The input features are separated from the target variable, which indicates whether heart disease is present or absent.

The dataset is divided into training and testing sets. Different classification algorithms are then trained using the training dataset. Each model learns the relationship between the input health-related features and the target class.

After training, the models are tested using unseen testing data. Each model predicts whether the given input represents a positive or negative heart disease class.

The predicted results are compared with the actual results using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix. The performance of the different classification algorithms is then compared, and the model with suitable evaluation results can be selected.

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/e05c0ddb-f5e2-49e7-a7dc-104e2a18bb0d" />

## RESULT
Thus, a Heart Disease Prediction model using machine learning classification algorithms was developed successfully. Logistic Regression, Decision Tree, and Random Forest models were trained and their performance was compared using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix. The model with the most suitable evaluation performance can be selected for the given dataset.
