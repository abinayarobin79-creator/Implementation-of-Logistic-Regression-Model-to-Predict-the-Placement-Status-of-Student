# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries 
2. Load the Placement_Data.csv dataset
3. Copy the dataset and preprocess
4. Convert categorical variables to numerical values
5. Convert categorical variables to numerical values
6. Split dataset into features and target variable
7. Split the dataset into training and testing sets
8. Train the logistic regression model
9. Predict the output on test data
10.Evaluate the model

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Abinaya R
RegisterNumber:  212225230004
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv(r'C:\Users\acer\Downloads\Placement_Data.csv')
le_workex = LabelEncoder()
data['workex'] = le_workex.fit_transform(data['workex'])
le_spec = LabelEncoder()
data['specialisation'] = le_spec.fit_transform(data['specialisation'])
le_status = LabelEncoder()
data['status'] = le_status.fit_transform(data['status'])
X = data[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'workex', 'specialisation']]
y = data['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
new_student = [[85.8, 73.6, 73.3, 96.8, 55.5, 0, 0]] 
placement_prediction = model.predict(new_student)
if placement_prediction[0] == 1:
    print("The student is likely to be Placed.")
else:
    print("The student is likely Not Placed.")
  
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

<img width="334" height="80" alt="Screenshot 2026-02-06 111700" src="https://github.com/user-attachments/assets/bc394782-3fb2-4876-8f5e-1c31e4fe0b6b" />


<img width="552" height="318" alt="Screenshot 2026-02-06 111712" src="https://github.com/user-attachments/assets/3d275991-4a61-4b0c-98bd-d4421b5d0b00" />


<img width="365" height="43" alt="Screenshot 2026-02-06 111719" src="https://github.com/user-attachments/assets/5eea2838-c0f8-4e41-9b12-058ac65d680f" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
