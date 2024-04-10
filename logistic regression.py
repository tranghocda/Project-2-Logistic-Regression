import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cr_loan_clean = pd.read_csv('/Users/batch/Desktop/python/study/cr_loan_clean.csv')

## One-hot encode the non-numeric columns

# Create two data sets for numeric and non-numeric data
cred_num = cr_loan_clean.select_dtypes(exclude=['object'])
cred_str = cr_loan_clean.select_dtypes(include=['object'])

# One-hot encode the non-numeric columns
cred_str_onehot = pd.get_dummies(cred_str)

# Union the one-hot encoded columns to the numeric ones
cr_loan_modeling = pd.concat([cred_num,cred_str_onehot], axis=1)

print(cr_loan_modeling.columns)

## Create the training and test sets
X = cr_loan_modeling.drop('loan_status', axis = 1)
y = cr_loan_modeling[['loan_status']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.4, random_state = 123)

# Train the logistic regression model on the training data
logistic = LogisticRegression(solver='lbfgs',max_iter=20000).fit(X_train, np.ravel(y_train))

# Print the coefficients of the model
print(logistic.coef_)

# Print the accuracy score the model
print(logistic.score(X_test, y_test))

# Create predictions of probability for loan status using test data
prediction = logistic.predict_proba(X_test)

# Create a dataframe for the probabilities of default
prediction_df = pd.DataFrame(prediction[:,1], columns = ['prob_default'])

# Set the threshold for defaults to 0.5 and print the confusion matrix
prediction_df["loan_status"] = prediction_df["prob_default"].apply(lambda x: 1 if x > 0.5 else 0)
print(confusion_matrix(y_test,prediction_df['loan_status']))

# Set the threshold for defaults to 0.4 and print the confusion matrix
prediction_df["loan_status"] = prediction_df["prob_default"].apply(lambda x: 1 if x > 0.4 else 0)
print(confusion_matrix(y_test,prediction_df["loan_status"]))

## Choose the 0.4 threshold as the recall(Default) of 0.4 threshold > that of 0.5 threshold.

# Print the row counts for each loan status
print(prediction_df["loan_status"].value_counts())

# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, prediction_df['loan_status'], target_names=target_names))