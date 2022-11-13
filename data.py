import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

churn_data = pd.read_csv("telecom_churn.csv")

data = churn_data.iloc[:, 1:11]
target = churn_data.Churn
# print(data)
# print(target)

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=1/3.0, random_state=100)


def evaluate(y_test, y_pred):
    accuracy = 100 * accuracy_score(y_test, y_pred)
    cfs_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Accuracy: ", np.round(accuracy, 2))
    print(cfs_matrix)
