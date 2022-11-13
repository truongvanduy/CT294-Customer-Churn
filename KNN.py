from data import X_train, X_test, y_train, y_test, evaluate
from sklearn.neighbors import KNeighborsClassifier

# Train
KNN_model = KNeighborsClassifier(n_neighbors=11)
KNN_model.fit(X_train, y_train)

# Predict
y_pred = KNN_model.predict(X_test)

# Evaluate
evaluate(y_test, y_pred)

# Accuracy:  88.93
# [[949  19]
#  [104  39]]
