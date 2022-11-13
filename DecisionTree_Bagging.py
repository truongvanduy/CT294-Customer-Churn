from data import X_train, X_test, y_train, y_test, evaluate
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Train
BC_model = BaggingClassifier(n_estimators=50, random_state=99)
BC_model.fit(X_train, y_train)

# Predict
y_pred = BC_model.predict(X_test)

# Evaluate
evaluate(y_test, y_pred)

# Accuracy:  93.79
# [[951  17]
#  [ 52  91]]
