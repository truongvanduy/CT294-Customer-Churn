from data import X_train, X_test, y_train, y_test, evaluate
from sklearn.tree import DecisionTreeClassifier

# Train
DT_model = DecisionTreeClassifier(
    criterion="gini", random_state=99, max_depth=10, min_samples_leaf=2)
DT_model.fit(X_train, y_train)

# Predict
y_pred = DT_model.predict(X_test)

# Evaluate
evaluate(y_test, y_pred)

# Accuracy:  91.9
# [[931  37]
#  [ 53  90]]
