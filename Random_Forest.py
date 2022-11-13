from data import X_train, X_test, y_train, y_test, evaluate
from sklearn.ensemble import RandomForestClassifier

# Train
RF_model = RandomForestClassifier(
    n_estimators=50, criterion="gini", random_state=99, max_depth=10, min_samples_leaf=2)
RF_model.fit(X_train, y_train)

# Predict
y_pred = RF_model.predict(X_test)

# Evaluate
evaluate(y_test, y_pred)

# Accuracy:  94.69
# [[956  12]
#  [ 47  96]]
