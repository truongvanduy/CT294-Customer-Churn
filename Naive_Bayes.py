from data import X_train, X_test, y_train, y_test, evaluate
from sklearn.naive_bayes import GaussianNB

# Train
NB_model = GaussianNB()
NB_model.fit(X_train, y_train)

# Predict
y_pred = NB_model.predict(X_test)

# Evaluate
evaluate(y_test, y_pred)

# Accuracy:  86.5
# [[896  72]
#  [78  65]]
