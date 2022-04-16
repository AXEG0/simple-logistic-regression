from sklearn.datasets import load_iris

X = load_iris().data
y = load_iris().target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=23)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000).fit(X_train, y_train)

import pandas as pd

prediction = model.predict(X_test)
pred_10 = pd.DataFrame(X_test[0:10], columns=[["sepal length", "sepal width", "petal length", "petal width"]])
pred_10["predicted"] = prediction[0:10]
pred_10["real"] = y_test[0:10]
print(pred_10)

from sklearn.metrics import precision_score

precision = precision_score(y_test, prediction, average="macro")
print("Precision:", round(precision, 2))
