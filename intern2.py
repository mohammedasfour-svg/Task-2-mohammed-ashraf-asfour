from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size= 0.2,
    shuffle= True,
    random_state= 42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors= 5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)* 100:.2f}%")
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, average= 'weighted')}")
print(f"Recall: {recall_score(y_test, y_pred, average= 'weighted')}")
print(f"F1 score: {f1_score(y_test, y_pred, average= 'weighted')}")