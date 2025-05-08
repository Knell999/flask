from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

data_diabetes = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
        data_diabetes.data, data_diabetes.target, random_state=1
)

if __name__ == "__main__":
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    predictions = clf.predict(X_test)
    score = accuracy_score(predictions, y_test)
    
    print("Score : {}".format(score))
    saved_model = joblib.dump(clf, "./model.pkl")