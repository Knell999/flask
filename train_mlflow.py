import mlflow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 데이터 로드 및 분할
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 모델 딕셔너리
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(C=0.001),
    "KNN": KNeighborsClassifier(n_neighbors=2),
    "Decision Tree": DecisionTreeClassifier(max_depth=2),
    "Random Forest": RandomForestClassifier(n_estimators=5, max_depth=5)
}

# MLflow 실험 설정
mlflow.set_experiment("Iris_Model_Experiment")

# 자동 로깅 설정
mlflow.sklearn.autolog()

# 각 모델 학습
for name, model in models.items():
    with mlflow.start_run():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
				
				# 테스트 정확도는 따로 저장함.
        mlflow.log_metric("test_accuracy", accuracy)

        print(f"Model: {name}, Accuracy: {accuracy}")