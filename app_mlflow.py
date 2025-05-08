from flask import Flask, request, jsonify
import mlflow.pyfunc
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

app = Flask(__name__)


# MLflow 설정 및 최고 성능 모델 로드
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()
best_run = client.search_runs(
    experiment_ids=['318557162735373948'],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)[0]

print(best_run)

model_uri = f"runs:/{best_run.info.run_id}/model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

@app.route("/predict", methods=["POST", "PUT"])
def inference():
    data = request.get_json()['data']
    prediction = loaded_model.predict(np.array(data))
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=5001)