# MLFlow에서 관리되는 모델 중 test_accuracy가 가장 높은 모델을 찾는 코드

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

best_run = client.search_runs(
    experiment_ids=['318557162735373948'],
    order_by=["metrics.test_accuracy desc"], # test_accuracy 기준으로 내림차순 정렬
    max_results=1, # 가장 높은 정확도 1개만 가져옴
)[0] # 가장 높은 정확도 1개만 가져옴

# best_run에 가장 성능이 좋은 모델의 메타데이터 정보가 들어있다.
#   best_run에서 실제 모델을 찾아서 로딩까지
print(best_run.info.run_id)

# MLFlow 서버에 요청하기 위한 uri 생성
model_uri = f"runs:/{best_run.info.run_id}/model"

# 모델 로딩
loaded_model = mlflow.pyfunc.load_model(model_uri)
print(loaded_model)