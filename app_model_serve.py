from flask import Flask, request, jsonify

import joblib # pkl로 저장된 모델을 불러오기 위한 라이브러리

# 저장한 모델 불러오기
MODEL_PATH = "./model.pkl"
loaded_model = joblib.load(MODEL_PATH)

app = Flask(__name__)
	
@app.route("/predict", methods=["POST", "PUT"])
def inference():
    # request : 클라이언트의 요청 정보가 들어있는 객체
    data = request.get_json()['data'] # 클라이언트가 요청하는 Feature를 받아옴

    prediction = loaded_model.predict(data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0.', port=5000)