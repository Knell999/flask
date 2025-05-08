from flask import Flask
import json

app = Flask(__name__)

# app.route : 어디로 요청을 하게 할 것인지
@app.route("/", methods=["GET"]) # methods : 어떻게 요청 하게 할 것인지
def index():

    # return : 요청에 대한 응답을 어떻게 할 것인지
    return "<h1>Hello, World!</h1>"

@app.route("/hello")
def hello():
    return {"Hello": "World!"}

# POST 방식
@app.route("/predict", methods=["POST"])
def inference():
    return json.dumps({"Hello": "World!"}), 200

if __name__ == "__main__":
    # 디버그 모드로 실행하며, 모든 IP에서 접근 허용, 포트 5000으로 실행
    #    debug=True로 설정하면 코드를 수정할 때마다 자동으로 서버가 재시작됨.
    app.run(debug=True, host="0.0.0.0.", port=5000)