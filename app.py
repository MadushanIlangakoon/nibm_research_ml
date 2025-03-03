from flask_cors import CORS
from flask import Flask
from Inference import inference_bp
from test_questions import test_questions
from general_questions import general_questions

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/inference/*": {"origins": "https://nibm-research-frontend.onrender.com"}})

# Register blueprint
app.register_blueprint(inference_bp, url_prefix='/inference')
app.register_blueprint(general_questions, url_prefix='/general_questions')
app.register_blueprint(test_questions, url_prefix='/test_questions')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
