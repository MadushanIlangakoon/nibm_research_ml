from flask_cors import CORS
from flask import Flask
from Inference import inference_bp
from test_questions import test_questions
from general_questions import general_questions
from archive import archive_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/inference/*": {"origins": "http://localhost:3000"}})

# Register blueprints
app.register_blueprint(inference_bp, url_prefix='/inference')
app.register_blueprint(general_questions, url_prefix='/general_questions')
app.register_blueprint(test_questions, url_prefix='/test_questions')
app.register_blueprint(archive_bp, url_prefix='/api/archive')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
