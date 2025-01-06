from flask import Flask
from flask_cors import CORS
import os
# Import blueprints
from routes.auth_routes import auth_bp
from routes.transaction_routes import transaction_bp
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configurations
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key')
app.config['UPLOAD_FOLDER'] = 'uploads'

# Enable CORS
CORS(app)

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(transaction_bp)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    print("hello world")
    return "Hello world"
    
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
