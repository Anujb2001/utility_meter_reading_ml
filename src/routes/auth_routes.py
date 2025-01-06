from flask import Blueprint, request, jsonify
from utils.jwt_utils import generate_jwt

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user_id = data.get('userId')
    password = data.get('password')

    # Check if credentials match
    if user_id == 'user123' and password == 'P@ssw0rd123':
        token = generate_jwt()  # You should define the generate_jwt function
        return jsonify({'token': token}), 200
    else:
        return jsonify({'message': 'Invalid credentials'}), 401