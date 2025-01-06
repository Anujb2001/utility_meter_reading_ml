from flask import request, jsonify
from functools import wraps
from utils.jwt_utils import validate_jwt

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        try:
            token = token.split(" ")[1]  # Bearer <token>
            if not validate_jwt(token):
                raise ValueError
        except (IndexError, ValueError):
            return jsonify({'error': 'Invalid or missing token!'}), 401
        return f(*args, **kwargs)
    return decorated_function
