import jwt
import os
from datetime import datetime, timedelta

SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')

def generate_jwt():
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {'exp': expiration}
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def validate_jwt(token):
    try:
        jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return True
    except jwt.ExpiredSignatureError:
        return False
    except jwt.InvalidTokenError:
        return False
