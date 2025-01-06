import os
import uuid
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
from io import BytesIO
from utils.middleware import token_required
import model_backend as model

transaction_bp = Blueprint('transaction', __name__)

# In-memory transaction store
transactions = {}

# Endpoint to upload an image
@transaction_bp.route('/upload', methods=['POST'])
@token_required
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(image.filename)
    transaction_id = str(uuid.uuid4())
    file_path = os.path.join('uploads', transaction_id + '_' + filename)

    image.save(file_path)

    # Store transaction details
    transactions[transaction_id] = {'file_path': file_path, 'processed': False}

    return jsonify({'transaction_id': transaction_id}), 201

from flask import request

# Endpoint to process an image by transaction ID
@transaction_bp.route('/segment', methods=['POST'])
@token_required
def process_transaction():
    data = request.get_json()
    if not data or 'transaction_id' not in data:
        return jsonify({'error': 'Transaction ID is required in the request body'}), 400

    transaction_id = data['transaction_id']
    transaction = transactions.get(transaction_id)
    if not transaction:
        return jsonify({'error': 'Transaction not found'}), 404

    if not transaction['processed']:
        # Process the image
        corp_file_path = os.path.join(os.getcwd(), 'segments', transaction_id + '_corp.png')
        segment_img = model.segment_input_img(transaction['file_path'])
        detect_text = model.read_number_eazyocr(segment_img)
        file_saved = model.save_segment_img(segment_img, corp_file_path)

        # Save the segmented image and detected text
        transactions[transaction_id]['processed'] = True
        transactions[transaction_id]['segment_img'] = segment_img
        transactions[transaction_id]['result'] = detect_text

        if file_saved and os.path.exists(corp_file_path):
            return send_file(
                corp_file_path,
                mimetype='image/png',
                as_attachment=True,
                download_name=f"detect_test:{detect_text}.png"
            ), 200, {"detected_text": detect_text}

        return jsonify({'transaction_id': transaction_id, "detected_text": detect_text}), 200

    return jsonify({
        'transaction_id': transaction_id,
        'message': 'Transaction already processed',
        'detected_text': transaction['result']
    }), 200

# Endpoint to process an image by transaction ID
@transaction_bp.route('/easyocr', methods=['POST'])
@token_required
def process_transaction_ocr():
    data = request.get_json()
    if not data or 'transaction_id' not in data:
        return jsonify({'error': 'Transaction ID is required in the request body'}), 400

    transaction_id = data['transaction_id']
    transaction = transactions.get(transaction_id)
    if not transaction:
        return jsonify({'error': 'Transaction not found'}), 404

    if not transaction['processed']:
        # Process the image
        detect_text = model.read_number_eazyocr(transaction['file_path'])
        # Save the segmented image and detected text
        transactions[transaction_id]['processed'] = True
        transactions[transaction_id]['result'] = detect_text

        return jsonify({'transaction_id': transaction_id, "detected_text": detect_text}), 200

    return jsonify({
        'transaction_id': transaction_id,
        'message': 'Transaction already processed',
        'detected_text': transaction['result']
    }), 200
