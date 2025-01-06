# Backend Project

## Overview
This project is a Flask-based backend application that provides an API for image processing and text detection. The system allows users to upload images, processes the images, and extracts text using advanced image processing techniques.

---

## Features
- **Token-Based Authentication**: Secure endpoints with JWT tokens.
- **Image Upload**: Upload images for processing.
- **Image Segmentation and Text Detection**: Segment images and detect text using pre-trained models.
- **Response Handling**: Return segmented images and detected text in responses.
- **CORS Enabled**: Allow cross-origin requests.

---

## Prerequisites
1. Python 3.8+
2. `pip` (Python package manager)
3. Virtual environment tool (`venv` or similar)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/backend-project.git
cd backend-project
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the root directory:
```
SECRET_KEY=your_secret_key
UPLOAD_FOLDER=uploads
```

### 5. Create Necessary Directories
```bash
mkdir uploads
mkdir segments
```

---

## Running the Application
```bash
python src/app.py
```

The server will start on `http://localhost:5000`.

---

## API Endpoints

### 1. **Get Token**
   **Endpoint**: `/token`
   **Method**: `GET`
   **Description**: Retrieve a JWT token.

   **Response**:
   ```json
   {
       "token": "<your_token_here>"
   }
   ```

### 2. **Upload Image**
   **Endpoint**: `/upload`
   **Method**: `POST`
   **Headers**:
   - `Authorization: Bearer <your_token>`

   **Body**:
   - `image` (multipart/form-data): The image file to upload.

   **Response**:
   ```json
   {
       "transaction_id": "<transaction_id_here>"
   }
   ```

### 3. **Process Image**
   **Endpoint**: `/segment`
   **Method**: `POST`
   **Headers**:
   - `Authorization: Bearer <your_token>`

   **Body**:
   ```json
   {
       "transaction_id": "<transaction_id_here>"
   }
   ```

   **Response**:
   - Returns a segmented image file and detected text.
   ```json
   {
       "detected_text": "Detected text from the image."
   }
   ```

---

## Project Structure
```
backend-project/
├── src/
│   ├── app.py           # Main Flask app
│   ├── routes/
│   │   ├── __init__.py  # Routes package initialization
│   │   ├── auth.py      # Authentication routes
│   │   ├── upload.py    # Image upload routes
│   │   └── segment.py   # Image processing routes
│   ├── model_backend/   # ML model utilities
│   ├── utils.py         # Utility functions
│   └── ...
├── uploads/             # Directory for uploaded images
├── segments/            # Directory for processed images
├── requirements.txt     # Dependencies
├── .env                 # Environment variables
└── README.md            # Project documentation
```

---

## Testing
1. Use tools like Postman or curl to test the endpoints.
2. Ensure the token is included in the `Authorization` header for secure routes.

---

## Requirements
See `requirements.txt` for all dependencies.

---

## License
This project is licensed under the MIT License.

