from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess
from werkzeug.utils import secure_filename
from tshirt import process_upload  # Import the process_upload function

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        measurements = process_upload(filepath)
        return jsonify(measurements)
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/run-tshirt-script', methods=['POST'])
def run_tshirt_script():
    try:
        result = subprocess.run(
            ['python', os.path.join(os.path.dirname(__file__), 'tshirt.py')],
            capture_output=True, text=True
        )
        print("Script stdout:", result.stdout)  # Log stdout
        print("Script stderr:", result.stderr)  # Log stderr
        if result.returncode == 0:
            return jsonify({'output': result.stdout})
        else:
            return jsonify({'error': result.stderr}), 500
    except Exception as e:
        print("Exception:", str(e))  # Log exceptions
        return jsonify({'error': str(e)}), 500

@app.route('/run-glasses-script', methods=['POST'])
def run_glasses_script():
    try:
        result = subprocess.run(['python', os.path.join(os.path.dirname(__file__), 'new.py')],
                                capture_output=True, text=True)
        return jsonify({'output': result.stdout})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/run-watch-script', methods=['POST'])
def run_watch_script():
    try:
        result = subprocess.run(['python', os.path.join(os.path.dirname(__file__), 'wrist_detection.py')],
                                capture_output=True, text=True)
        return jsonify({'output': result.stdout})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/virtual-tryon', methods=['POST'])
def run_virtual_tryon_script():
    try:
        result = subprocess.run(['python', os.path.join(os.path.dirname(__file__), 'newcode.py')],
                                capture_output=True, text=True)
        return jsonify({'output': result.stdout})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
