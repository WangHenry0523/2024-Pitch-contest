from flask import Flask, request, render_template, redirect, url_for, send_file
from datetime import datetime
import os
# import inference_resnet as inference
import inference_new as inference
app = Flask(__name__)
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['RESULT_FOLDER']):
    os.makedirs(app.config['RESULT_FOLDER'])

@app.route("/")
@app.route("/index")
def index():
    print(f'Homepage')
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def upload():
    print(f'Upload')
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = timestamp + '.' + file.filename.split('.')[-1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        save_path1 = os.path.join(app.config['RESULT_FOLDER'], timestamp + '_1.jpg')
        save_path2 = os.path.join(app.config['RESULT_FOLDER'], timestamp + '_2.jpg')
        save_path3 = os.path.join(app.config['RESULT_FOLDER'], timestamp + '_3.jpg')
        try:
            file.save(file_path)
            print(f'File saved')
        except Exception as e:
            print(f'ERROR: {e}')
        # 這裡呼叫 inference.py
        result = inference.predict_images(file_path, save_path1, save_path2, save_path3)
        result['image_url'] = file_path
        result['result_url_1'] = save_path1
        result['result_url_2'] = save_path2
        result['result_url_3'] = save_path3
        return render_template('result.html', result = result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000')

