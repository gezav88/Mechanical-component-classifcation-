import numpy as np
import os
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import keras 
import pickle
import json
from utils.util_funcs import*

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.obj', '.off']
app.config['UPLOAD_PATH'] = 'uploads'

# load model based on version:
model_version_to_load = "model_v1"
model = keras.models.load_model("models/"+model_version_to_load)

# load mapping dicts:
load_class_map_full = load_dict("data/class_map_full")
load_class_map_reduced = load_dict("data/class_map_reduced")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    print(uploaded_file)
    filename = secure_filename(uploaded_file.filename)
    print(filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    #file_to_class = [str(x) for x in request.form.values()]
    file = request.files['file']
    filename = str(secure_filename(file.filename))
    full_path = os.path.abspath(filename)
    try:
        mesh = trimesh.load("uploads/"+filename)
        points = mesh.sample(2048)
        full_classes = load_dict("data/class_map_full")
        map_class_reduced = load_dict("data/class_map_reduced")
        
        array_points = np.array(points).reshape(1, 2048, 3)

        classification = model.predict(array_points)
        classification = tf.math.argmax(classification, -1)
        print(classification)

        # remap from dictionaries:
        name_class_map = dict((full_classes.get(k, k), v) for (k, v) in map_class_reduced.items())
        reversed_dictionary = {value : key for (key, value) in name_class_map.items()}

        output = reversed_dictionary[classification[0].numpy()]
        print(output)
        return render_template('index.html', classification_text="The object is classified as  :   {}".format(output))
    except:
        upload_files()
        return redirect(url_for('index'))
        
    return render_template('index.html')
    
    

if __name__ == "__main__":
    app.run(debug=True)