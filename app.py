from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model('artifacts//gender_recog_model.keras')

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def homepage():
    return render_template('index.html')

@app.route("/detect", methods=['POST'])
def recognize():
    imgfile = request.files['imag']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imgfile.filename)
    imgfile.save(image_path)
    
    img = load_img(image_path, target_size=(224, 224))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    
    pred = model.predict(img_arr)

    msg = 'The person above is a female' if pred > 0.5 else 'The person above is a male'
    
    return render_template('index.html', text=msg, img_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
