from flask import Flask, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# load the model on startup
model = tf.keras.models.load_model('models/damage_model.keras')

@app.route('/summary', methods=['GET'])
def model_summary():
    return {
        "name": "damage_classifier",
        "version": "v1",
        "description": "Custom CNN classification model for detecting building damage using satellite images",
        "input_shape": [128, 128, 3],
        "number_of_parameters": model.count_params()
    }

@app.route('/inference', methods=['POST'])
def classify_damage_image():
    if 'image' not in request.files:
        # simple error check if user did not pass image
        return '{"error": "Invalid request; pass a binary image file as a multi-part form under the image key."}'
    
    # gets the data
    data = request.files['image']
    
    try:
        temp_path = '/tmp/temp_image.jpg'
        data.save(temp_path)
        
        img = load_img(temp_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0  # normalize
        
        img_array = np.expand_dims(img_array, axis=0)
        
        result = model.predict(img_array)
        
        prediction = "damage" if result[0][0] > 0.5 else "no_damage"
        
        return {"prediction": prediction}
        
    except Exception as e:
        return {"error": f"Could not process the image; details: {e}"}, 400


# start dev server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')