# Flask Api for the project, this is the main file,
# The API should be able to take in an image as input and return the predicted mask for the image

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import base64 # for encoding and decoding the image
import segmentation_models as sm
import cv2
import numpy as np

IMG_SIZE = 192


linknet_augment = sm.Linknet(classes=8)
linknet_augment.compile( optimizer='adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
linknet_augment._name = 'Linknet_augmented_input'
linknet_augment.load_weights('linknet_augment.h5')


app = Flask(__name__)
api = Api(app)

def predict_mask(image):
    # This function takes in an image and return the predicted mask for the image
    img_bytes = base64.b64decode(image)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
    image = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    pred = linknet_augment.predict(image)
    pred = pred.reshape(IMG_SIZE, IMG_SIZE, 8)
    return pred



class Predict(Resource):
    def get(self):
        return {"data":"Hello World"}




    def post(self):
        request_data = request.get_json()
        # we read the image from the request
        image = request_data["image"]
        # we predict the mask
        mask = predict_mask(image)
        # we encode the mask
        encoded_mask = base64.b64encode(mask)
        # we return the mask
        return {"mask":encoded_mask.decode("utf-8")}

api.add_resource(Predict, "/")

if __name__ == "__main__":
    app.run(debug=True)

