import cvzone
import cv2 
import tensorflow
import numpy as np

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
cap = cv2.VideoCapture(0)

model = tensorflow.keras.models.load_model("model/keras_model.h5") 

"""
the closer to "1" the predicted value is, more precise the model.
Android is the defaulting value, so if there's no matching result, the prediction
defaults to android.
"""


while True:
    ret, frame = cap.read()
    cv2.imshow("Window", frame)
    frame = cv2.resize(frame, (224, 224))
    image_array = np.asarray(frame)
    normalised_image_array = (image_array.astype(np.float32)/127.0) - 1
    data[0] = normalised_image_array
    pv = model.predict(data)
    print("Prediction:", f"iPhone: {pv[0][0]}\tAndroid: {pv[0][1]}", sep="\n")
    key = cv2.waitKey(1)
    if key == (ord('q')):
        break
cv2.destroyAllWindows()
cap.release()
