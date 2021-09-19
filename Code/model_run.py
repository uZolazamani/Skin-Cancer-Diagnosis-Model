from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout, Flatten)
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

print (tf.__version__)
def catch():
    
    # allow the camera to warmup
    time.sleep(0.3)
    vid = cv2.VideoCapture(0) 
  
    while(True): 
      
    # Capture the video frame 
    # by frame 
        ret, frame = vid.read() 
  
    # Display the resulting frame 
        cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice to quit and capture an image
    
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    vid.release()
    cv2.destroyAllWindows()
    
    #Capture Image After Orientation
    camera = PiCamera()
    rawCapture = PiRGBArray(camera)
    
    # grab an image from the camera
    image = camera.capture(rawCapture, format="rgb")
    image = rawCapture.array
    img = cv2.imwrite('./image1.jpg', image)
    # display the image on screen and wait for a keypress
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
    



def create_model():

    model = Sequential()
    model.add(mobilenet_model)
    model.add(Dropout(dropout_dense))
    model.add(BatchNormalization())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout_dense))
    model.add(BatchNormalization())
    model.add(Dense(7, activation="softmax"))
    return model

def load_trained_model():
    model = create_model()
    model.load_model("./model.h5") #Load saved model
    #loaded_model.summary()
    model.layers[0].input_shape #(1,224,224,3)
    #batch_holder = np.zeros((20,224,224,3))
    pic_loc = "./image1.jpg"
    pic = image.load_img(pic_loc, target_size=(224,224))
    plt.imshow(pic)
    pic = np.expand_dims(pic, axis=0)
    result = loaded_model.predict_classes(pic) #predict image using saved model
    plt.title(get_label_name(result[0][0]))
    plt.show()
    
#loaded_model.layers[0].input_shape(None, 244, 244, 3)
#pic = image.load_img(img, target_size = (224,224))
#pic_array = image.img_to_array(pic)
#pic_batch = np.expand_dims(pic_array, axis=0)

#pic_pre = preprocess_input(pic_batch)

#model = load_model("./model.h5")
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("Done!")
