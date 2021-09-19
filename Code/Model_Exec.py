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


#img = cv2.imread("./ISIC_0029313.jpg")
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#Define model for prediction
def create_model():

    model = Sequential()
    model.add(mobilenet_model)
    model.add(Dropout(dropout_dense))
    model.add(BatchNormalization())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout_dense))
    model.add(BatchNormalization())
    model.add(Dense(7, activation="softmax"))
    loaded_model = model.load_model("model.h5")
    loaded_model.summary()
    return model

def load_trained_model():
    model = create_model()
    loaded_model = model.load_model("model.h5") #Load saved model
    print(loaded_model.summary())
    model.layers[0].input_shape #(1,224,224,3)
    #batch_holder = np.zeros((20,224,224,3))
    pic_loc = "./ISIC_0029313.jpg"
    pic = image.load_img(pic_loc, target_size=(224,224))
    plt.imshow(pic)
    pic = np.expand_dims(pic, axis=0)
    result = loaded_model.predict_classes(pic) #predict image using saved model
    plt.title(get_label_name(result[0][0]))
    plt.show()
    plt.waitKey(0)
#loaded_model.layers[0].input_shape(None, 244, 244, 3)
#pic = image.load_img(img, target_size = (224,224))
#pic_array = image.img_to_array(pic)
#pic_batch = np.expand_dims(pic_array, axis=0)

#pic_pre = preprocess_input(pic_batch)

#model = load_model("./model.h5")
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("Done!")
