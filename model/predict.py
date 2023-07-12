import os
import numpy as np
import cv2
import pathlib
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


# Input image's path
path_img = "C:\\Users\\ACER\\Pattern Regconition\\VN_handwriting_OCR\\Vietnamese-Handwriting-Recognition-OCR-main\\vn_handwritten_images\\data_outside\\2.png"

def read_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

    # in this dataset, we don't need to do any resize at all here.
    img = cv2.resize(img,(2167,118))

    #print(str(val_img_path) + ": "+ str(width))
    img = np.pad(img, ((0,0),(0, 0)), 'median')

    # YOUR PART: Blur it
    img = cv2.GaussianBlur(img, (5,5), 0)

    # YOUR PART: Threshold the image using adapative threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # add channel dimension
    img = np.expand_dims(img , axis = 2)

    # Normalize each image
    img = img/255.

    # expand dimension
    img = tf.expand_dims(img, axis=0)
    return img

char_list = " #'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"

# OUR FULL MODEL OF CRNN AND LSTM


# input with shape of height=32 and width=128
inputs = Input(shape=(118,2167,1))

# Block 1
x = Conv2D(64, (3,3), padding='same')(inputs)
x = MaxPool2D(pool_size=3, strides=3)(x)
x = Activation('relu')(x)
x_1 = x

# Block 2
x = Conv2D(128, (3,3), padding='same')(x)
x = MaxPool2D(pool_size=3, strides=3)(x)
x = Activation('relu')(x)
x_2 = x

# Block 3
x = Conv2D(256, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x_3 = x

# Block4
x = Conv2D(256, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x,x_3])
x = Activation('relu')(x)
x_4 = x

# Block5
x = Conv2D(512, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x_5 = x

# Block6
x = Conv2D(512, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x,x_5])
x = Activation('relu')(x)

# Block7
x = Conv2D(1024, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(3, 1))(x)
x = Activation('relu')(x)

# pooling layer with kernel size (2,2) to make the height/2 #(1,9,512)
x = MaxPool2D(pool_size=(3, 1))(x)

# # to remove the first dimension of one: (1, 31, 512) to (31, 512)
squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)

# # # bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(blstm_1)

# # this is our softmax character proprobility with timesteps
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# model to be used at test time

act_model = Model(inputs, outputs)

# load the saved best model weights
act_model.load_weights(os.path.join(f'checkpoint_weights.h5'))

prediction = act_model.predict(read_image(path_img))

# use CTC decoder
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                        greedy=True)[0][0])

y=0
for x in out:
    print("predicted text = ", end = '')
    pred = ""
    for p in x:
        if int(p) != -1:
            pred += char_list[int(p)]
    print(pred)
    y+=1