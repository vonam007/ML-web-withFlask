import os
import numpy as np
import cv2
import pathlib
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
# import our model, different layers and activation function 
from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

char_list = " #'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
# convert the words to array of indexs based on the char_list
def encode_to_labels(txt):
    # encoding each output word into digits of indexes
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print("No found in char_list :", char)

    return dig_lst

# our raw folders
current_directory_path = "./vn_handwritten_images"
RAW_FOLDER = os.path.join(str(current_directory_path),"raw")
# label json files
TRAIN_JSON = os.path.join(current_directory_path, "labels.json")


# open these label jsons files
with open(TRAIN_JSON, 'r', encoding='utf8') as f:
    train_labels = json.load(f)

#lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

#Matching full absoluate paths and labels instead of filenames and labels
dict_filepath_label={}
raw_data_path = pathlib.Path(os.path.join(RAW_FOLDER))
for item in raw_data_path.glob('**/*.*'):
    file_name=str(os.path.basename(item))
    if file_name != "labels.json":
      label = train_labels[file_name]
      dict_filepath_label[str(item)]=label

all_image_paths = list(dict_filepath_label.keys())

# being a good Data Scientist, we need to have train set and test set
test_size = 0.2
train_image_paths, val_image_paths = train_test_split(all_image_paths, test_size=test_size, random_state=42)

# Constants (we choose this number as we later discover that the output of our CNN is (1,9,512) from out input)
TIME_STEPS = 240
resize_max_width=0

i=0

for val_img_path in val_image_paths:
    # print(f_name)
    # read input image and convert into gray scale image
    img = cv2.cvtColor(cv2.imread(val_img_path), cv2.COLOR_BGR2GRAY)

    height, width = (118, 2167)

    # in this dataset, we don't need to do any resize at all here.
    img = cv2.resize(img,(2167,118))

    if img.shape[1] > resize_max_width:
        resize_max_width = img.shape[1]

    #print(str(val_img_path) + ": "+ str(width))
    img = np.pad(img, ((0,0),(0, 2167-width)), 'median')

    # YOUR PART: Blur it
    img = cv2.GaussianBlur(img, (5,5), 0)

    # YOUR PART: Threshold the image using adapative threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # add channel dimension
    img = np.expand_dims(img , axis = 2)

    # Normalize each image
    img = img/255.

    label = dict_filepath_label[val_img_path]

    valid_orig_txt.append(label)
    valid_label_length.append(len(label))

    # our time steps for valid input
    valid_input_length.append(TIME_STEPS)

    # expand dimension
    img = tf.expand_dims(img, axis=0)

    valid_img.append(img)

    # convert words to digits based on charlist
    valid_txt.append(encode_to_labels(label))
    i+=1
    if (i%500 == 0):
        print ("has processed test {} files".format(i))
#print(resize_max_width)

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

# predict outputs on validation images
NO_PREDICTS = 100
OFFSET=0
# prediction = act_model.predict(valid_img[OFFSET:OFFSET+NO_PREDICTS])

print(len(valid_img))

def cer(ref, hyp):
    """
    Computes the Character Error Rate (CER) between two strings.
 
    Arguments:
    ref -- the reference string
    hyp -- the hypothesis string
 
    Returns:
    The character error rate as a float.
    """
 
    # Create a matrix of zeros
    d = [[0 for j in range(len(hyp)+1)] for i in range(len(ref)+1)]
 
    # Initialize the first row and column of the matrix
    for i in range(len(ref)+1):
        d[i][0] = i
    for j in range(len(hyp)+1):
        d[0][j] = j
 
    # Fill in the rest of the matrix
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
 
    # Return the CER
    return d[len(ref)][len(hyp)] / float(len(ref))

predictions = []
y = 0
ori = []
hypo = []
sum_cer = 0
for i in valid_img:
    prediction = act_model.predict(i)


    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                            greedy=True)[0][0])

    # see the results
    all_predictions =[]

    for x in out:
        print("original_text  = ", valid_orig_txt[y+OFFSET])
        print("predicted text = ", end = '')
        ori.append(valid_orig_txt[y+OFFSET])
        
        pred = ""
        for p in x:
            if int(p) != -1:
                pred += char_list[int(p)]
        print(pred)
        all_predictions.append(pred)
        hypo.append(pred)
        y+=1
        print(" CER là " + str(1-cer(ori, hypo)))
        sum_cer += cer(ori,hypo)


print("Average CER là ", str(1-sum_cer/len(valid_img)))
def wer(ref, hyp ,debug=True):
    r = ref.split()
    h = hyp.split()
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
 
    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
    DEL_PENALTY = 1
    INS_PENALTY = 1
    SUB_PENALTY = 1
    
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL
    
    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS
    
    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1
                
                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL
                
    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    # return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    return {'WER':wer_result, 'numCor':numCor, 'numSub':numSub, 'numIns':numIns, 'numDel':numDel, "numCount": len(r)}

def wers(refs, hyps):
    numSub = 0
    numDel = 0
    numCor = 0
    numIns = 0
    numCount = 0
    for ref, hyp in zip(refs, hyps):
        result = wer(ref, hyp, False)
        numSub += result["numSub"]
        numDel += result["numDel"]
        numCor += result["numCor"]
        numIns += result["numIns"]
        numCount += result["numCount"]
 
    return round( (numSub + numDel + numIns) / (float) (numCount), 3)

print("WER là ", str(wers(ori, hypo)))

def cer(ref, hyp):
    """
    Computes the Character Error Rate (CER) between two strings.
 
    Arguments:
    ref -- the reference string
    hyp -- the hypothesis string
 
    Returns:
    The character error rate as a float.
    """
 
    # Create a matrix of zeros
    d = [[0 for j in range(len(hyp)+1)] for i in range(len(ref)+1)]
 
    # Initialize the first row and column of the matrix
    for i in range(len(ref)+1):
        d[i][0] = i
    for j in range(len(hyp)+1):
        d[0][j] = j
 
    # Fill in the rest of the matrix
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
 
    # Return the CER
    return d[len(ref)][len(hyp)] / float(len(ref))