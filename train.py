import random
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils import pad_sequences
import keras.backend as K

from model import CRNN
from utils import preprocess_img, encode_to_labels, char_list

MAX_IMAGES = 150000

training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

# lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

max_label_len = 0

annot = open('data generate/annotation.txt', 'r').readlines()
imagenames = []
txts = []

for cnt in annot:
    filename, txt = cnt.split(',')[0], cnt.split(',')[1].split('\n')[0]
    imagenames.append(filename)
    txts.append(txt)

c = list(zip(imagenames, txts))

random.shuffle(c)

imagenames, txts = zip(*c)

for i in range(len(imagenames)):
    img = cv2.imread('data generate/images/' + imagenames[i], 0)

    img = preprocess_img(img, (128, 32))
    img = np.expand_dims(img, axis=-1)
    img = img / 255.
    txt = txts[i]

    # compute maximum length of the text
    if len(txt) > max_label_len:
        max_label_len = len(txt)

    # split the 150000 data into validation and training dataset as 10% and 90% respectively
    if i % 10 == 0:
        valid_orig_txt.append(txt)
        valid_label_length.append(len(txt))
        valid_input_length.append(31)
        valid_img.append(img)
        valid_txt.append(encode_to_labels(txt))
    else:
        orig_txt.append(txt)
        train_label_length.append(len(txt))
        train_input_length.append(31)
        training_img.append(img)
        training_txt.append(encode_to_labels(txt))

        # break the loop if total data is equal to MAX_IMAGES
    if i == MAX_IMAGES:
        flag = 1
        break
    i += 1

# pad each output label to maximum text length
train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value=len(char_list))
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value=len(char_list))

CRNN_model = CRNN(max_label_len)
model = CRNN_model.get_model()
act_model = CRNN_model.get_act_model()

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

filepath = "best_model.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)

batch_size = 256
epochs = 15
model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length],
          y=np.zeros(len(training_img)),
          batch_size=batch_size, epochs=epochs,
          validation_data=([valid_img, valid_padded_txt, valid_input_length, valid_label_length],
                           [np.zeros(len(valid_img))]), verbose=1, callbacks=callbacks_list)


# test the model

# load the saved best model weights
act_model.load_weights('best_model.hdf5')

# predict outputs on validation images
prediction = act_model.predict(valid_img[10:20])

# use CTC decoder
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                               greedy=True)[0][0])

# see the results
i = 10
for x in out:
    print("original_text =  ", valid_orig_txt[i])
    print("predicted text = ", end='')
    for p in x:
        if int(p) != -1:
            print(char_list[int(p)], end='')
    print('\n')
    i += 1