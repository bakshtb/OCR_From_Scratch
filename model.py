from keras import Input, Model
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Lambda, Bidirectional, LSTM, Dense
import keras.backend as K

from utils import char_list


class CRNN:
    def __init__(self, max_label_len):
        # CRNN model
        self.inputs = Input(shape=(32, 128, 1))

        # convolution layer with kernel size (3,3)
        self.conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(self.inputs)
        # poolig layer with kernel size (2,2)
        self.pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(self.conv_1)

        self.conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(self.pool_1)
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(self.conv_2)

        self.conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(self.pool_2)

        self.conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(self.conv_3)
        # poolig layer with kernel size (2,1)
        self.pool_4 = MaxPool2D(pool_size=(2, 1))(self.conv_4)

        self.conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(self.pool_4)
        # Batch normalization layer
        self.batch_norm_5 = BatchNormalization()(self.conv_5)

        self.conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(self.batch_norm_5)
        self.batch_norm_6 = BatchNormalization()(self.conv_6)
        self.pool_6 = MaxPool2D(pool_size=(2, 1))(self.batch_norm_6)

        self.conv_7 = Conv2D(512, (2, 2), activation='relu')(self.pool_6)

        self.squeezed = Lambda(lambda x: K.squeeze(x, 1))(self.conv_7)

        # bidirectional LSTM layers with units=128
        self.blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(self.squeezed)
        self.blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(self.blstm_1)

        self.outputs = Dense(len(char_list) + 1, activation='softmax')(self.blstm_2)

        # model to be used at test time
        self.act_model = Model(self.inputs, self.outputs)

        # add CTC layer
        self.labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')

        # CTC layer declaration using lambda.
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([self.outputs, self.labels, self.input_length, self.label_length])

        # Including the CTC layer to train the model.
        self.model = Model(inputs=[self.inputs, self.labels, self.input_length, self.label_length], outputs=loss_out)


    def ctc_lambda_func(self, args):
        # Defining the CTC loss.
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def get_act_model(self):
        return self.act_model

    def get_model(self):
        return self.model
