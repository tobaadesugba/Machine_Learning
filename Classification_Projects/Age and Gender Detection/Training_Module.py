import tensorflow as tf
from datetime import datetime


class AgeGenderModel(object):
    # model Initialisation
    def __init__(self, output_size, data, input_dim=(64, 64, 3), dropout_val=0.4):
        self.data = data # gender or age

        # CNN Layers
        self.model: object = tf.keras.models.Sequential()

        # Convolutional Layer #1
        self.model.add(tf.keras.layers.Conv2D(
                        filters=32,
                        kernel_size=[5, 5],
                        activation='relu',
                        padding='same',
                        input_shape=input_dim,
                        name='conv1'
        ))

        # Pooling Layer #1
        self.model.add(tf.keras.layers.MaxPool2D(
                        pool_size=[2, 2],
                        strides=2,
                        name='pool1'
        ))

        # Convolutional Layer #2
        self.model.add(tf.keras.layers.Conv2D(
                        filters=64,
                        kernel_size=[5, 5],
                        padding='same',
                        activation='relu',
                        name='conv2'
        ))

        # Pooling Layer #2
        self.model.add(tf.keras.layers.MaxPool2D(
                        pool_size=[2, 2],
                        strides=2,
                        name='pool2'
        ))

        # Convolutional Layer #3
        self.model.add(tf.keras.layers.Conv2D(
                        filters=128,
                        kernel_size=[5, 5],
                        padding='same',
                        activation='relu',
                        name='conv1'
        ))

        # Pooling Layer #3
        self.model.add(tf.keras.layers.MaxPool2D(
                        pool_size=[2, 2],
                        strides=2,
                        name='pool3'
        ))

        # Convolutional Layer #4
        self.model.add(tf.keras.layers.Conv2D(
                        filters=128,
                        kernel_size=[5, 5],
                        padding='same',
                        activation='relu',
                        name='conv4'
        ))

        # Pooling Layer #4
        self.model.add(tf.keras.layers.MaxPool2D(
                        pool_size=[2, 2],
                        strides=2,
                        name='pool4'
        ))

        # Flattening Pool Layer
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(
                        1024,
                        activation='relu',
                        name='dense'
        ))

        # Applying Dropout
        self.model.add(tf.keras.layers.Dropout(rate=dropout_val))

        # setting logits layer
        # use sigmoid function for binary class. in gender data
        if self.data == 'gender':
            self.model.add(tf.keras.layers.Dense(
                output_size,
                activation='sigmoid',
                name='logits'
            ))
            self.model.summary()
            return

        # use softmax function for multi class. in age data
        self.model.add(tf.keras.layers.Dense(
            output_size,
            activation='softmax',
            name='logits'
        ))
        self.model.summary()
        return

    def train_model(self, train_data, val_data, epochs=100):
        # check that model has been defined
        if self.model is None:
            print("error: model has not been defined")
            return

        # set configuration
        optimizer = 'adam'

        # use sparse categorical for age data
        if self.data == 'age':
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif self.data == 'gender':
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            print("null or invalid data selected")
            return

        # compile the self.model
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy'])

        # begin training and store history in 'history'
        # set callback for early stopping of training
        earlystopping_callback: object = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        # set callback for tensorboard debugging
        logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        history: object = self.model.fit(train_data,
                                         epochs=epochs,
                                         verbose=0,
                                         validation_data=val_data,
                                         callbacks=[earlystopping_callback, tensorboard_callback])
        return history

    def evaluate_model(self, test_data):
        # only evaluate if the model has been defined
        if self.model is not None:
            # return loss, accuracy
            return self.model.evaluate(test_data, verbose=2)
        else:
            print("error: model has not been defined")
            return

    def model_predict(self, data):
        return self.model.predict(data)
