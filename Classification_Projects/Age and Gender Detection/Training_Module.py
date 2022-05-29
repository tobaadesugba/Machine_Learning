import tensorflow as tf


class AgeGenderModel(object):
    # Model Initialisation
    def __init__(self, inputs, input_dim, output_size, data, dropout_val=0.4):
        self.inputs = inputs
        self.input_dim = input_dim
        self.output_size = output_size
        self.data = data
        self.dropout_val = dropout_val

    # CNN Layers
    def model_layers(self):
        model: object = tf.keras.models.Sequential()

        # Convolutional Layer #1
        model.add(tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            activation='relu',
            padding='same',
            name='conv1'
        ))

        # Pooling Layer #1
        model.add(tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            strides=2,
            name='pool1'
        ))

        # Convolutional Layer #2
        model.add(tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation='relu',
            name='conv2'
        ))

        # Pooling Layer #2
        model.add(tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            strides=2,
            name='pool2'
        ))

        # Convolutional Layer #3
        model.add(tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[5, 5],
            padding='same',
            activation='relu',
            name='conv1'
        ))

        # Pooling Layer #3
        model.add(tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            strides=2,
            name='pool3'
        ))

        # Convolutional Layer #4
        model.add(tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[5, 5],
            padding='same',
            activation='relu',
            name='conv4'
        ))

        # Pooling Layer #4
        model.add(tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            strides=2,
            name='pool4'
        ))

        # Flattening Pool Layer
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            1024,
            activation='relu',
            name='dense'
        ))

        # Applying Dropout
        model.add(tf.keras.layers.Dropout(
            rate=self.dropout_val
        ))

        # setting logits layer
        # use sigmoid function for binary class. in gender data
        if self.data == 'gender':
            model.add(tf.keras.layers.Dense(
                self.output_size,
                activation='sigmoid',
                name='logits'
            ))
            model.summary()
            return model

        # use softmax function for multi class. in age data
        model.add(tf.keras.layers.Dense(
            self.output_size,
            activation='softmax',
            name='logits'
        ))
        model.summary()
        return model

    def train_model(self, model, train_data, val_data, epochs=100):
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

        # compile the model
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy'])

        # begin training and store history in 'history'
        # set callback for early stopping of training
        callback: object = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[callback])
        return (history, model)
