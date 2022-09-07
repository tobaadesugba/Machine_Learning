# import necessary files
import os
import tensorflow as tf
from .custom_model_module import StyleContentModel
from .image_processing_module import ImageProcessing
import streamlit as st


# load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


class TransferStyle():
    def __init__(self, content_path, style_path, intensity: int = 70, quality: int = 256):
        self.content_image = ImageProcessing().load_image(path_to_image=content_path, max_dim=quality)
        self.style_image = ImageProcessing().load_image(path_to_image=style_path, max_dim=quality)
        self.steps_per_epoch = intensity
        # vgg model layer names for content and style
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1']
        self.extractor = StyleContentModel(self.style_layers, self.content_layers)  # define extractor model

    def style_content_loss(self, model_outputs):
        """Uses mean squared error to calculate the loss for image's output relative to
            each target and taking the weighted sum of the losses

        Args:
            model_outputs (dicts): dictionary output of the model

        Returns:
            float: mean squared error loss
        """

        # number of content layers
        num_content_layers = len(self.content_layers)
        # number of style layers
        num_style_layers = len(self.style_layers)

        # set style and content target values
        style_targets = self.extractor(self.style_image)['style']
        content_targets = self.extractor(self.content_image)['content']

        style_outputs = model_outputs['style']
        content_outputs = model_outputs['content']

        # use weighted combination of style and content weights to get total loss
        style_weight = 1e-2
        content_weight = 1e4

        # calculate mean squared error for all style outputs
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        # calculate mean squared error for all content outputs
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        return style_loss + content_loss  # return weighted sum of losses

    @tf.function
    def training_step(self, image, total_variation_weight=30):
        """Function to train the model and update with the specified optimizer"

        Args:
            image (tf.Tensor): image inputs

        Returns: None
        """
        # use tf.gradienttape to update the image
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss((outputs))
            # add regularization
            loss += total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)  # apply tf.GradientTape
        self.optimizer.apply_gradients([(grad, image)])  # apply optimizer
        image.assign(ImageProcessing().clip_0_to_1(image))  # apply effect on image

    def __call__(self):
        # defining a tf.variable to hold the content image
        image = tf.Variable(self.content_image)

        # create an optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        # run optimization
        import time
        start = time.time()

        epochs = 10

        step = 0
        for epoch in range(epochs):
            for step_count in range(self.steps_per_epoch):
                step += 1
                self.training_step(image)

        end = time.time()
        st.write("")
        st.info(f"Training steps: {step} steps")
        st.info("Total time spent: {:.1f} seconds".format(end - start))

        return ImageProcessing().tensor_to_image(image)
