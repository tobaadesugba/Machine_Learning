# import necessary files
import tensorflow as tf
from os import getcwd


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        """ Builds a custom model that returns the style and content tensors

        Attributes:
            self.vgg (tf.keras.models.Model) representing the vgg model with only the intermediate layers
            self.style_layers (list of layers) a list of style layers from the intermediate layers
            self.content_layers (list of layers) a list of content layers from the intermediate layers
            self.num_style_layers (int) total number of intermediate style layers
        """
        super(StyleContentModel, self).__init__()
        model_dir = "models//vgg_model.h5"
        self.vgg_model = tf.keras.models.load_model(model_dir, compile=False)  # load no-head model
        self.vgg = self.vgg_intermediate_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False  # freeze model layers

    def gram_matrix(self, input_tensor):
        """ Calculates gram matrix which describes the means and correlations across the
            different intermediate feature maps representing the content of an image.

        Args:
            input_tensor (tf.Tensor): input image as tensor.

        Returns:
            float: calculated style
        """

        # implement gram matrix equation
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def vgg_intermediate_layers(self, layer_names):
        """Creates a model that returns the list of intermediate layer output values

        Args:
            layer_names (list): list of names of the intermediate layers.

        Returns:
            tf.keras.Model: model with intermediate output values
        """
        self.vgg_model.trainable = False  # freeze output layers of model

        outputs = [self.vgg_model.get_layer(name).output for name in layer_names]  # get output_name of each layer

        model = tf.keras.Model([self.vgg_model.input], outputs)  # create functional model with input and output arguments
        return model

    def __call__(self, inputs):
        """Expects float input between 0 and 1"

        Args:
            inputs (tf.Tensor): image inputs

        Returns:
            dict: dictionary containing content and style tensor dictionaries
        """
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        # separating style and content layers from model outputs
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        # applying gram matrix on style layer outputs
        style_outputs = [self.gram_matrix(style_output)
                         for style_output in style_outputs]

        # zip content layer names and outputs into dictionary
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        # zip style layer names and outputs into dictionary
        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
