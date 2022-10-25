# import necessary libraries
import numpy as np
import tensorflow as tf
from PIL import Image


class ImageProcessing():
    """ Contains sets of functions for loading and transforming images"""

    def __init__(self):
        pass

    def tensor_to_image(self, tensor):
        """Converts tensors to PIL image format

        Args:
            tensor (tf.Tensor): tensor object to be converted to PIL image

        Returns:
            PIL.Image: output image
        """

        # limit image pixel values to 255
        tensor = tensor * 255

        # convert tensor to numpy array for easy manipulation
        array_tensor = np.array(tensor, dtype=np.uint8)

        # 3-D: [height, width, channels]
        # 4-D: [batch, height, width, channels]
        if np.ndim(array_tensor) > 3:  # if the tensor has more than 3 dimensions,
            assert array_tensor.shape[0] == 1  # ensure it has only 1 batch
            array_tensor = array_tensor[0]  # and replace with the batch.

        return Image.fromarray(array_tensor)  # return image converted from tensor

    def load_image(self, path_to_image, max_dim=256):
        """A function that loads an image and limit its maximum dimension to max_dim to load into the custom model

        Args:
            path_to_image (str): A filename (string), pathlib.Path object or a file object.
            max_dim (int): Maximum dimension to use to load the image

        Returns:
            tf.Tensor: If images was 4-D, a 4-D float Tensor of shape [batch, new_height, new_width, channels].
            If images was 3-D, a 3-D float Tensor of shape [new_height, new_width, channels]
        """
        max_dim = max_dim
        # img = Image.open(path_to_image)  # open image file, not necessary since it's in app.py
        img = tf.keras.utils.img_to_array(path_to_image)  # convert image to numpy array
        img = tf.cast(img, tf.uint8)  # convert image from int to unsigned int 8bits
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)  # cast image height and width shape as float
        long_dim = max(shape)  # store max dim of shape (one of height or width)
        scale = max_dim / long_dim  # get stated scale of maximum dimension to the dimension of the image

        new_shape = tf.cast(shape * scale, tf.int32)  # scale up the image dimensions with 'scale'

        img = tf.image.resize(img, new_shape)  # resize image with new image shape
        img = img[tf.newaxis, :]  # expanding dimension by adding a new axis at the beginning of the tensor

        return img

    def load_output_image(self, image):
        """Loads an image and limit its maximum dimension to 256 pixels for output viewing

        Args:
            img (tf.Tensor): tensor object to be outputted to PIL image

        Returns:
            PIL.Image: output image
        """
        img = tf.keras.utils.img_to_array(image)  # convert image to numpy array
        img = tf.cast(img, tf.uint8)  # convert image from int to unsigned int 8bits
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)  # cast image height and width shape as float
        long_dim = max(shape)  # store max dim of shape (one of height or width)
        scale = 256 / long_dim  # get stated scale of maximum dimension to the dimension of the image
        new_shape = tf.cast(shape * scale, tf.int32)  # scale up the image dimensions with 'scale'
        new_out = tf.image.resize(img, new_shape)
        return self.tensor_to_image(new_out)

    def clip_0_to_1(self, image):
        """Compresses image values between 0 and 1

        Args:
            image (tf.Tensor): float image inputs

        Returns:
            float list: outputs image with values between 0 and 1
        """
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
