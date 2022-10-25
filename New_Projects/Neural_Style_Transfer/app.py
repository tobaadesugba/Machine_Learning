import streamlit as st
from modules.style_transfer_module import TransferStyle
from modules.image_processing_module import ImageProcessing
from os import getcwd
from PIL import Image

# from io import BytesIO
# import urllib
path = getcwd()


def get_test_image():
    test_options = ["Image 1", "Image 2", "Image 3"]
    test_selection = st.radio(label="choose a test image", options=test_options)

    if test_selection == "Image 1":
        content_image = path + '/data/content1.jpg'
        content_image = Image.open(content_image)
        # url = IMAGE URL
        # file_obj = BytesIO(return urllib.request.urlopen(url).read())
    elif test_selection == "Image 2":
        content_image = path + '/data/content2.jpg'
        content_image = Image.open(content_image)
    elif test_selection == "Image 3":
        content_image = path + '/data/content3.jpg'
        content_image = Image.open(content_image)
    else:
        content_image = None
    return content_image


def get_style_image():
    test_options = ["Image 1", "Image 2", "Image 3"]
    test_selection = st.radio(label="choose a style image", options=test_options)

    if test_selection == "Image 1":
        style_image = path + '/data/style1.jpg'
        style_image = Image.open(style_image)
    elif test_selection == "Image 2":
        style_image = path + '/data/style2.jpg'
        style_image = Image.open(style_image)
    elif test_selection == "Image 3":
        style_image = path + '/data/style3.jpg'
        style_image = Image.open(style_image)
    else:
        style_image = None
    return style_image


# add sidebar elements
with st.sidebar:
    options = [None, "Upload Selection", "Webcam Selection", "Test Selection"]
    file_selection = st.radio("Pick your image input mode", options=options)

    if file_selection == "Test Selection":
        image = get_test_image()
    elif file_selection == "Upload Selection":
        image = st.file_uploader(label="Please choose your image file", type=['png', 'jpg', 'jpeg'])
        if image:
            image = Image.open(image)
    elif file_selection == "Webcam Selection":
        image = st.camera_input(label="smile for the camera")
        if image:
            image = Image.open(image)
    else:
        image = None
        st.warning("Please pick an option")
        st.stop()

    # display input image if available
    if image and file_selection != "Webcam Selection":
        st.image(image=image, caption="Input Image")

st.header("Output Viewer")
st.write("Don't forget to set your input image in the sidebar 👈")
# choose which style to use
styles = [None, "Upload Style", "Test Styles"]
style_selection = st.radio("Pick your style input mode", options=styles)
if style_selection == "Upload Style":
    style_image = st.file_uploader(label="Upload the style to use", type=['png', 'jpg', 'jpeg'])
    if style_image:
        style_image = Image.open(style_image)
elif style_selection == "Test Styles":
    style_image = get_style_image()
else:
    style_image = None
    st.warning("Please pick an option")
    st.stop()
intensity = st.slider('Set the intensity of the style', min_value=1, max_value=50, value=50)
quality = st.slider('Set the image quality of the style', min_value=128, max_value=256, step=16)
# load style transfer function
if style_image:
    style_transfer = TransferStyle(content_path=image,
                                 style_path=style_image,
                                 intensity=intensity,
                                 quality=quality)
# button to start training
if st.button('Transfer Image Style'):
    if style_image:
        with st.spinner("Getting Image Style. Please wait..."):
            styled_image = style_transfer()
        st.success("Image Style Successfully Transferred")

        # set columns for displaying output
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Your Image")
            st.image(image)
        with col2:
            st.subheader("Style Image")
            st.image(style_image)
        with col3:
            st.subheader("Output Image")
            st.image(ImageProcessing().load_output_image(styled_image))
