import streamlit as st
from modules.style_transfer_module import TransferStyle
from modules.image_processing_module import ImageProcessing
from os import getcwd
from PIL import Image

# from io import BytesIO
# import urllib
path = getcwd()


def get_test_image():
    test_options = ["Doggy", "Puppy", "Puppies Sleeping"]
    test_selection = st.radio(label="Choose a Test Image", options=test_options)

    if test_selection == "Doggy":
        content_image = path + '/data/content1.jpg'
        content_image = Image.open(content_image)
    elif test_selection == "Puppy":
        content_image = path + '/data/content2.jpg'
        content_image = Image.open(content_image)
    elif test_selection == "Puppies Sleeping":
        content_image = path + '/data/content3.jpg'
        content_image = Image.open(content_image)
    else:
        content_image = None
    return content_image


def get_style_image():
    test_options = ["Lines & Edges", "Van Gogh", "Flowers"]
    test_selection = st.radio(label="Choose a Style Image", options=test_options)

    if test_selection == "Lines & Edges":
        style_image = path + '/data/style1.jpg'
        style_image = Image.open(style_image)
    elif test_selection == "Van Gogh":
        style_image = path + '/data/style2.jpg'
        style_image = Image.open(style_image)
    elif test_selection == "Flowers":
        style_image = path + '/data/style3.jpg'
        style_image = Image.open(style_image)
    else:
        style_image = None
    return style_image


st.header("Output Section")
st.write("Set your input image in the sidebar first 👈")

# add sidebar elements
with st.sidebar:
    st.header("Input Section")
    options = [None, "Upload Selection", "Webcam Selection", "Test Selection"]
    file_selection = st.radio("Pick Your Image Input Mode", options=options)

    if file_selection == "Test Selection":
        image = get_test_image()
    elif file_selection == "Upload Selection":
        image = st.file_uploader(label="Please Choose Your Image File", type=['png', 'jpg', 'jpeg'])
        if image:
            image = Image.open(image)
    elif file_selection == "Webcam Selection":
        image = st.camera_input(label="Smile for the Camera")
        if image:
            image = Image.open(image)
    else:
        image = None
        st.warning("Please Pick an Option")
        st.stop()

    # display input image if available
    if image and file_selection != "Webcam Selection":
        st.image(image=image, caption="Input Image")


# choose which style to use
styles = [None, "Upload Style", "Test Styles"]
style_selection = st.radio("Pick Your Style Input Mode", options=styles)
if style_selection == "Upload Style":
    style_image = st.file_uploader(label="Upload the Style to use", type=['png', 'jpg', 'jpeg'])
    if style_image:
        style_image = Image.open(style_image)
elif style_selection == "Test Styles":
    style_image = get_style_image()
else:
    style_image = None
    st.warning("Please pick an option")
    st.stop()
intensity = st.slider('Set the Intensity of the Style Transfer', min_value=1, max_value=50, value=10)
quality = st.slider('Set the Image Quality of the Style Transfer', min_value=128, max_value=256, step=16)
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
    else:
        st.warning("Please pick a style image")
