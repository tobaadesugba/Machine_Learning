import streamlit as st
from modules.style_transfer_module import TransferStyle
from modules.image_processing_module import ImageProcessing

def get_test_image():
    test_options = ["Image 1", "Image 2", "Image 3"]
    test_selection = st.radio(label="choose a test image", options=test_options)

    if test_selection == "Image 1":
        image = 'data//content1.jpg'
    elif test_selection == "Image 2":
        image = 'data//content2.jpg'
    elif test_selection == "Image 3":
        image = 'data//content3.jpg'
    return image

def get_style_image():
    test_options = ["Image 1", "Image 2", "Image 3"]
    test_selection = st.radio(label="choose a style image", options=test_options)

    if test_selection == "Image 1":
        image = 'data//style1.jpg'
    elif test_selection == "Image 2":
        image = 'data//style2.jpg'
    elif test_selection == "Image 3":
        image = 'data//style3.jpg'
    return image

# add side bar elements
with st.sidebar:
    options = [None, "Upload Selection", "Webcam Selection", "Test Selection"]
    file_selection = st.radio("Pick your image input mode", options=options)

    if file_selection == "Test Selection":
        image = get_test_image()
    elif file_selection == "Upload Selection":
        image = st.file_uploader(label="Please choose your image file", type=['png', 'jpg', 'jpeg'])
    elif file_selection == "Webcam Selection":
        image = st.camera_input(label="smile for the camera")
    else:
        image = None
        st.warning("Please pick an option")
        st.stop()

    # display input image if available
    if image:
        if (file_selection != "Webcam Selection"):
            st.image(image=image, caption="Input Image")

st.header("Output Viewer")
st.write("Don't forget to set your input image in the sidebar 👈")
# choose which style to use
styles = [None, "Upload Style", "Test Styles"]
style_selection = st.radio("Pick your style input mode", options=styles)
if style_selection == "Upload Style":
    style_image = st.file_uploader(label="Upload the style to use", type=['png', 'jpg', 'jpeg'])
elif style_selection == "Test Styles":
    style_image = get_style_image()
else:
    style_image = None
    st.warning("Please pick an option")
    st.stop()
intensity = st.slider('Set the intensity of the style', min_value=1, max_value=50, value=50)
quality = st.slider('Set the image quality of the style', min_value=128, max_value=256, step=16)

# button to start training
if st.button('Transfer Image Style'):
    if style_image:
        with st.spinner("Getting Image Style. Please wait..."):
            styled_image = TransferStyle(content_path=image,
                                         style_path=style_image,
                                         intensity=intensity,
                                         quality=quality)()
        st.success("Image Style Successfully Transferred")

        # set columns for displaying output
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Your Image")
            st.image(image)
        with col2:
            st.subheader("Style Image")
            st.image((style_image))
        with col3:
            st.subheader("Output Image")
            st.image(ImageProcessing().load_output_image(styled_image))
