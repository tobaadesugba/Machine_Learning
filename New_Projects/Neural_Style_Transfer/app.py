import streamlit as st

def get_test_image():
    test_options = ["Image 1", "Image 2", "Image 3"]
    test_selection = st.radio(label="choose a test image", options=test_options)
    '''
    if test_selection == "Image 1":
        image = 
    elif test_selection == "Image 2":
        image =
    elif test_selection == "Image 3":
        image
    return image
    '''

def get_style_image():
    test_options = ["Image 1", "Image 2", "Image 3"]
    test_selection = st.radio(label="choose a test image", options=test_options)
    '''
    if test_selection == "Image 1":
        image = 
    elif test_selection == "Image 2":
        image =
    elif test_selection == "Image 3":
        image
    return image
    '''

def get_style(img):
    pass

# add side bar elements
with st.sidebar:
    options = [None, "Upload Selection", "Webcam Selection", "Test Selection"]
    file_selection = st.radio("Pick your input mode", options=options)

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
style_selection = st.radio("Pick your input mode", options=styles)
if style_selection == "Upload Style":
    style_image = st.file_uploader(label="Upload the style to use", type=['png', 'jpg', 'jpeg'])
elif style_selection == "Test Styles":
    style_image = get_style_image()
else:
    style_image = None
    st.warning("Please pick an option")
    st.stop()

# button to start training
if st.button('Transfer Image Style'):
    if style_image:
        st.spinner("Getting Image Style...")
        styled_image = get_style(image)
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
            st.header("Output Image")
            st.image(styled_image)
