import streamlit as st

st.set_page_config(page_title="Instructions", page_icon="👋")

st.write("# Welcome to the FMCG Detection Application! 👋")
st.sidebar.success("Start the video or image capture when you are ready.")
st.markdown(
    """
    The FMCG Detection Tool aims to capture and label FMCG goods.
    ###### **👈 Navigate to the video or image recogition from the sidebar when you are ready!** 
    ### How to use the video application?
    - Start the video recognition, and move your objects near the webcam. 
    - When the video capture is satisfactory, stop the recording. 
    - If the video needs to be relabelled, fill in the relabelling form. 
    - If not, record as many items as you need! 
    ### Features TBC: 
    - TRUN server (faster connection)
    - data explorer and navigator 
    - cloud storage 
    """
)