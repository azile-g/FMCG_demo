import os

#streamlit related
import streamlit as st

#ownself 
from my_utils import helper

print([name for name in os.listdir(".") if os.path.isdir("/writer/")])
#data_paths = helper.reader_paths("/writer/")
#st.write(data_paths)