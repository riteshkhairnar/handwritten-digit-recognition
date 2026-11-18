import streamlit as st
import base64
from io import BytesIO
from PIL import Image

st.title("Image Display Test")

img = Image.new("RGB", (200, 200), "red")
buffer = BytesIO()
img.save(buffer, format="PNG")

encoded = base64.b64encode(buffer.getvalue()).decode()

st.markdown(
    f'<img src="data:image/png;base64,{encoded}" width="200">',
    unsafe_allow_html=True
)