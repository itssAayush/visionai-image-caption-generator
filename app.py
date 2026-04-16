import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from inference import predict_caption
import os

st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️")

st.title("Image Caption Generator 🖼️ ➡️ 📝")

st.write(
    """
    Welcome to the Image Caption Generator! Upload an image and the Deep Learning model 
    (CNN-LSTM) will generate a descriptive caption for it.
    """
)

# Sidebar with status
with st.sidebar:
    st.header("System Status")
    model_path = "caption_model.pth"
    vocab_path = "vocab.pkl"
    
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        st.success("✅ Model and Vocabulary loaded")
        model_ready = True
    else:
        st.error("❌ Model or Vocabulary not found")
        st.write("Please run `python train.py` first to train the model and generate these files.")
        model_ready = False

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Generated Caption")
        if not model_ready:
            st.warning("Cannot generate caption. Model is not trained yet.")
        else:
            with st.spinner("Generating caption..."):
                # Save uploaded file temporarily for the inference script
                temp_path = "temp_image.jpg"
                image.convert("RGB").save(temp_path)
                
                try:
                    caption = predict_caption(temp_path, model_path, vocab_path)
                    st.success(f"**{caption.capitalize()}**")
                except Exception as e:
                    st.error(f"Error generating caption: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

st.markdown("---")
st.markdown("*Built with PyTorch & Streamlit*")
