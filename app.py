import streamlit as st
from PIL import Image
from blip_inference import generate_blip_caption
import os

st.set_page_config(page_title="Image Caption Generator (BLIP)", page_icon="🖼️")

st.title("Image Caption Generator 🖼️ ➡️ 📝")

st.write(
    """
    Welcome to the Image Caption Generator! Upload an image and the **Salesforce BLIP** (Bootstrapping Language-Image Pre-training) model will generate an incredibly accurate, descriptive caption for it.
    """
)

# Sidebar with status
with st.sidebar:
    st.header("System Status")
    st.success("✅ BLIP Model Integrated")
    st.write("The model will automatically download from HuggingFace on your very first inference if it isn't cached locally.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Generated Caption")
        
        with st.spinner("Generating caption (BLIP)..."):
            # Save uploaded file temporarily for the inference script
            temp_path = "temp_image.jpg"
            image.convert("RGB").save(temp_path)
            
            try:
                # Call the new BLIP inference function
                caption = generate_blip_caption(temp_path)
                st.success(f"**{caption.capitalize()}**")
            except Exception as e:
                st.error(f"Error generating caption: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

st.markdown("---")
st.markdown("*Built with PyTorch, Transformers & Streamlit*")
