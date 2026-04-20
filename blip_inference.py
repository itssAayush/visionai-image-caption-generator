import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize the model and processor once globally so it doesn't reload on every inference
# using Streamlit caching
import streamlit as st

@st.cache_resource
def load_model():
    print("Loading BLIP model...")
    # This automatically downloads the large model for maximum accuracy
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    return processor, model, device

def generate_blip_caption(image_path):
    processor, model, device = load_model()
    
    # Process image
    raw_image = Image.open(image_path).convert('RGB')
    
    # The processor prepares both images and optional text prompts
    inputs = processor(raw_image, return_tensors="pt").to(device)
    
    # Generate the caption
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
        
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
