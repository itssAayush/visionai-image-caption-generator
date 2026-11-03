import torch
import streamlit as st
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


@st.cache_resource
def load_model():
    """Load and cache the BLIP model and processor."""
    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    # Keep model and inputs on the same device for inference.
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def generate_blip_caption(image_path, max_new_tokens=50):
    """Generate a caption for the image at ``image_path``."""
    processor, model, device = load_model()
    with Image.open(image_path) as image:
        raw_image = image.convert("RGB")

    inputs = processor(raw_image, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.decode(output[0], skip_special_tokens=True)
