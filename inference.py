import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
import pickle
import argparse
import sys

def predict_caption(image_path, model_path, vocab_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load Vocabulary
    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        print(f"Vocabulary file {vocab_path} not found. Please train the model first.")
        sys.exit(1)

    vocab_size = len(vocab)
    embed_size = 256
    hidden_size = 256
    num_layers = 1

    # Load Model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
        sys.exit(1)
        
    model.eval()

    # Load and Transform Image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict caption
    caption_list = model.caption_image(image_tensor, vocab)
    caption = " ".join(caption_list)
    print("--------------------------------------------------")
    print(f"Generated Caption: {caption.capitalize()}")
    print("--------------------------------------------------")
    return caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Caption Generator Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model", type=str, default="caption_model.pth", help="Path to trained model")
    parser.add_argument("--vocab", type=str, default="vocab.pkl", help="Path to saved vocabulary")
    args = parser.parse_args()
    
    predict_caption(args.image, args.model, args.vocab)
