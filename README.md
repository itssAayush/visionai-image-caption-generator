---
title: Image Caption Generator
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# Image Caption Generator 🖼️ ➡️ 📝

An advanced image captioning application that automatically generates highly accurate, descriptive captions for uploaded images using Deep Learning. 

This project features a dual-architecture approach:
1. **Salesforce BLIP (Bootstrapping Language-Image Pre-training)**: A state-of-the-art transformer model integrated via Hugging Face for maximum accuracy and robust production-level inference.
2. **Custom CNN-LSTM Encoder-Decoder**: A custom-built PyTorch architecture (ResNet-50 encoder and an LSTM decoder) trained from scratch on the Flickr8k dataset.

It provides a clean, interactive **Streamlit** web interface for generating captions instantly.

## 🌟 Features

- **High-Accuracy Inference**: Uses the robust Salesforce BLIP model for state-of-the-art zero-shot image captioning.
- **Custom Model Training Pipeline**: Complete end-to-end pipeline for training a CNN-LSTM model on custom datasets.
- **Interactive UI**: Built with Streamlit for a fast, responsive, and user-friendly experience.
- **Hugging Face Spaces Ready**: Pre-configured metadata for seamless deployment.

## 🛠️ Technology Stack

- **Deep Learning Framework**: PyTorch
- **Models**: Salesforce BLIP, ResNet-50, LSTM
- **Libraries**: Transformers (Hugging Face), Torchvision, NLTK, PIL
- **Frontend**: Streamlit

## 📂 Project Structure

```text
├── app.py                # Streamlit web application (uses BLIP for frontend)
├── blip_inference.py     # BLIP model inference script
├── model.py              # Custom CNN-LSTM architecture (ResNet50 + LSTM)
├── train.py              # Training script for the custom model
├── dataset.py            # Custom PyTorch Dataset loader for Flickr8k
├── build_vocab.py        # Vocabulary builder for custom training
├── inference.py          # Command-line inference for the custom CNN-LSTM model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd image-caption-generator
   ```

2. **Create a virtual environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Running the Web App (BLIP Model)
The easiest way to use the application is through the interactive Streamlit web interface. It will automatically download the BLIP Large model on the very first run.
```bash
streamlit run app.py
```
Simply upload an image through the UI to generate an accurate caption!

### 2. Custom Model Training (CNN-LSTM)
If you wish to train the custom Encoder-Decoder model from scratch:
1. Download the Flickr8k dataset (images and captions).
2. Configure the dataset paths appropriately (see `dataset.py` and `train.py`).
3. Run the training script:
   ```bash
   python train.py
   ```
4. This process will generate a vocabulary dictionary (`vocab.pkl`) and save model weights (`caption_model.pth`).

### 3. Custom Model Inference
Once the custom CNN-LSTM model is fully trained, you can use the command-line inference script to generate captions:
```bash
python inference.py --image path/to/image.jpg
```

## 📝 Troubleshooting
- **macOS SSL Verification**: If you encounter SSL certificate verification errors on macOS when downloading pretrained ResNet weights, the code includes a built-in patch (`ssl._create_unverified_context` in `model.py`) to resolve this automatically.

## 📄 License
This project is open-source and available under the MIT License.
