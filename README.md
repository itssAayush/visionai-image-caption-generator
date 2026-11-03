---
title: Image Caption Generator
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# Image Caption Generator

VisionAI is an AI-powered Image Caption Generator that transforms images into meaningful natural-language descriptions using Computer Vision and Natural Language Processing (NLP).

Built using Python, Streamlit, PyTorch and Salesforce BLIP, the application allows users to upload images and generate intelligent captions through a simple, interactive web interface.

## ✨ Features

- **AI-Powered Image Captioning:** Generate natural-language descriptions from uploaded images.
- **Computer Vision + NLP:** Combines visual understanding with language generation.
- **Multiple Caption Detail Levels:** Quick, Balanced and Detailed caption generation.
- **Interactive Web Interface:** A modern dark-themed interface built with Streamlit.
- **Image Preview:** Preview uploaded images before generating captions.
- **Caption Export:** Download generated captions as text files.
- **Inference Timing:** View the time taken to generate captions.

## 🧠 Model & Technology

| Component | Technology |
|---|---|
| Programming Language | Python |
| Computer Vision | Salesforce BLIP |
| Deep Learning Framework | PyTorch |
| NLP / Language Generation | BLIP Transformer Architecture |
| Frontend | Streamlit |
| Image Processing | Pillow |
| Model Ecosystem | Hugging Face Transformers |

**Model:** `Salesforce/blip-image-captioning-large`

## 🏗️ Project Architecture

1. User uploads an image.
2. The image is processed using Pillow.
3. BLIP Processor prepares the image for inference.
4. The pretrained BLIP model generates a caption.
5. The generated text is displayed in the VisionAI interface.
6. Users can copy or download the caption.

## 📁 Project Structure

```text
VisionAI/
│
├── app.py
├── blip_inference.py
├── model.py
├── inference.py
├── dataset.py
├── requirements.txt
├── README.md
│
└── assets/
    └── screenshots/
```

*The structure above represents the intended project organization; adjust it to match the files actually included in the repository.*

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/visionai-image-caption-generator.git
cd visionai-image-caption-generator
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🖥️ Application Preview

Add your project screenshots here:

```markdown
![VisionAI Homepage](assets/screenshots/homepage.png)
![Caption Generation](assets/screenshots/caption-generation.png)
```

## 🔬 Future Improvements

- Fine-tuning on custom image-caption datasets.
- Integration of more advanced vision-language models.
- Multilingual image caption generation.
- Text-to-speech support for generated captions.
- Image question answering.
- Cloud deployment and API integration.

## 🎯 Project Objective

The objective of VisionAI is to explore the integration of Computer Vision and Natural Language Processing to enable machines to interpret visual information and communicate it through human-readable language.

## 👨‍💻 Author

**Aayush Jha**

B.Tech Computer Science & Engineering

Interested in Artificial Intelligence, Machine Learning, Computer Vision and NLP.

---

⭐ If you find this project interesting, consider giving the repository a star!

## 📝 Troubleshooting
- **macOS SSL Verification**: If you encounter SSL certificate verification errors on macOS when downloading pretrained ResNet weights, the code includes a built-in patch (`ssl._create_unverified_context` in `model.py`) to resolve this automatically.

## 📄 License
This project is open-source and available under the MIT License.
