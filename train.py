import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataset import get_loader
from model import CNNtoRNN
import sys

def train():
    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 5
    batch_size = 32

    # Transforms for image processing
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    print("Loading dataset and building vocabulary...")
    # Load dataset
    try:
        # Note: loading dataset might take some time and network bandwidth
        train_loader, dataset = get_loader(
            transform=transform, 
            batch_size=batch_size, 
            split="train"
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please ensure you have internet access and the Huggingface datasets library is installed.")
        sys.exit(1)

    vocab_size = len(dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Initialize model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    # Freeze CNN layers
    model.encoderCNN.fine_tune(False)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions) # Let model.py handle the slicing array

            # Loss requires outputs shape (batch*seq_len, vocab_size) and targets (batch*seq_len)
            # targets should be captions starting from the second token
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {total_loss/len(train_loader):.4f}")
        
    print("Training Complete. Saving model...")
    torch.save(model.state_dict(), "caption_model.pth")
    # Also save the vocab so we can use it in inference
    import pickle
    with open("vocab.pkl", "wb") as f:
        pickle.dump(dataset.vocab, f)
    print("Model and vocabulary saved locally.")

if __name__ == "__main__":
    train()
