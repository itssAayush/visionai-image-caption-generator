import torch
import torch.nn as nn
import ssl
import torchvision.models as models

# Fix for macOS SSL certificate verification error when downloading weights
ssl._create_default_https_context = ssl._create_unverified_context

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        
        # Load a pretrained ResNet-50 model
        resnet = models.resnet50(pretrained=True)
        # We replace the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        # features will have shape (batch_size, 2048, 1, 1)
        features = self.resnet(images)
        features = features.view(features.size(0), -1) # reshape to (batch_size, 2048)
        features = self.dropout(self.relu(self.fc(features)))
        return features
        
    def fine_tune(self, train_CNN=False):
        self.train_CNN = train_CNN
        for param in self.resnet.parameters():
            param.requires_grad = train_CNN

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        # Remove the <EOS> token from captions before feeding into LSTM
        embeddings = self.embed(captions[:, :-1])
        # Concatenate the image features as the first 'word'
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
        
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs
        
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(1)
            states = None
            
            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(1))
                predicted = output.argmax(1)
                
                word = vocabulary.itos[predicted.item()]
                result_caption.append(word)
                
                if word == "<EOS>":
                    break
                    
                x = self.decoderRNN.embed(predicted).unsqueeze(1)
                
        return [word for word in result_caption if word not in ["<SOS>", "<EOS>", "<UNK>", "<PAD>"]]
