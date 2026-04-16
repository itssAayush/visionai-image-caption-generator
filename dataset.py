import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from build_vocab import Vocabulary
import torchvision.transforms as transforms
from PIL import Image

class Flickr8kDataset(Dataset):
    def __init__(self, hf_dataset, vocab, transform=None):
        self.dataset = hf_dataset
        self.vocab = vocab
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        image = item["image"]
        
        # Handle different column names for captions in various HF datasets
        caption_keys = ["caption", "captions", "text", "text_en", "caption_0"]
        caption = None
        for key in caption_keys:
            if key in item:
                caption = item[key]
                break
                
        # If the dataset provides a list of captions per image, take the first one
        if isinstance(caption, list):
            caption = caption[0]
            
        # Convert grayscale to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        if self.transform is not None:
            image = self.transform(image)
            
        # Add <SOS> and <EOS> tokens
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(str(caption))
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return image, torch.tensor(numericalized_caption)

class CapsCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        
        return imgs, targets

def get_loader(dataset_name="jxie/flickr8k", split="train", transform=None, batch_size=32, num_workers=0, shuffle=True, vocab_threshold=5, vocab=None):
    # jxie/flickr8k is a common HF dataset for Flickr8k
    hf_dataset = load_dataset(dataset_name, split=split)
    
    if vocab is None:
        vocab = Vocabulary(vocab_threshold)
        captions = []
        # Build vocab
        for item in hf_dataset:
            caption_keys = ["caption", "captions", "text", "text_en", "caption_0"]
            for key in caption_keys:
                if key in item:
                    caps = item[key]
                    if isinstance(caps, list):
                        captions.extend([str(c) for c in caps])
                    else:
                        captions.append(str(caps))
                    break
        vocab.build_vocabulary(captions)
        
    dataset = Flickr8kDataset(hf_dataset, vocab, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=CapsCollate(pad_idx=pad_idx)
    )
    
    return loader, dataset
