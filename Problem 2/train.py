import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import BasicRNN, BiLSTM, RNNAttention
import pickle

# Device config - use GPU if you have one, otherwise CPU is fine
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NameDataset(Dataset):
    def __init__(self, filepath):
        # Loading the raw text file
        with open(filepath, 'r', encoding='utf-8') as f:
            self.names = [line.strip() for line in f if line.strip()]
            
        # Extracting every unique character present in the dataset
        all_chars = sorted(list(set(''.join(self.names))))
        
        # We need two special tokens for generation to work:
        # '#' is used to pad shorter names so we can process them in batches.
        # '.' acts as both the Start-Of-Sequence (SOS) and End-Of-Sequence (EOS) marker.
        self.chars = ['#', '.'] + all_chars 
        self.vocab_size = len(self.chars)
        
        # Creating dictionaries to translate characters to numbers and vice versa
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        
        # Finding the longest name to determine our fixed sequence length.
        # +2 because every name will be wrapped by '.' like: .Aarav.
        self.max_len = max(len(name) for name in self.names) + 2

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        
        # Converting the string name into a list of integer indexes, wrapping it in '.'
        idxs = [self.stoi['.']] + [self.stoi[c] for c in name] + [self.stoi['.']]
        
        # Padding the rest of the sequence with '#' up to the max length.
        while len(idxs) < self.max_len:
            idxs.append(self.stoi['#'])
            
        tensor = torch.tensor(idxs, dtype=torch.long)
        return tensor[:-1], tensor[1:]

def train_model(model, dataloader, model_name, pad_idx, epochs=15):
    print(f"\n--- Training {model_name} ---")
    model = model.to(device)
    
    # Using ignore_index so that our model completely ignore the '#' padding token we used earlier
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            
            # Flattening the batch and sequence length
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Printing updates every 5 epochs to track progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
            
    # Save the trained weights for this model
    torch.save(model.state_dict(), f"Problem 2/{model_name}_weights.pth")
    print(f"Saved Problem 2/{model_name}_weights.pth")

if __name__ == "__main__":
    print(f"Using device: {device}")
    
    # Initialize the dataset and build the vocabulary
    dataset = NameDataset("Problem 2/Training_Names.txt")
    print(f"Loaded {len(dataset)} names. Vocab size: {dataset.vocab_size}")
    
    # DataLoader handles the random shuffling and chunks data into batches of 64
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Instantiate the three models
    rnn_model = BasicRNN(dataset.vocab_size)
    lstm_model = BiLSTM(dataset.vocab_size)
    attn_model = RNNAttention(dataset.vocab_size)

    pad_idx = dataset.stoi['#']
    
    # Training starts
    train_model(rnn_model, dataloader, "BasicRNN", pad_idx, epochs=15)
    train_model(lstm_model, dataloader, "BiLSTM", pad_idx, epochs=15)
    train_model(attn_model, dataloader, "RNNAttention", pad_idx, epochs=15)
    
    # Exporting the character dictionaries as a pickle file
    with open("Problem 2/char_mappings.pkl", "wb") as f:
        pickle.dump({'stoi': dataset.stoi, 'itos': dataset.itos, 'vocab_size': dataset.vocab_size}, f)