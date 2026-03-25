import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters for all models
HIDDEN_DIM = 128
LAYERS = 2
LR = 0.005
EMBED_SZ = 64

class BasicRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # This maps raw character indexes into dense feature vectors
        self.embed = nn.Embedding(vocab_size, EMBED_SZ)
        self.rnn = nn.RNN(EMBED_SZ, HIDDEN_DIM, LAYERS, batch_first=True)
        # This maps the hidden states back to our character vocabulary for prediction
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, x, h=None):
        out = self.embed(x)
        out, h = self.rnn(out, h)
        
        # We return the output for the ENTIRE sequence here (not just the last step).
        # This is because we need to calculate the loss for every single character 
        # prediction during training to generate words properly.
        return self.fc(out), h

class BiLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_SZ)
        # This reads the sequence left-to-right and right-to-left simultaneously
        self.lstm = nn.LSTM(EMBED_SZ, HIDDEN_DIM, LAYERS, batch_first=True, bidirectional=True)
        
        # we have to multiply by 2 here because the forward and backward hidden states get concatenated
        self.fc = nn.Linear(HIDDEN_DIM * 2, vocab_size)

    def forward(self, x, h=None):
        out = self.embed(x)
        out, h = self.lstm(out, h)
        return self.fc(out), h

class RNNAttention(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_SZ)
        self.rnn = nn.RNN(EMBED_SZ, HIDDEN_DIM, LAYERS, batch_first=True)
        
        # A simple local attention mechanism. 
        # Instead of squashing the whole sequence into one vector, we score each time step.
        self.attn = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, 1) # outputs a single scalar score per time step
        )
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, x, h=None):
        emb = self.embed(x)
        out, h = self.rnn(emb, h)
        
        # softmax ensures that all the weights sum to 1 across the sequence length
        attn_weights = F.softmax(self.attn(out), dim=1) 
        
        # applying the weights to the rnn outputs element-wise.
        # this preserves the sequence length so we can still train character-by-character.
        context = out * attn_weights 
        
        return self.fc(context), h

def get_param_count(model):
    # Counts the number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Using dummy vocab just to initialize the models and check the parameter counts
    dummy_vocab = 30 
    
    print("Trainable parameters (Copy these to the report!):")
    print(f"RNN:      {get_param_count(BasicRNN(dummy_vocab)):,}")
    print(f"BiLSTM:   {get_param_count(BiLSTM(dummy_vocab)):,}")
    print(f"RNN+Attn: {get_param_count(RNNAttention(dummy_vocab)):,}")