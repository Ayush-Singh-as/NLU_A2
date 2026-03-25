import torch
import torch.nn.functional as F
import pickle
from models import BasicRNN, BiLSTM, RNNAttention

## Device config - use GPU if you have one, otherwise CPU is fine
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load up the vocab dictionaries and training data
try:
    with open("Problem 2/char_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    stoi = mappings['stoi']
    itos = mappings['itos']
    vocab_size = mappings['vocab_size']
    
    # keeping training names in a set so we can quickly do set math for the novelty score later
    with open("Problem 2/Training_Names.txt", "r", encoding="utf-8") as f:
        train_names = set([line.strip() for line in f if line.strip()])
except FileNotFoundError:
    print("Missing files! Make sure you ran the training script first.")
    exit()

def generate_names(model, num_names=500, max_len=12):
    model.eval()
    model.to(device)
    generated = []
    
    with torch.no_grad():
        for _ in range(num_names):
            # prime the pump with the start token '.'
            curr_seq = [stoi['.']]
            
            # generating character by character
            for _ in range(max_len):
                x = torch.tensor([curr_seq], dtype=torch.long).to(device)
                logits, _ = model(x)
                
                # grab the predictions for the very last time step
                probs = F.softmax(logits[0, -1, :], dim=0)
                
                # sample from the distribution rather than taking the strict max.
                # this introduces randomness so we get different names!
                next_char_idx = torch.multinomial(probs, 1).item()
                
                # stop if the word is finished
                if itos[next_char_idx] == '.':
                    break 
                    
                curr_seq.append(next_char_idx)
            
            # converting the sequence of IDs back to a string (skipping the first '.' token)
            name = ''.join([itos[idx] for idx in curr_seq[1:]])
            
            # throwing out garbage generations (too short, or accidentally generated a pad token)
            if len(name) > 2 and '#' not in name: 
                generated.append(name.capitalize())
                
    return generated

def calculate_metrics(generated_list):
    if not generated_list:
        return 0.0, 0.0, []
        
    unique_names = set(generated_list)
    
    # Calculating diversity: what percentage of the generated names are actually distinct from each other?
    diversity = (len(unique_names) / len(generated_list)) * 100
    
    # Calculating novelty: what percentage of those unique names are brand new (not in the training file)?
    novel_names = unique_names - train_names
    novelty = (len(novel_names) / len(unique_names)) * 100
    
    # Printing 5 random samples to the terminal
    samples = list(unique_names)[:5]
    return diversity, novelty, samples

if __name__ == "__main__":
    print("Loading models and hallucinating names...\n")
    
    # initialize the blank blueprints
    rnn = BasicRNN(vocab_size)
    lstm = BiLSTM(vocab_size)
    attn = RNNAttention(vocab_size)
    
    # load in the weights we just trained
    rnn.load_state_dict(torch.load("Problem 2/BasicRNN_weights.pth", map_location=device, weights_only=True))
    lstm.load_state_dict(torch.load("Problem 2/BiLSTM_weights.pth", map_location=device, weights_only=True))
    attn.load_state_dict(torch.load("Problem 2/RNNAttention_weights.pth", map_location=device, weights_only=True))
    
    models_to_test = [
        ("Vanilla RNN", rnn), 
        ("BiLSTM", lstm), 
        ("RNN+Attention", attn)
    ]
    
    for name, model in models_to_test:
        print(f"{'='*30}")
        print(f"Evaluating {name}")
        print(f"{'='*30}")
        
        gen_names = generate_names(model, num_names=500)
        div, nov, samples = calculate_metrics(gen_names)
        
        print(f"Diversity Score : {div:.2f}%")
        print(f"Novelty Score   : {nov:.2f}%")
        print(f"Sample Outputs  : {', '.join(samples)}\n")