import os
from gensim.models import Word2Vec

if __name__ == "__main__":
    # load the tokens we saved in task 1
    try:
        with open("cleaned_tokens.txt", "r", encoding="utf-8") as f:
            text = f.read()
            # just split by space since it's already tokenized and cleaned
            words = text.split() 
    except FileNotFoundError:
        print("Can't find cleaned_tokens.txt file. Run the scrap.py and preprocess.py files first to generate it.")
        exit()

    print(f"Loaded {len(words)} words.")

    # gensim needs a list of sentences (list of lists) as input 
    # since we lost the original sentence boundaries in task 1, 
    # I am just chopping the text into chunks of 50 words which should be close enough.
    sentences = []
    chunk_size = 50 
    for i in range(0, len(words), chunk_size):
        sentences.append(words[i:i + chunk_size])
        
    print(f"Created {len(sentences)} fake sentences for training")

    # hyperparams to test
    dims = [50, 100]                    # embedding dimensions
    windows = [3, 5]                    # context window size
    neg_samples = [5, 10]               # number of negative samples

    # saving all the models in a folder
    if not os.path.exists("Problem 1/models"):
        os.makedirs("Problem 1/models")

    print("\n---- Training CBOW Models ----")
    for d in dims:
        for w in windows:
            for n in neg_samples:
                print(f"training cbow: dim={d}, win={w}, neg={n}...")
                
                # I have set the min_count to 1 so that it doesn't drop any words as the corpus is already small.
                model_cbow = Word2Vec(sentences, vector_size=d, window=w, sg=0, negative=n, min_count=1, workers=4)
                
                fname = f"Problem 1/models/cbow_d{d}_w{w}_n{n}.model"
                model_cbow.save(fname)

    print("\n---- Training Skip-Gram Models ----")
    for d in dims:
        for ws in windows: # swapped w for ws here
            for ns in neg_samples:
                print(f"training sg: dim={d}, win={ws}, neg={ns}...")
                
                model_sg = Word2Vec(sentences, vector_size=d, window=ws, sg=1, negative=ns, min_count=1, workers=4)
                
                fname = f"Problem 1/models/sg_d{d}_w{ws}_n{ns}.model"
                model_sg.save(fname)

    print("\nDone training all models. Check the Problem 1/models folder.")