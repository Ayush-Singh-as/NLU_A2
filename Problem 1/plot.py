import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

# I have used words that we want to see grouped together to check for clusters
# Mixing some academic terms and some general terms to see if the models separate them
words_to_plot = [
    'research', 'phd', 'thesis', 'project', 'lab',
    'student', 'btech', 'mtech', 'ug', 'pg',
    'exam', 'theory', 'practical', 'grades',
    'campus', 'hostel', 'sports', 'festival'
]

def get_2d_vectors(model_path):
    print(f"Loading {model_path} for plotting...")
    try:
        model = Word2Vec.load(model_path)
    except Exception as e:
        print(f"Failed to load: {e}")
        return [], []
        
    valid_words = []
    vectors = []
    
    # Only grab words that have min_count > 0 in Task 2
    for w in words_to_plot:
        if w in model.wv:
            valid_words.append(w)
            vectors.append(model.wv[w])
        else:
            print(f"  [Skipping '{w}' - not in vocab]")
            
    if not vectors:
        return [], []
        
    # Crushing the 50/100 dimensions down to 2 so we can actually draw them
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    return valid_words, vectors_2d

if __name__ == "__main__":
    cbow_file = "Problem 1/models/cbow_d100_w5_n5.model"
    sg_file = "Problem 1/models/sg_d100_w5_n5.model"
    
    # Get the 2D coordinates for both models
    cbow_words, cbow_vecs = get_2d_vectors(cbow_file)
    sg_words, sg_vecs = get_2d_vectors(sg_file)
    
    if not cbow_vecs.any() or not sg_vecs.any():
        print("Not enough data to plot. Check your models. Run the train.py file first to generate the models.")
        exit()

    # Plotting CBOW and Skip-gram side by side for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot CBOW
    ax1.scatter(cbow_vecs[:, 0], cbow_vecs[:, 1], c='blue', alpha=0.6)
    for i, word in enumerate(cbow_words):
        ax1.annotate(word, xy=(cbow_vecs[i, 0], cbow_vecs[i, 1]), xytext=(3, 3), textcoords='offset points')
    ax1.set_title("CBOW Word Clusters")
    
    # Plot Skip-gram
    ax2.scatter(sg_vecs[:, 0], sg_vecs[:, 1], c='red', alpha=0.6)
    for i, word in enumerate(sg_words):
        ax2.annotate(word, xy=(sg_vecs[i, 0], sg_vecs[i, 1]), xytext=(3, 3), textcoords='offset points')
    ax2.set_title("Skip-Gram Word Clusters")
    
    plt.tight_layout()
    
    # Save it as a png file
    plt.savefig("Problem 1/word_clusters.png")
    print("\n--> Saved cluster plot as 'Problem 1/word_clusters.png'")
    
    plt.show()