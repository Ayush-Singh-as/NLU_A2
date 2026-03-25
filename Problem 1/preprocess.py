import string
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Uncomment and run these once
# nltk.download('punkt')
# nltk.download('punkt_tab')

if __name__ == "__main__":
    dataset_file = "Problem 1/iitj_corpus.txt"
    
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            raw_data = f.read()
    except FileNotFoundError:
        print("Error: Couldn't find the corpus file. Run the scraping.py file first.")
        exit()
        
    print(f"Loaded {len(raw_data)} chars. Starting cleanup...")
    
    # Converting to lowercase
    text_lower = raw_data.lower()
    
    # Using NLTK tokenization
    tokens = word_tokenize(text_lower)
    
    # Filter out numbers and weird punctuation
    clean_words = []
    for word in tokens:
        if word.isalpha() and word not in string.punctuation:
            clean_words.append(word)

    vocab = set(clean_words)
    
    # Printing corpus statistics
    print("\n------ Stats ------")
    print(f"Total Documents: 63")
    print(f"Total Tokens: {len(clean_words)}")
    print(f"Vocab size: {len(vocab)}")
    
    # Checking the top 5 most common words in the corpus
    word_freq = Counter(clean_words)
    print("\nTop 5 words:", word_freq.most_common(5))
    
    print("\nGenerating word cloud...")
    
    # Creating a single string of all the words
    cloud_text = " ".join(clean_words)
    
    # Standard styling
    wc = WordCloud(width=800, height=400, background_color='white', max_words=150).generate(cloud_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off') 
    
    plt.savefig("Problem 1/iitj_wordcloud.png", bbox_inches='tight')
    print("Saved cloud to Problem 1/iitj_wordcloud.png")
    
    # Save the final tokens
    with open("Problem 1/cleaned_corpus.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(clean_words))
        
    print("Tokens saved to Problem 1/cleaned_corpus.txt")