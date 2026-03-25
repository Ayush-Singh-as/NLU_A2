import sys
from gensim.models import Word2Vec

# Words used for evaluation of the models as given in the assignment.
target_words = ['research', 'student', 'phd', 'exam']

def evaluate_model(filepath):
    print(f"\nLoading {filepath}...")
    try:
        m = Word2Vec.load(filepath)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    print("\nNeighbors test:")
    for w in target_words:
        try:
            # Pulling the top 5 closest vectors in the space
            sims = m.wv.most_similar(w, topn=5)
            print(f"  {w}:")
            for word, score in sims:
                print(f"    {word} ({round(score, 3)})")
        except KeyError:
            # This can happen if a word gets filtered out during the min_count check in Task 2
            print(f"  [Warning: '{w}' dropped from vocab]")

    print("\nAnalogy test:")
    # Testing standard academic relationships
    cases = [
        ("ug", "btech", "pg"), 
        ("btech", "student", "phd"),
        ("theory", "exam", "practical") 
    ]
    
    for w1, w2, w3 in cases:
        try:
            # Finding the closest match to w2+w3 that is far from w1
            preds = m.wv.most_similar(positive=[w2, w3], negative=[w1], topn=3)
            print(f"  {w1} : {w2} :: {w3} : ?")
            for p, score in preds:
                print(f"    -> {p} ({round(score, 3)})")
        except KeyError as err:
            print(f"  Skipping {w1}-{w2}-{w3}, missing word: {err}")

if __name__ == "__main__":
    # Just comparing the deepest CBOW and SG models to see the difference
    # in clustering behavior instead of dumping all 16 to the terminal.
    cbow_model = "Problem 1/models/cbow_d100_w5_n5.model"
    sg_model = "Problem 1/models/sg_d100_w5_n5.model"
    
    print("---- CBOW ----")
    evaluate_model(cbow_model)
    
    print("\n---- SKIP-GRAM ----")
    evaluate_model(sg_model)