import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

# ==================================================
# LOAD DATASET
# ==================================================

# Dataset:
# Same dataset used in Assignment 17

df = pd.read_csv("text_dataset.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# Use cleaned text column from Assignment 17

texts = df["final_clean_text"].astype(str)

# ==================================================
# PART 1 : ONE-HOT ENCODING
# ==================================================

# ==================================================
# Task 1 : Manual One-Hot Encoding
# ==================================================

sample_sentences = texts.head(5).tolist()

vocabulary = sorted(
    list(
        set(
            " ".join(sample_sentences).split()
        )
    )
)

print("\nVocabulary:")
print(vocabulary)

one_hot_vectors = []

for sentence in sample_sentences:

    words = sentence.split()

    vector = []

    for vocab_word in vocabulary:

        if vocab_word in words:
            vector.append(1)
        else:
            vector.append(0)

    one_hot_vectors.append(vector)

print("\nManual One-Hot Encoded Matrix")

one_hot_df = pd.DataFrame(
    one_hot_vectors,
    columns=vocabulary
)

print(one_hot_df)

# ==================================================
# Task 2 : One-Hot Encoding using Scikit-Learn
# ==================================================

onehot_vectorizer = CountVectorizer(
    binary=True
)

onehot_matrix = onehot_vectorizer.fit_transform(
    sample_sentences
)

print("\nScikit-Learn Vocabulary")

print(
    onehot_vectorizer.vocabulary_
)

print("\nEncoded Matrix")

print(
    pd.DataFrame(
        onehot_matrix.toarray(),
        columns=onehot_vectorizer.get_feature_names_out()
    )
)

# ==================================================
# PART 2 : BAG OF WORDS
# ==================================================

# ==================================================
# Task 3 : BoW Representation
# ==================================================

bow_vectorizer = CountVectorizer()

bow_matrix = bow_vectorizer.fit_transform(
    texts
)

print("\nVocabulary Size")

print(
    len(
        bow_vectorizer.vocabulary_
    )
)

print("\nBoW Matrix Shape")

print(
    bow_matrix.shape
)

print("\nSample Feature Vectors")

print(
    bow_matrix.toarray()[:5]
)

# ==================================================
# Task 4 : Word Frequency Analysis
# ==================================================

word_frequency = np.array(
    bow_matrix.sum(axis=0)
).flatten()

words = bow_vectorizer.get_feature_names_out()

frequency_df = pd.DataFrame({
    "Word": words,
    "Frequency": word_frequency
})

frequency_df = frequency_df.sort_values(
    by="Frequency",
    ascending=False
)

print("\nTop 10 Frequent Words")

print(
    frequency_df.head(10)
)

print("\nLeast Frequent Words")

print(
    frequency_df.tail(10)
)

print("""

BoW captures frequency information by
counting how many times a word appears
inside documents.

""")

# ==================================================
# PART 3 : N-GRAMS
# ==================================================

# ==================================================
# Task 5 : Unigram
# ==================================================

unigram_vectorizer = CountVectorizer(
    ngram_range=(1,1)
)

unigram_matrix = unigram_vectorizer.fit_transform(
    texts
)

print("\nUnigram Vocabulary Size")

print(
    len(
        unigram_vectorizer.vocabulary_
    )
)

# ==================================================
# Bigram
# ==================================================

bigram_vectorizer = CountVectorizer(
    ngram_range=(2,2)
)

bigram_matrix = bigram_vectorizer.fit_transform(
    texts
)

print("\nBigram Vocabulary Size")

print(
    len(
        bigram_vectorizer.vocabulary_
    )
)

# ==================================================
# Trigram
# ==================================================

trigram_vectorizer = CountVectorizer(
    ngram_range=(3,3)
)

trigram_matrix = trigram_vectorizer.fit_transform(
    texts
)

print("\nTrigram Vocabulary Size")

print(
    len(
        trigram_vectorizer.vocabulary_
    )
)

print("\nSample Unigram Features")

print(
    unigram_matrix.toarray()[:3]
)

print("\nSample Bigram Features")

print(
    bigram_matrix.toarray()[:3]
)

print("\nSample Trigram Features")

print(
    trigram_matrix.toarray()[:3]
)

# ==================================================
# Task 6 : Combined N-Grams
# ==================================================

combined_vectorizer = CountVectorizer(
    ngram_range=(1,2)
)

combined_matrix = combined_vectorizer.fit_transform(
    texts
)

print("\nCombined N-Gram Vocabulary Size")

print(
    len(
        combined_vectorizer.vocabulary_
    )
)

print("""

Using Unigrams + Bigrams provides
better context than unigrams alone.

Example:

good movie

is more meaningful than

good
movie

separately.

""")

# ==================================================
# PART 4 : TF-IDF
# ==================================================

# ==================================================
# Task 7 : TF-IDF Implementation
# ==================================================

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(
    texts
)

print("\nTF-IDF Vocabulary")

print(
    tfidf_vectorizer.vocabulary_
)

print("\nTF-IDF Matrix Shape")

print(
    tfidf_matrix.shape
)

print("\nSample TF-IDF Matrix")

print(
    tfidf_matrix.toarray()[:5]
)

# ==================================================
# Task 8 : BoW vs TF-IDF
# ==================================================

tfidf_scores = np.array(
    tfidf_matrix.sum(axis=0)
).flatten()

tfidf_df = pd.DataFrame({
    "Word":
    tfidf_vectorizer.get_feature_names_out(),
    "TFIDF":
    tfidf_scores
})

tfidf_df = tfidf_df.sort_values(
    by="TFIDF",
    ascending=False
)

print("\nHighest TF-IDF Words")

print(
    tfidf_df.head(10)
)

print("\nLowest TF-IDF Words")

print(
    tfidf_df.tail(10)
)

print("""

Why TF-IDF Down-Weights Common Words?

Words appearing in almost every
document carry less information.

TF-IDF reduces their importance
and gives more weight to unique words.

""")

# ==================================================
# PART 5 : PRACTICAL INSIGHTS
# ==================================================

# ==================================================
# Task 9 : Parameter Exploration
# ==================================================

vectorizer_1 = CountVectorizer(
    max_features=50
)

vectorizer_1.fit(texts)

print("\nVocabulary Size (max_features=50)")

print(
    len(
        vectorizer_1.vocabulary_
    )
)

vectorizer_2 = CountVectorizer(
    min_df=2
)

vectorizer_2.fit(texts)

print("\nVocabulary Size (min_df=2)")

print(
    len(
        vectorizer_2.vocabulary_
    )
)

vectorizer_3 = CountVectorizer(
    max_df=0.8
)

vectorizer_3.fit(texts)

print("\nVocabulary Size (max_df=0.8)")

print(
    len(
        vectorizer_3.vocabulary_
    )
)

print("""

Observations:

max_features limits vocabulary size.

min_df removes rare words.

max_df removes very common words.

""")

# ==================================================
# Task 10 : Conceptual Questions
# ==================================================

print("""

1. Difference Between One-Hot Encoding and BoW

One-Hot:
- Presence or absence only
- No frequency information

BoW:
- Counts frequency of words
- More informative

------------------------------------------------

2. Why N-Grams Increase Dimensionality?

Every new word combination creates
additional features.

More combinations = larger vocabulary.

------------------------------------------------

3. When To Prefer TF-IDF Over BoW?

Use TF-IDF when common words occur
frequently and unique words are more important.

------------------------------------------------

4. Limitations Of Count-Based Vectorization

- High dimensionality
- Sparse matrices
- No semantic meaning
- Ignores word order (BoW)

------------------------------------------------

""")

print("\nAssignment 18 Completed Successfully")
