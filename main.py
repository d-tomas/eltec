import argparse
from collections import Counter
import glob
import itertools
import numpy as np
from operator import itemgetter
import os
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_chunks(list_tokens, length, overlap):
    """
    Given a list of tokens, create chunks of overlapping tokens of 'length' words
    :param list_tokens: list containing all the tokens of the text
    :param length: number of words of each chunk
    :param overlap: number of overlapping words between consecutive chunks
    :return: list of chunks extracted
    """
    list_chunks = [list_tokens[0:length]]  # Already store the first chunk

    for i in range(length-overlap, len(list_tokens), length-overlap):
        list_chunks.append(list_tokens[i:i+length])

    return list_chunks


def weight_chunks(list_chunks, dict_tf, dict_idf, n):
    """
    Calculate the weight of each chunk based on the frequency of its tokens. Show the top n chunks
    :param list_chunks: list of overlapping chunks extracted from the text
    :param dict_tf: dictionary of tokens and their frequency (TF) in the corpus
    :param dict_idf: dictionary of tokens and their inverse document frequency (IDF) in the corpus
    :param n: number of chunks to show
    :return: none
    """
    list_weighted = []
    weight = 0
    
    for chunk in list_chunks:
        for token in chunk:
            weight += dict_tf[token]
        list_weighted.append((weight, chunk))

    list_weighted = sorted(list_weighted, key=itemgetter(0), reverse=True)  # Sort by weight, descending
    
    # Show the n top weighted chunks
    for index, chunk in enumerate(list_weighted[:n]):
        print('#{} Score: {}'.format(index+1, list_weighted[index][0]))
        print(list_weighted[index][1])
        print()
    

def calculate_tfidf(vectorizer, matrix, idx, chunk):
    list_weights = []
    for token in chunk:
        try:
            list_weights.append(matrix[idx, vectorizer.vocabulary_[token]])
        except KeyError:
            pass
    return np.array(list_weights).mean()


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Process ELTeC corpus.')
    parser.add_argument('folder', help='Name of the folder that stores the files to be processed')
    parser.add_argument('-n', '--number', default=10, type=int, help='Number of chunks to retrieve for each file (10 by default)')
    parser.add_argument('-l', '--length', default=100, type=int, help='Length of each chunk (100 words by default)')
    parser.add_argument('-o', '--overlap',  default=50, type=int, help='Length of overlapping text in consecutive chunks (50 words by default)')
    parser.add_argument('-t', '--type', default='lemma', choices=['lemma', 'word'], help='Type of token to use: lemma or original word (lemma by default)')
    args = parser.parse_args()
    
    list_files = glob.glob(os.path.join(args.folder, '*.csv'))
    list_documents = []  # Stores the content of every document in the corpus as a list of tokens
    for file in list_files:
        with open(file) as input_file:
            list_tokens = []  # Stores the list of tokens in the file
            for line in input_file:
                if line.split():
                    if args.type == 'word':
                        list_tokens.append(line.split()[0])  # Use the original word
                    else:
                        list_tokens.append(line.split()[1])  # Use the lemma
            # list_documents.append(' '.join(list(itertools.chain.from_iterable(list_tokens))))
            list_documents.append(list_tokens)

    # Create TF-IDF vectorizer
    corpus = [' '.join(list_tokens).strip() for list_tokens in list_documents]  # Join the tokens in a single string for each document
    vectorizer = TfidfVectorizer(lowercase=False)  # Already lowercased
    matrix = vectorizer.fit_transform(corpus)
    # Extract chunks
    for idx, list_tokens in enumerate(list_documents):
        list_weight = []
        list_chunks = extract_chunks(list_tokens, args.length, args.overlap)  # Stores the list of chunks extracted from the file with length '-l' and overlap '-o'
        for chunk in list_chunks:
            list_weight.append((calculate_tfidf(vectorizer, matrix, idx, chunk), chunk))
        list_weight = sorted(list_weight, key=itemgetter(0), reverse=True)  # Sort by weight, descending

        # Show the n top weighted chunks
        for index, chunk in enumerate(list_weight[:args.number]):
            print('#{} Score: {}'.format(index + 1, list_weight[index][0]))
            print(list_weight[index][1])
            print()

    # dict_tf = Counter(list_tokens)  # Calculate the TF for each token
    # weight_chunks(list_chunks, dict_tf, dict_idf, args.number)  # Weight the chunks by TF and obtain the top n chunks
    

if __name__ == '__main__':
    main()
