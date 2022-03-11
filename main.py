import argparse
import glob
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


def calculate_tfidf(vectorizer, matrix, idx, chunk):
    """
    Calculate TF-IDF for a given chunk as the mean of TF-IDF of every token in the chunk
    :param vectorizer: model trained on the corpus and used to get the id of every token in the chunk
    :param matrix: documents (rows) by terms (columns) matrix with TF-IDF values in the cells
    :param idx: identifier of the document
    :param chunk: chunk of text from the document to calculate the TF-IDF
    :return: float value storing the mean TF-IDF of every token in the chunk
    """
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

        # Show the top n weighted chunks
        for index, chunk in enumerate(list_weight[:args.number]):
            print('#{} Score: {}'.format(index + 1, list_weight[index][0]))
            print(list_weight[index][1])
            print()


if __name__ == '__main__':
    main()
