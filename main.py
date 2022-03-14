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

    # The finishing condition of "range" discards the remaining tokens of the text that do not add up to "length"
    # E.g. With 61,567 tokens and a length of 100 tokens, the last 67 tokens will be discarded
    for i in range(length-overlap, len(list_tokens)-length, length-overlap):
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


def process_corpus(list_files, list_documents, args, vectorizer, matrix):
    """
    Process the corpus and extract the top n chunks for every document based on their TF-IDF

    :param list_files: list of filenames
    :param list_documents: content of every document as a list of tokens
    :param args: command line arguments (lenght, overlap and n)
    :param vectorizer: model that stores TF-IDF for each token in the corpus
    :param matrix: document x term matrix
    :return: dictionary where filenames (without path) are keys and list of top n chunks are the values
    """
    dict_chunks = {}
    for idx, list_tokens in enumerate(list_documents):
        list_weight = []
        # Stores the list of chunks extracted from the file with length '-l' and overlap '-o'
        list_chunks = extract_chunks(list_tokens, args.length, args.overlap)
        for chunk in list_chunks:
            list_weight.append((calculate_tfidf(vectorizer, matrix, idx, chunk), chunk))
        list_weight = sorted(list_weight, key=itemgetter(0), reverse=True)  # Sort by weight, descending
        dict_chunks[os.path.basename(list_files[idx])] = list_weight[:args.number]  # Store the top n chunks

    return dict_chunks


def show_chunks(dict_chunks):
    """
    Show the top n chunks for each file

    :param dict_chunks: dictionary where keys are filenames and values are the list of top n chunks
    :return: none
    """
    for file in dict_chunks:
        print('File: ' + file)
        for index, chunk in enumerate(dict_chunks[file]):
            print('#{} Score: {}'.format(index + 1, chunk))
        print()


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Process ELTeC corpus.')
    parser.add_argument('folder', help='Name of the folder that stores the files to be processed')
    parser.add_argument('-n', '--number', default=10, type=int, help='Number of chunks to retrieve for each file (10 by default)')
    parser.add_argument('-l', '--length', default=100, type=int, help='Length of each chunk (100 words by default)')
    parser.add_argument('-o', '--overlap',  default=50, type=int, help='Length of overlapping text in consecutive chunks (50 words by default)')
    parser.add_argument('-t', '--type', default='lemma', choices=['lemma', 'word'], help='Type of token to use: lemma or original word (lemma by default)')
    args = parser.parse_args()
    
    list_files = glob.glob(os.path.join(args.folder, '**/*.csv'), recursive=True)   # Check also subfolders
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
    # Process every document and extract the top n chunks
    dict_chunks = process_corpus(list_files, list_documents, args, vectorizer, matrix)
    # Show the top n chunks for each file
    show_chunks(dict_chunks)


if __name__ == '__main__':
    main()
