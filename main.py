import argparse
from collections import defaultdict
import glob
import numpy as np
from operator import itemgetter
import os
import random
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_passages(list_tokens, length, overlap):
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


def process_corpus(dict_documents, args, vectorizer, matrix, strategy):
    """
    Process the corpus and extract the top n chunks for every document based on their TF-IDF

    :param dict_documents: content of every document as a dictionary of tokens (keys are filenames)
    :param args: command line arguments (length, overlap and n)
    :param strategy: "random" passages or "tfidf"
    :return: dictionary where filenames (without path) are keys and list of top n chunks are the values
    """
    dict_passages = {}
    for idx, table_name in enumerate(dict_documents):



        list_weight = []
        # Stores the list of chunks extracted from the file with length '-l' and overlap '-o'
        list_passages = extract_passages(list_tokens, args.length, args.overlap)
        for passage in list_passages:
            if strategy == 'tfidf':
                list_weight.append((calculate_tfidf(vectorizer, matrix, idx, passage), passage))
            else:  # 'random'
                list_weight.append((0, passage))  # The weight is irrelevant (random selection)

        if strategy == 'tfidf':
            list_weight = sorted(list_weight, key=itemgetter(0), reverse=True)  # Sort by weight, descending
        else:  # 'random'
            random.shuffle(list_weight)  # Shuffle the passages

        dict_passages[os.path.basename(list_files[idx])] = list_weight[:args.number]  # Store the top n chunks

    return dict_passages


def main():
    parser = argparse.ArgumentParser(description='Process ELTeC corpus.')
    parser.add_argument('folder', help='Name of the folder that stores the files to be processed')
    parser.add_argument('-n', '--number', default=10, type=int,
                        help='Number of passages to retrieve for each file (250 by default)')
    parser.add_argument('-l', '--length', default=100, type=int, help='Length of each passages (50 words by default)')
    parser.add_argument('-t', '--type', default='lemma', choices=['lemma', 'word'],
                        help='Type of token to use: lemma or original word (lemma by default)')
    parser.add_argument('-s', '--strategy', default='random', choices=['random', 'tfidf'],
                        help='Type of strategy followed to extract passages: random ("random") or TF-IDF ("tfidf")')
    args = parser.parse_args()

    list_files = glob.glob(os.path.join(args.folder, '**/*.csv'), recursive=True)   # Check also subfolders
    # For each language, store the list of tokens of each document
    dict_documents = defaultdict(dict)  # Nested dictionary
    for file in list_files:
        with open(file) as input_file:
            filename = os.path.basename(file).split('.')[0]  # Name of the file used as key to the dictionary
            language = filename[:3]  # The language is coded in the three first letters of the filename
            list_tokens = []  # Stores the list of tokens in the file
            for line in input_file:
                if line.split():
                    if args.type == 'word':
                        list_tokens.append(line.split()[0])  # Use the original word
                    else:
                        list_tokens.append(line.split()[1])  # Use the lemma
            dict_documents[language][filename] = list_tokens

    if True:
    #if args.strategy == 'tf-idf':
        # The TF-IDF is calculated for each language independently
        for language in dict_documents:
            # Join the tokens in a single string for each document
            corpus = [' '.join(list_tokens).strip() for list_tokens in dict_documents[language].values()]
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(lowercase=False)  # Already lowercased
            matrix = vectorizer.fit_transform(corpus)
            # Process every document and extract the top n chunks
            dict_documents[language] = process_corpus(dict_documents[language], args, vectorizer, matrix, args.strategy)

    else:  # 'random'
        vectorizer = ''
        matrix = ''

    # Process every document and extract the top n chunks
    dict_chunks = process_corpus(list_files, list_documents, args, vectorizer, matrix)

    # Load SentenceBERT model
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


    # Show the top n chunks for each file
    show_chunks(dict_chunks)


    # Process every document and extract the top n chunks
    dict_chunks = extract_passages_random(list_files, list_documents, args)
    # Show the top n chunks for each file
    show_chunks(dict_chunks)


if __name__ == '__main__':
    main()
