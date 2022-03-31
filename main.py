import argparse
import operator
from collections import defaultdict
import faiss
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


def process_corpus(dict_documents, args, vectorizer, matrix, strategy, model):
    """
    Process the corpus and extract the top n chunks for every document based on their TF-IDF

    :param dict_documents: content of every document as a dictionary of tokens (keys are filenames)
    :param args: command line arguments (length, overlap and n)
    :param vectorizer: model trained on the corpus and used to get the id of every token in the chunk
    :param matrix: documents (rows) by terms (columns) matrix with TF-IDF values in the cells
    :param strategy: "random" passages or "tfidf"
    :param model: word embedding model to encode the paragraphs
    :return: dictionary where filenames (without path) are keys and list of top n chunks are the values
    """
    dict_passages = {}
    for idx, table_name in enumerate(dict_documents):
        list_tokens = []
        list_passages = extract_passages(dict_documents[table_name], args.length, args.overlap)
        for passage in list_passages:
            if strategy == 'tfidf':
                list_tokens.append((calculate_tfidf(vectorizer, matrix, idx, passage), passage))
            else:  # 'random'
                list_tokens.append(passage)
        if strategy == 'tfidf':
            list_tokens = sorted(list_tokens, key=itemgetter(0), reverse=True)  # Sort by weight, descending
            list_tokens = list(map(operator.itemgetter(1), list_tokens))  # Discard the weights and keep only the tokens
        else:  # 'random'
            random.shuffle(list_tokens)  # Shuffle the passages
        # Encode and store only the top "args.number" passages
        dict_passages[table_name] = model.encode([' '.join(tokens) for tokens in list_tokens[:args.number]])

    return dict_passages


"""        
        dict_passages[table_name] = model.encode()

        for tokens in list_tokens[:args.number]:
            dict_passages[table_name].append(model.encode(' '.join(tokens)))
"""


def create_index(dict_documents, dim):
    # For each vector in the FAISS index store its position and the document it belongs to
    dict_index = {}
    dict_inverted_index = defaultdict(list)
    for language in dict_documents:
        print('Creating FAISS index for "' + language + '"...', end=' ')
        new_index = faiss.IndexFlatL2(dim)
        for document in dict_documents[language]:
            num_vectors = len(dict_documents[language][document])  # Number of vectors for each document
            new_index.add(dict_documents[language][document])  # Index all the vectors for this document
            # Add the document to the inverted index as many times as vectors indexed
            dict_inverted_index[language].extend([document for i in range(num_vectors)])
        dict_index[language] = new_index
        print('OK')

    return dict_index, dict_inverted_index


def main():
    parser = argparse.ArgumentParser(description='Process ELTeC corpus.')
    parser.add_argument('folder', help='Name of the folder that stores the files to be processed')
    parser.add_argument('-n', '--number', default=250, type=int,
                        help='Number of passages to retrieve for each file (250 by default)')
    parser.add_argument('-l', '--length', default=100, type=int, help='Length of each passages (50 words by default)')
    parser.add_argument('-o', '--overlap', default=0, type=int,
                        help='Length of overlapping text in consecutive chunks (no overlap by default)')
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
            # Name of the file without extension used as key to the dictionary
            filename = os.path.splitext(os.path.basename(file))[0]
            language = filename[:3]  # The language is coded in the three first letters of the filename
            list_tokens = []  # Stores the list of tokens in the file
            for line in input_file:
                if line.split():
                    if args.type == 'word':
                        list_tokens.append(line.split()[0])  # Use the original word
                    else:
                        list_tokens.append(line.split()[1])  # Use the lemma
            # {lang_1: {doc_1: [tok_1, tok_2, ... tok_n], doc_2: [tok_1, tok_2, ... tok_n]}, lang_2: {doc_1: ....}}
            dict_documents[language][filename] = list_tokens

    # Load SentenceBERT model
    print('Loading model...', end=' ')
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print('OK')
    # model.max_seq_length = 1000

    # The passages are calculated for each language independently
    for language in dict_documents:
        print('Extracting paragraphs for "' + language + '"...', end=' ')
        # Join the tokens in a single string for each document
        corpus = [' '.join(list_tokens).strip() for list_tokens in dict_documents[language].values()]
        vectorizer = ''
        matrix = ''
        if args.strategy == 'tfidf':
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(lowercase=False)  # Already lowercased
            matrix = vectorizer.fit_transform(corpus)
        # Process every document and extract the word embedding vectors for the top n passages
        dict_documents[language] = process_corpus(dict_documents[language], args, vectorizer, matrix, args.strategy, model)
        # {lang_1: {doc_1: [we_1, we_2, ..., we_n], doc_2: [we_1, we_2, ...], ...}, lang_2: {...}, ...}
        print('OK')

    # Create a FAISS index for each language
    dim = model.get_sentence_embedding_dimension()  # Dimension of the word embedding vector
    dict_index, dict_inverted_index = create_index(dict_documents, dim)

    # Search
    n = 10  # Return the top n passages
    dict_results = defaultdict(dict)
    for origin_language in dict_documents:
        list_languages = list(dict_documents.keys())
        list_languages.remove(origin_language)  # Do not compare a language with itself
        for query_document in dict_documents[origin_language]:
            for target_language in list_languages:
                # Get the n most similar passages for each passage of the "query document"
                # E.g. result for three documents and four passages: [[0 393 363 78][1 555 277 364][2 223 343 121]
                list_weights, list_vectors = dict_index[target_language].search(dict_documents[origin_language][query_document], n)
                # Each query passage returns a list of indexes representing vectors
                for row, vector in enumerate(list_vectors):
                    for column, idx in enumerate(vector):  # Locate the original document based on the index in FAISS
                        target_document = dict_inverted_index[target_language][idx]
                        # Check if the document returned was already return previously
                        if query_document in dict_results[origin_language]:
                            if target_document in dict_results[origin_language][query_document]:
                                dict_results[origin_language][query_document][target_document] += list_weights[row][column]
                            else:
                                dict_results[origin_language][query_document][target_document] = list_weights[row][column]
                        else:
                            dict_results[origin_language][query_document] = defaultdict()
                            dict_results[origin_language][query_document][target_document] = list_weights[row][column]

    print(dict_results)


if __name__ == '__main__':
    main()
