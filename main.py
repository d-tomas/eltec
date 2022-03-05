import argparse
from collections import Counter
from operator import itemgetter


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


def weight_chunks(list_chunks, dict_tokens, n):
    """
    Calculate the weight of each chunk based on the frequency of its tokens. Show the top n chunks
    :param list_chunks: list of overlapping chunks extracted from the text
    :param dict_tokens: dictionary of tokens and their frequency in the corpus
    :param n: number of chunks to show
    :return: none
    """
    list_weighted = []
    weight = 0
    
    for chunk in list_chunks:
        for token in chunk:
            weight += dict_tokens[token]
        list_weighted.append((weight, chunk))

    list_weighted = sorted(list_weighted, key=itemgetter(0), reverse=True)  # Sort by weight, descending
    
    # Show the n top weighted chunks
    for index, chunk in enumerate(list_weighted[:n]):
        print('#{} Score: {}'.format(index+1, list_weighted[index][0]))
        print(list_weighted[index][1])
        print()
    

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Process ELTeC corpus.')
    parser.add_argument('file', help='Name of the file to process')
    parser.add_argument('-n', '--number', default=10, type=int, help='Number of chunks to retrieve (10 by default)')
    parser.add_argument('-l', '--length', default=100, type=int, help='Length of each chunk (100 words by default)')
    parser.add_argument('-o', '--overlap',  default=50, type=int, help='Length of overlapping text in consecutive chunks (50 words by default)')
    parser.add_argument('-t', '--type', default='lemma', choices=['lemma', 'word'], help='Type of token to use: lemma or original word (lemma by default)')
    args = parser.parse_args()
    
    list_tokens = []  # Stores the list of tokens in the file
    with open(args.file) as file:
        for line in file:
            if line.split():
                if args.type == 'word':
                    list_tokens.append(line.split()[0])  # Use the original word
                else:
                    list_tokens.append(line.split()[1])  # Use the lemma
        
        list_chunks = extract_chunks(list_tokens, args.length, args.overlap)  # Stores the list of chunks extracted from the file with length '-l' and overlap '-o'
        dict_tokens = Counter(list_tokens)  # Calculate the TF for each token
        weight_chunks(list_chunks, dict_tokens, args.number)  # Weight the chunks by TF and obtain the top n chunks
    

if __name__ == '__main__':
    main()
