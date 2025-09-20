import nltk
from nltk.tokenize import TweetTokenizer
from collections import Counter
import string

nltk.download('punkt')

input_file_path = 'test01_cc_sharealike.txt'

with open(input_file_path, 'r', encoding='utf-8') as file:
    text = file.read()
    
tokenizer = TweetTokenizer()
tokens = tokenizer.tokenize(text)

filtered_tokens = [
    token.lower() for token in tokens
    if token not in string.punctuation
]

word_counts = Counter(filtered_tokens)

top_10 = word_counts.most_common(10)

for word, count in top_10:
    print(f"{word}\t{count}")
    
    