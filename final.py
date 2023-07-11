import numpy as np
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import gensim

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocess_text(text):
    sent = decontracted(text)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
    preprocessed_text = sent.lower().strip()
    return preprocessed_text


def word_tokenizer(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def calculate_similarity(text1, text2):
    wordmodelfile = "GoogleNews-vectors-negative300.bin.gz"
    wordmodel = gensim.models.KeyedVectors.load_word2vec_format(wordmodelfile, binary=True)
    
    s1 = preprocess_text(text1)
    s2 = preprocess_text(text2)

    if s1 == s2:
        return 0.0  # 0 means highly similar
    else:
        s1words = word_tokenizer(s1)
        s2words = word_tokenizer(s2)
        
        vocab = wordmodel.key_to_index

        if len(s1words and s2words) == 0:
            return 1.0

        else:
            for word in s1words.copy():
                if word not in vocab:
                    s1words.remove(word)

            for word in s2words.copy():
                if word not in vocab:
                    s2words.remove(word)

            return (1 - wordmodel.n_similarity(s1words, s2words))


def calculate_similarity_dataframe(df):
    similarity_scores = []
    
    for ind, row in df.iterrows():
        text1 = row['text1']
        text2 = row['text2']
        similarity = calculate_similarity(text1, text2)
        similarity_scores.append(similarity)
    
    df['Similarity_score'] = similarity_scores
    return df


# Usage example
text_data = pd.read_csv("Precily_Text_Similarity.csv")
text_data.reset_index(inplace=True)
text_data.rename(columns={'index': 'ID'}, inplace=True)

single_row_data = pd.DataFrame({'text1': ['Text 1 example'], 'text2': ['Text 2 example']})

text_data_processed = calculate_similarity_dataframe(text_data)
single_row_data_processed = calculate_similarity_dataframe(single_row_data)

print(text_data_processed.head())
print(single_row_data_processed.head())