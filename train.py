import pandas as pd
import re
import logging
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
#nltk.download('stopwords')
#nltk.download('punkt')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


# pre processing data
def cleanData(sentence):
    # convert to lowercase, ignore all special characters - keep only
    # alpha-numericals and spaces
    sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())

    # remove stop words
    sentence = " ".join([word for word in sentence.split()
                        if word not in stopwords.words('french')])

    return sentence


df = pd.read_csv('./data/offres_salma.csv', encoding='ISO-8859-1', index_col=0)
print(df.columns)

# drop duplicate rows
df = df.drop_duplicates(subset=['titre','description'])

# clean data
#df['titre'] = df['titre'].map(lambda x: cleanData(x))
df = df.applymap(lambda x: cleanData(x) if isinstance(x, str) else x)

# get array of titles
titles = df['titre'].astype(str).values.tolist()

# tokenize the each title
tok_titles = [word_tokenize(title) for title in titles]

# get array of descriptions
descriptions = df['description'].astype(str).values.tolist()

# tokenize each description
tok_descriptions = [word_tokenize(description) for description in descriptions]

# concatenate tok_titles and tok_descriptions
tok_titles.extend(tok_descriptions)

# refer to here for all parameters:
# https://radimrehurek.com/gensim/models/word2vec.html
model = Word2Vec(tok_titles, sg=1, vector_size=100, window=5, min_count=1, workers=4,
                epochs=100)

#model.train(tok_titles, total_examples=model.corpus_count, epochs=100)
# save model to file
model.save('./data/salma-offres-vectors.model')


# convert model to txt file
modelAsText = KeyedVectors.load('./data/salma-offres-vectors.model')
modelAsText.wv.save_word2vec_format('./data/salma-vectors-francais.txt', fvocab=None, binary=False)