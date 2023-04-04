import nltk
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from gensim.models import Word2Vec

nltk.download('stopwords')
stop = stopwords.words('english')

def data_cleaning(data): 
    # Remove the tagged labels and word tokenize the sentence
    data["essay"] = data["essay"].apply(lambda x: re.sub("\s\[.*\]|@(\w+)", "", x))
    data["essay"] = data["essay"].apply(lambda x: re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", x))
    data["essay"] = data["essay"].apply(lambda x: re.sub("\s+", " ", x))

    # lowercase
    data["essay"] = data["essay"].apply(lambda x: x.lower())

    # Removing Stopwords
    data['essay'] = data['essay'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))


data = pd.read_csv("kaggle_essay_set_8.csv", encoding = 'unicode_escape')
data_cleaning(data)
data.to_csv("cleaned_kaggle_essay_set_8.csv")