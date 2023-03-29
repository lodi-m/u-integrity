import nltk
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


student_data = pd.read_csv(filepath, usecols=[col0, col1, col2, col3, col4, col4], nrows=900 )
df = pd.DataFrame(student_data)
##print(df)

df[col2] = df[col2].str.lower()

stop_words = stopwords.words('english')
df[col2] = df[col2].apply(lambda x: ' '.join([words for words in x.split() if words not in (stop_words)]))
stemmer = PorterStemmer()
df[col2] = df[col2].apply(lambda x: ' '.join([stemmer.stem(words) for words in x.split()]))

df.to_csv(filepath, index=False)

print(df)


