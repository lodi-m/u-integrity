import pandas as pd
import  numpy as np
import gensim
from gensim.models import Word2Vec
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.utils.vis_utils import plot_model
import visualkeras

## Kaggle Data
kaggle_data = pd.read_csv("https://raw.githubusercontent.com/lodi-m/u-integrity/main/data/normalized_scores/normalized_kaggle_essay_set.csv")
kaggle_data.head(5)
##print(kaggle_data)

## LSTM Model

def get_lstm():
    model = Sequential()
    model.add(LSTM(200, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 200], return_sequences=True))
    # model.add(LSTM(100, recurrent_dropout=0.4, input_shape=[1, 300]))
    model.add(LSTM(100, recurrent_dropout=0.2))
    model.add(Dropout(0.75))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mse'])
    model.summary()
    

    return model


##Word2Vec
def get_w2v(df, min_word_count, num_features, num_workers, max_distance, downsample):
  w2v = Word2Vec(workers=num_workers, vector_size=num_features, min_count=min_word_count, window=max_distance, sample=downsample)

  w2v.build_vocab(df, progress_per=progress_val)
  w2v.train(df, total_examples=w2v.corpus_count, epochs=w2v.epochs)

  return w2v

##Word embeddings
def make_feature_vectors(words, model, num_features):
    feature_vector = np.zeros((num_features, ))
    index_keys = set(model.wv.index_to_key)
    
    for word in words:
      if word in index_keys:
        feature_vector = np.add(feature_vector, model.wv[word])
    return feature_vector
def avg_feature_vectors(essays, model, num_features):
    final_essay_vector = np.zeros((len(essays), num_features))
    
    for i in range(len(essays)):
        final_essay_vector[i] = make_feature_vectors(essays[i], model, num_features)
    return final_essay_vector

##Parameters
X = kaggle_data["Essay"]
y = kaggle_data["normalized_score"]
     

num_fold = 5
num_features = 200

min_word_count = 40
num_workers = 4
max_distance = 10
downsampling = 1e-3
progress_val = 2000

##Training LSTM
cv = KFold(n_splits=num_fold, shuffle=True)
mae_results = []
mse_results = []
     

for traincv, testcv in cv.split(X):
  X_test, X_train, y_test, y_train = X.iloc[testcv], X.iloc[traincv], y.iloc[testcv], y.iloc[traincv]
  
  w2v = get_w2v(X_train, min_word_count, num_features, num_workers, max_distance, downsampling)

  clean_train_essays = X_train.to_list()
  clean_test_essays = X_test.to_list()

  train_vectors = avg_feature_vectors(clean_train_essays, w2v, num_features)
  test_vectors = avg_feature_vectors(clean_test_essays, w2v, num_features)

  train_vectors = np.array(train_vectors)
  test_vectors = np.array(test_vectors)
  
  train_vectors = np.reshape(train_vectors, (train_vectors.shape[0], 1, train_vectors.shape[1]))
  test_vectors = np.reshape(test_vectors, (test_vectors.shape[0], 1, test_vectors.shape[1]))
  
  lstm_model = get_lstm()
  lstm_model.fit(train_vectors, y_train, batch_size=32, epochs=50)

  y_pred = lstm_model.predict(test_vectors)

  y_pred = np.around(y_pred)

  mse = mean_squared_error(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)

  mse_results.append(mse)
  mae_results.append(mae)

print(np.around(np.array(mae_results).mean(), decimals=3))
print(np.around(np.array(mse_results).mean(), decimals=3))

