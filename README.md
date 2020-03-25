# binary-toxicity-classification


## Introduction


## Define the model
```python
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam

model = Sequential()
model.add(Embedding(nb_words,300,weights = [embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable = False))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(256)))
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

```
