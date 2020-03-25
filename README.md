# binary-toxicity-classification


## Introduction


## Glove
```python
from tqdm import tqdm

embedding_vector = {}
f = open('glove.840B.300d.txt')
for line in tqdm(f):
    value = line.split(' ')
    word = value[0]
    coef = np.array(value[1:],dtype = 'float32')
    embedding_vector[word] = coef
```

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

```python
history = model.fit(X_pad, Y, validation_split=0.2, nb_epoch=5, batch_size=128)
```
```
Train on 275467 samples, validate on 68867 samples
Epoch 1/5
275467/275467 [==============================] - 2158s 8ms/step - loss: 0.3569 - accuracy: 0.8428 - val_loss: 0.2810 - val_accuracy: 0.8791
Epoch 2/5
275467/275467 [==============================] - 2149s 8ms/step - loss: 0.2700 - accuracy: 0.8853 - val_loss: 0.2603 - val_accuracy: 0.8892
Epoch 3/5
275467/275467 [==============================] - 2133s 8ms/step - loss: 0.2524 - accuracy: 0.8933 - val_loss: 0.2532 - val_accuracy: 0.8917
Epoch 4/5
275467/275467 [==============================] - 2164s 8ms/step - loss: 0.2392 - accuracy: 0.8992 - val_loss: 0.2535 - val_accuracy: 0.8928
Epoch 5/5
275467/275467 [==============================] - 2198s 8ms/step - loss: 0.2236 - accuracy: 0.9054 - val_loss: 0.2512 - val_accuracy: 0.8959
```
