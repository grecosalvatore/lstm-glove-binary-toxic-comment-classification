# LSTM training for Toxic Comment classification

## Introduction
The task consists of predicting wheater the input comment is Toxic or Clean.
## Dataset
The dataset comes from Kaggle and is an extension of the Civil Comments dataset and it can be download at the following [link](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data).
The text of the individual comment is found in the `comment_text` column and each comment has also a `target` column that specifies the toxicity of the text. In this example, the `target` column is used as label:
* target >= 0.5 `Toxic Comment`
* target < 0.5  `Clean Comment`
```python
train = pd.read_csv("train.csv")
train.shape
```
```
> (1804874, 45)
```
Convert the **target** column from continuous values into **labels** 0 or 1.
```python
Y = [1 if x >= 0.5 else 0 for x in train["target"]]
Y = np.array(Y)
```
```python
df= train[['id','comment_text']]
df_labeled = df.assign(label = Y) 
```
```python
df_labeled.head()
```
|       |id      |comment_text                                      |label|
| ----- |:------:|:------------------------------------------------:| ---:|
| 0     | 5967432|amazing this is first time in years i actually... |  0  |
| 1     | 5869644|i know more than the generals trust me            |  0  |
| 2     | 605006 |what is this world coming too how were these t... |  1  |
| 3     | 5094159|it does not matter who wins the leadership of ... |  1  |
| 4     | 450628 |trash sits on the south bank of the willamette... |  0  |

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_NB_WORDS = 20000000

tokenizer = Tokenizer(lower=False, filters='',oov_token="<OOV>",num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)
```

## GloVe
Load the content of the GloVe file into a dictionary with the **word** as **key** and the **Word Embedding** vector as **value**. Using a pre-trained word embedding allows us to use less data and conseguently reduce the training time. In this case it has been used the GloVe with 840B tokens and an embedding vector dimension of 300.
* glove.42B.300d.zip [link](https://nlp.stanford.edu/projects/glove/)
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
Create the Embedding Matrix by assigning to each distinct tokens founded by the tokenizer, the corresponing embedding vector loaded from GloVe.
```python
embedding_matrix = np.zeros((nb_words,300))

for word,i in tqdm(tokenizer.word_index.items()):
    embedding_value = embedding_vector.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value
```

## Define the model
The first layer 
Then follows two Bidirectional LSTM layers with 256 units and a Dense layer with 128 hidden units.
Exploiting transfer learning it is possible to reduce training time and 
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
```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, 320, 300)          40373400  
_________________________________________________________________
bidirectional_3 (Bidirection (None, 320, 512)          1140736   
_________________________________________________________________
dropout_4 (Dropout)          (None, 320, 512)          0         
_________________________________________________________________
bidirectional_4 (Bidirection (None, 512)               1574912   
_________________________________________________________________
dropout_5 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 128)               65664     
_________________________________________________________________
dropout_6 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 129       
=================================================================
Total params: 43,154,841
Trainable params: 2,781,441
Non-trainable params: 40,373,400
_________________________________________________________________
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
