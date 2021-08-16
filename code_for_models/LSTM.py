#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM, GRU, SimpleRNN, Conv1D, MaxPooling1D
import pandas as pd
import numpy as np
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping



auc=metrics.AUC(name="roc_auc")




def trans(str1):
    a = []
    dic = {'A':1,'B':22,'U':23,'J':24,'Z':25,'O':0,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':21}
    for i in range(len(str1)):
        a.append(dic.get(str1[i]))
    return a


# In[4]:


# Embedding
max_features = 26
embedding_size = 128

# Convolution

#filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 70

# Training
batch_size = 128
nb_epoch = 100


def make_model():
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=nb_filter,
                            kernel_size=10,
                            padding='valid',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_length))
    model.add(Conv1D(filters=nb_filter,
                            kernel_size=5,
                            padding='valid',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_length))

    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[auc])
    return model


# In[7]:


def data_gen_mask(batch_size, train_set, Y):
    trues=np.nonzero(Y)[0]
    falses=np.nonzero(1-Y)[0]
    len_T=trues.shape[0]
    len_F=falses.shape[0]
    N=train_set.shape[0]
    while True:
        true_vs_false=np.random.randint(2,size=batch_size)
        xit=trues[np.random.randint(len_T,size=batch_size)]
        xif=falses[np.random.randint(len_F,size=batch_size)]
        xi=xit*true_vs_false+xif*(1-true_vs_false)
        x=(train_set.iloc[xi])['Sequence'].apply(trans)
        x = sequence.pad_sequences(x, maxlen=maxlen)
        y = Y[xi]
        yield [x], [y]



PATH_TO_RESULTS='../revised_cnn_results/'
PATH_TO_CSV="../revised_protein_data/"
suffix="40_1000" #change for different data set
maxlen = 1000 #change for different data set
name="random"

test_set=pd.read_csv(f"{PATH_TO_CSV}{name}_test_{suffix}.csv")
train_set_full=pd.read_csv(f"{PATH_TO_CSV}{name}_train_{suffix}.csv")
X_test=test_set['Sequence'].apply(trans)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
np.random.seed(42)
N=train_set_full.shape[0]
split=[n for n in range(N)]
np.random.shuffle(split)
N20=int(N*.2)
mask_int=np.zeros(N)
for i in range(4):
    mask_int[split[i*N20:(i+1)*N20]]=i
mask_int[split[4*N20:]]=4
for n_i in [0,1,2,3,4]:
    print(name,n_i)
    msk = mask_int!=n_i
    train_set=train_set_full[msk]
    valid_set=train_set_full[~msk]
    Y=train_set['dna_binding'].values
    Yv=valid_set['dna_binding'].values

    model=make_model()
    stopping_callback = EarlyStopping(
            monitor="val_roc_auc", mode="max", min_delta=0, patience=10
        )
    callbacks_list = [stopping_callback]
    X_valid=valid_set['Sequence'].apply(trans)
    X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)
    
    X_train=train_set['Sequence'].apply(trans)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    model.fit(
            x=X_train,
            y=Y,
            batch_size = 128,
            epochs=100,
            verbose=2,
            #steps_per_epoch=ceil(2*np.sum(train_set['dna_binding'])/batch_size),
            #validation_data=(X_valid,Yv),
        )
    predictions=model.predict(X_test)[:,0]
    np.save(f'results_lstm_orig_{name}_{suffix}_{n_i}.npy',predictions)
    predictions=model.predict(X_valid)[:,0]
    np.save(f'results_valid_lstm_orig_{name}_{suffix}_{n_i}.npy',predictions)
    np.save(f'true_valid_lstm_orig_{name}_{suffix}_{n_i}.npy',Yv)


# In[ ]:




