#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Embedding, LSTM, GRU, SimpleRNN, Conv1D,MaxPooling1D,
    Lambda, TimeDistributed, Input, Masking, Bidirectional)
import pandas as pd
import numpy as np
from math import ceil
from random import randrange
from random import sample
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping


# In[2]:


auc=metrics.AUC(name="roc_auc")


# In[3]:


def encode(s1): #translate
    a = []
    dic = {'A':1,'B':22,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,
           'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,
           'V':18,'W':19,'Y':20,'X':21,'U':23,'J':24,'Z':25,'O':0}
    for i in range(len(s1)):
        a.append(dic.get(s1[i]))
    return a


# In[4]:


max_features = 26
embedding_size = 8
# Convolution
nb_filter = 32
pool_length = 2
# BiLSTM
bilstm_output_size = 70
# TrainingSet
batch_size = 128
epochs = 200



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
    model.add(Bidirectional(LSTM(bilstm_output_size, return_sequences=True)))
    model.add(Bidirectional(LSTM(bilstm_output_size)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[auc])
    return model




def data_gen(batch_size, train_set, Y):
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
        x=(train_set.iloc[xi])['Sequence'].apply(encode)
        x = sequence.pad_sequences(x, maxlen=maxlen)
        y = Y[xi]
        yield [x], [y]





good_species=['escherichia_coli', 'mycobacterium_tuberculosis'] #changed for different data set
VALID_SIZE=128
PATH_TO_RESULTS='../results/'
PATH_TO_CSV="../data/" 
suffix="full" #change for different data set
for name in good_species:
    if name=='escherichia_coli':
        maxlen = 5500 #change for different data set
    else:
        maxlen = 5500 #change for different data set
    test_set=pd.read_csv(f"{PATH_TO_CSV}{name}_test_{suffix}.csv")
    train_set=pd.read_csv(f"{PATH_TO_CSV}{name}_train_{suffix}.csv")
    valid_set=pd.read_csv(f"{PATH_TO_CSV}{name}_valid_{suffix}.csv")
    Y=train_set['dna_binding'].values
    Yv=valid_set['dna_binding'].values
    Yt=test_set['dna_binding'].values
    list_of_pred=[]
    for n_i,rand_ind in enumerate([42,421,4211]):
        print(name,n_i)
        X_test=test_set['Sequence'].apply(encode)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        model=make_model()
        stopping_callback = EarlyStopping(
                monitor="val_roc_auc", mode="max", min_delta=0, patience=10, restore_best_weights=False
            )
        callbacks_list = [stopping_callback]
        X_valid=valid_set['Sequence'].apply(encode)
        X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)
        model.fit(
                data_gen(batch_size, train_set, Y),
                epochs=1000,
                verbose=2,
                steps_per_epoch=125,
                validation_data=(X_valid,Yv),
                callbacks=callbacks_list,
            )
        predictions=model.predict(X_test)[:,0]
        list_of_pred.append(predictions)
    pred=np.median(np.stack(list_of_pred),axis=0)
    np.save(f'{PATH_TO_RESULTS}results_bilstm_{name}_{suffix}.npy',pred)


# In[ ]:





# In[ ]:




