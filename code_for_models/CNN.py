import pandas as pd
import numpy as np
from math import ceil
from random import randrange
from random import sample
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Dense,
    Input,
    Lambda,
    Concatenate,
    BatchNormalization,
    MaxPooling2D,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    RepeatVector,
    Reshape,
    Activation,
    Dropout,
    Flatten,
    Conv2D,
    Conv1D,
    MaxPooling1D,
    Dot,
    Multiply,
    Add
)
from tensorflow.keras.regularizers import l1, l2
import tensorflow as tf



auc=tf.keras.metrics.AUC(name="roc_auc")




def get_amino_dict():
    amino_dict = dict(
        zip(
            [
                "A",
                "R",
                "N",
                "D",
                "C",
                "E",
                "Q",
                "G",
                "H",
                "I",
                "L",
                "K",
                "M",
                "F",
                "P",
                "S",
                "T",
                "W",
                "Y",
                "V"
            ],
            range(20),
        )
    )
    return amino_dict

def make_code_overhead(input_length):
    amino_dict = get_amino_dict()
    def make_code(sequence):
        code = np.zeros((input_length,20))
        for i,index in enumerate(map(amino_dict.get,sequence)):
            if index is not None:
                code[i%input_length,amino_dict[sequence[i]]] = 1.0
        return code
    return make_code


# In[4]:

def chunks(df, n):
    for i in range(0, len(df), n):
        yield df.iloc[i:i+n],i


# In[5]:


def make_predictions_mask(test_set,channels_first_layer,comp_model):
    N=64
    predictions=np.zeros(test_set.shape[0])
    @tf.function(experimental_relax_shapes=True)
    def predict(t):
        return comp_model(t)
    
    for Xt_temp,ind in chunks(test_set, N):
        sequences=Xt_temp['Sequence']
        max_length = max([len(seq) for seq in sequences])
        temp=np.stack(sequences.apply(make_code_overhead(max_length)).values)
        mask=np.transpose(np.tile(np.sum(temp,axis=2),(channels_first_layer,1,1)),[1,2,0])
        predictions_temp=predict([temp,mask]).numpy()
        predictions[ind:ind+N]=predictions_temp[:,0]
    return predictions


# In[6]:


def data_gen_mask(batch_size,channels_first_layer, train_set, Y):
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
        sequences=train_set['Sequence'].iloc[xi]
        max_length = max([len(seq) for seq in sequences])
        x = np.stack(sequences.apply(make_code_overhead(max_length)).values)
        mask=np.transpose(np.tile(np.sum(x,axis=2),(channels_first_layer,1,1)),[1,2,0])
        y = Y[xi]
        yield [x,mask], [y]


# In[7]:


def data_gen_valid_mask(batch_size,channels_first_layer, valid_set, Y):
    while True:
        for df,ind in chunks(valid_set, batch_size):
            sequences=df['Sequence']
            max_length = max([len(seq) for seq in sequences])
            x = np.stack(sequences.apply(make_code_overhead(max_length)).values)
            mask=np.transpose(np.tile(np.sum(x,axis=2),(channels_first_layer,1,1)),[1,2,0])
            y = Y[ind:ind+batch_size]
            yield [x,mask], [y]


# In[8]:


def define_model_mask():
    amino_inputs = Input(
        shape=(None,20)
    )  # 20 amino acids plus null/beginning/end
    
    mask_input = Input(
        shape=(None,64)
    )
    mask=mask_input

    amino_model_1 = Conv1D(
        64, 7,strides=2, kernel_regularizer=l2(0.000),padding='same', activation="swish"
    )(amino_inputs)
    
    mask=MaxPooling1D(pool_size=2,strides=2,padding='same')(mask)
    amino_model_1=Multiply()([mask,amino_model_1])
    
    
    def unit(channels, amino_model_1,mask,strides=1):
        amino_model = Conv1D(channels,3,strides=strides, kernel_regularizer=l2(0.000)
                             ,padding='same', activation="swish")(amino_model_1)
        amino_model=Multiply()([mask,amino_model])

        amino_model = Conv1D(channels, 3, kernel_regularizer=l2(0.000),padding='same', activation=None)(
            amino_model
        )
        amino_model=Multiply()([mask,amino_model])
        amino_model=Activation("swish")(amino_model)
        return amino_model
    
    for i in range(2):
        channels=64*(2**i)
        if i:
            mask=Concatenate()([mask,mask])
            mask=MaxPooling1D(pool_size=2,strides=2,padding='same')(mask)
            amino_model_1 = Dropout(0.25)(amino_model)
            amino_model=unit(channels, amino_model_1,mask,strides=2)
        else:
            amino_model=unit(channels, amino_model_1,mask)
    
    
    amino_model=Activation("relu")(amino_model)
    model = GlobalMaxPooling1D()(amino_model)


    model = Dense(80, activation="swish")(model)
    model = Dense(50, activation="swish")(model)
    prob = Dense(1, activation="sigmoid", name="prob")(model)
    
    comp_model = models.Model(
        inputs=[amino_inputs,mask_input], outputs=prob,
    )
    comp_model.compile(
        optimizer=optimizers.Adam(), loss=losses.binary_crossentropy, metrics=[auc]
    )
    
    return comp_model


# In[14]:


def run_model_make_pred(train_set,test_set,valid_set,name,suffix,path_to_results):
    BATCH_SIZE=16
    VALID_SIZE=50
    channels_first_layer=64
    Y=train_set['dna_binding'].values
    Yv=valid_set['dna_binding'].values
    list_of_pred=[]
    for n_i,rand_ind in enumerate([42,421,4211]):
        np.random.seed(rand_ind)
        tf.random.set_seed(rand_ind)
        comp_model = define_model_mask()
        stopping_callback = EarlyStopping(
            monitor="val_roc_auc", mode="max", min_delta=0, patience=10, restore_best_weights=False
        )
        callbacks_list = [stopping_callback]
        comp_model.fit(
            data_gen_mask(BATCH_SIZE,channels_first_layer, train_set, Y),
            epochs=1000,
            verbose=2,
            steps_per_epoch=1000,
            validation_data=data_gen_valid_mask(VALID_SIZE,channels_first_layer, valid_set, Yv),
            validation_steps=ceil(valid_set.shape[0]/VALID_SIZE),
            callbacks=callbacks_list,
        )
        predictions=make_predictions_mask(test_set,channels_first_layer,comp_model)
        list_of_pred.append(predictions)
    pred=np.median(np.stack(list_of_pred),axis=0)
    np.save(f'{path_to_results}results_cnn_{name}_{suffix}.npy',pred)




good_species=['escherichia_coli', 'mycobacterium_tuberculosis']


level="species"
PATH_TO_RESULTS='../results/'
PATH_TO_CSV="../data/"
suffix="full" #change for different data set
for name in good_species:
    print(name)
    test_set=pd.read_csv(f"{PATH_TO_CSV}{name}_test_{suffix}.csv")
    train_set=pd.read_csv(f"{PATH_TO_CSV}{name}_train_{suffix}.csv")
    valid_set=pd.read_csv(f"{PATH_TO_CSV}{name}_valid_{suffix}.csv")
    run_model_make_pred(train_set,test_set,valid_set,name,suffix,PATH_TO_RESULTS)




