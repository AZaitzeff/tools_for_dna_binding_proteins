import numpy as np
import pandas as pd


# In[2]:


def get_closest_entry(ser):
    ind=ser['pident'].argmax()
    return ser['sseqid'].iloc[ind]


# In[11]:

suffix="40_1000"  #change for different data set
PATH_TO_CSV="../revised_protein_data/"
PATH_TO_RESULTS='../revised_cnn_results/'
name="random"

columns=["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", "qstart", "qend",
     "sstart", "send", "evalue", "bitscore"]
df_blast_tt=pd.read_csv(f'{PATH_TO_CSV}{name}_test_train_{suffix}.csv', '\t', header=None, names=columns) 
max_df_tt=df_blast_tt[['qseqid','sseqid','pident']].groupby('qseqid').apply(get_closest_entry).reset_index()
train_set=pd.read_csv(f"{PATH_TO_CSV}{name}_train_{suffix}.csv") 
max_df_tt.columns=['org_entry','closest_entry']
df_pred=pd.merge(max_df_tt,train_set[['Entry','dna_binding']],how="left",left_on="closest_entry",right_on="Entry")
df_pred.rename({'dna_binding':'pred'},axis=1,inplace=True)
test_set=pd.read_csv(f"{PATH_TO_CSV}{name}_test_{suffix}.csv") 
pi_pred=pd.merge(test_set,df_pred,how="left",left_on="Entry",right_on="org_entry")
pred=pi_pred['pred'].values
np.save(f'{PATH_TO_RESULTS}results_nn_{name}_{suffix}.npy',pred) 

