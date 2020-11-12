
import numpy as np
import pandas as pd

PI_LIMIT=50

suffix="full"
a_second_suffix="50"

PATH_TO_CSV="data/"
species_to_test=['escherichia_coli', 'mycobacterium_tuberculosis']
    test_set=pd.read_csv(f"{PATH_TO_CSV}{name}_test_{a_second_suffix}.csv")
    N=max(test_set['Length'])
    columns=["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", "qstart", "qend",
         "sstart", "send", "evalue", "bitscore"]
    df_blast_tt=pd.read_csv(f'{PATH_TO_CSV}{name}_train_test.csv', '\t', header=None, names=columns)
    max_df_tt=df_blast_tt[['qseqid','pident']].groupby('qseqid').max().reset_index()
    train_set=pd.read_csv(f"{PATH_TO_CSV}{name}_train_{suffix}.csv")

    less_50_qseqid=max_df_tt[max_df_tt['pident']<=PI_LIMIT]['qseqid'].values
    new_train_set=pd.DataFrame(train_set[np.logical_and(train_set['Entry'].isin(less_50_qseqid),
                                                        train_set['Length']<=N)])
    new_train_set.to_csv(f"{PATH_TO_CSV}{name}_train_{a_second_suffix}.csv",index=False)

    df_blast_vt=pd.read_csv(f'{PATH_TO_CSV}{name}_valid_test.csv', '\t', header=None, names=columns)
    max_df_vt=df_blast_vt[['qseqid','pident']].groupby('qseqid').max().reset_index()
    valid_set=pd.read_csv(f"{PATH_TO_CSV}{name}_valid_{suffix}.csv")

    less_50_qseqid=max_df_vt[max_df_vt['pident']<=PI_LIMIT]['qseqid'].values
    new_valid_set=pd.DataFrame(valid_set[np.logical_and(valid_set['Entry'].isin(less_50_qseqid),
                                                        valid_set['Length']<=N)])
    new_valid_set.to_csv(f"{PATH_TO_CSV}{name}_valid_{a_second_suffix}.csv",index=False)


# In[ ]:




