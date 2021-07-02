#!/usr/bin/env python
# coding: utf-8

import pandas as pd


def dftofasfa(df,filename):
    num=df.shape[0]
    with open(filename, 'w') as writer:
        for i in range(num):
            entry=df['Entry'].iloc[i]
            seq=df['Sequence'].iloc[i]
            if i:
                writer.write(f'\n>{entry}\n{seq}')
            else:
                writer.write(f'>{entry}\n{seq}')


# In[3]:
suffix="full"

PATH_TO_CSV="data/"
PATH_TO_BLAST="blast/"
PATH_TO_FASTA="blast/data/"
good_species=['escherichia_coli', 'mycobacterium_tuberculosis']
level="species"
for name in good_species:
    test_set=pd.read_csv(f"{PATH_TO_CSV}{name}_test_{suffix}.csv")
    train_set=pd.read_csv(f"{PATH_TO_CSV}{name}_train_{suffix}.csv")
    valid_set=pd.read_csv(f"{PATH_TO_CSV}{name}_valid_{suffix}.csv")
    train_set=pd.concat([train_set,valid_set])
    dftofasfa(train_set,f'{PATH_TO_FASTA}{name}_train.fsa')
    dftofasfa(test_set,f'{PATH_TO_FASTA}{name}_test.fsa')


# In[4]:


for name in good_species:
    script=f"""#!/bin/bash

ncbi-blast-2.10.1+/bin/makeblastdb -in data/{name}_train.fsa -title "{name} Train DNA" -dbtype prot -parse_seqids

ncbi-blast-2.10.1+/bin/blastp -query data/{name}_test.fsa -db data/{name}_train.fsa -max_target_seqs 5 -outfmt 6 -out {name}_test_train_{suffix}.csv

mv {name}_test_train_{suffix}.csv ../data/
"""
    with open(f'{PATH_TO_BLAST}blast_{name}.sh','w') as file:
        file.write(script)


# In[ ]:




