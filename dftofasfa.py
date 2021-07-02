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
a_second_suffix="50"

PATH_TO_CSV="data/"
PATH_TO_BLAST="blast/"
PATH_TO_FASTA="blast/data/"
good_species=['escherichia_coli', 'mycobacterium_tuberculosis']
level="species"
for name in good_species:
    test_set=pd.read_csv(f"{PATH_TO_CSV}{name}_test_{a_second_suffix}.csv")
    train_set=pd.read_csv(f"{PATH_TO_CSV}{name}_train_{suffix}.csv")
    valid_set=pd.read_csv(f"{PATH_TO_CSV}{name}_valid_{suffix}.csv")

    dftofasfa(train_set,f'{PATH_TO_FASTA}{name}_train.fsa')
    dftofasfa(valid_set,f'{PATH_TO_FASTA}{name}_valid.fsa')
    dftofasfa(test_set,f'{PATH_TO_FASTA}{name}_test.fsa')


# In[4]:


for name in good_species:
    script=f"""#!/bin/bash
ncbi-blast-2.10.1+/bin/makeblastdb -in data/{name}_test.fsa -title "{name} Test DNA" -dbtype prot -parse_seqids

ncbi-blast-2.10.1+/bin/blastp -query data/{name}_train.fsa -db data/{name}_test.fsa -max_target_seqs 5 -outfmt 6 -out {name}_train_test.csv
ncbi-blast-2.10.1+/bin/blastp -query data/{name}_valid.fsa -db data/{name}_test.fsa -max_target_seqs 5 -outfmt 6 -out {name}_valid_test.csv

mv {name}_train_test.csv ../data/
mv {name}_valid_test.csv ../data/
"""
    with open(f'{PATH_TO_BLAST}blast_{name}.sh','w') as file:
        file.write(script)


# In[ ]:




