import numpy as np
import pandas as pd
from goatools.godag.go_tasks import get_go2children
from goatools.obo_parser import GODag


def get_all_children(start):
    godag = GODag('go-basic.obo',optional_attrs={'relationship'})
    optional_relationships = set('part_of')
    children_isa_partof = get_go2children(godag, optional_relationships)
    def get_children(parent):
        family={parent}
        children=children_isa_partof.get(parent,{})
        for child in children:
            family.update(get_children(child))
        return family
    
    return get_children(start)

def go_id_labeler(go_id='GO:0003677'):
    related_go_codes=get_all_children(go_id)
    def go_id_to_indicator(x):
        go_list=x.split('; ')
        return int(any([go_code in related_go_codes for go_code in go_list]))
    return go_id_to_indicator


def sanitize_string(the_string):
    return (the_string.replace(" ", "_")).lower()
def sequences_unsure(df):
    df_mean=df.groupby('Sequence')['dna_binding'].mean().reset_index()
    tol=1e-5
    seqs=df_mean[np.logical_and(df_mean['dna_binding']<(1-tol),df_mean['dna_binding']>tol)]['Sequence'].values
    return seqs

PATH_TO_CSV="data/"
df_bac=pd.read_csv(f'{PATH_TO_CSV}uniprot_data.tab',sep='\t')

GO_CODE='GO:0003677'
label='dna_binding'
df_bac[label]=df_bac['Gene ontology IDs'].apply(go_id_labeler(GO_CODE))
df_bac=df_bac[np.logical_not(df_bac['Sequence'].isin(sequences_unsure(df_bac)))]


species_to_test=['Escherichia coli', 'Mycobacterium tuberculosis']
valid='Bacillus subtilis'


def train_valid_test(df,hold_out,valid,label="dna_binding",level="species",remove_test_data_in_train=True):
    col_name=f'Taxonomic lineage ({level.upper()})'
    train_set=df[df[col_name]!=hold_out][['Entry','Sequence',label,'Length',col_name]]
    test_set=df[df[col_name]==hold_out][['Entry','Sequence',label,'Length',col_name]]

    
    train_set.drop_duplicates('Sequence',inplace=True,ignore_index=True)
    test_set.drop_duplicates('Sequence',inplace=True,ignore_index=True)
    if remove_test_data_in_train:
    #removes Sequences in test set that are in the training set
        test_set=test_set[np.logical_not(test_set['Sequence'].isin(train_set['Sequence']))]
    
    valid_set=train_set[train_set[col_name]==valid]
    train_set=train_set[train_set[col_name]!=valid]
    return train_set,valid_set,test_set


# In[38]:

suffix="full"
a_second_suffix="50"
for name in good_species:
    train_set,valid_set,test_set=train_valid_test(df_bac,name,valid)
    new_name=sanitize_string(name)
    train_set.to_csv(f"{PATH_TO_CSV}{new_name}_train_{suffix}.csv",index=False)
    valid_set.to_csv(f"{PATH_TO_CSV}{new_name}_valid_{suffix}.csv",index=False)
    test_set.to_csv(f"{PATH_TO_CSV}{new_name}_test_{suffix}.csv",index=False)
    train_set,valid_set,test_set=train_valid_test(df_bac,name,valid,remove_test_data_in_train=False)
    test_set.to_csv(f"{PATH_TO_CSV}{new_name}_test_{a_second_suffix}.csv",index=False)


