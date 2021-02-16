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


def sequences_unsure(df):
    df_mean=df.groupby('Sequence')['dna_binding'].mean().reset_index()
    tol=1e-5
    seqs=df_mean[np.logical_and(df_mean['dna_binding']<(1-tol),df_mean['dna_binding']>tol)]['Sequence'].values
    return seqs

def train_test(df,hold_out,label="dna_binding",level="species",remove_test_data_in_train=True,max_length=-1):
    col_name=f'Taxonomic lineage ({level.upper()})'
    if max_length>0:
        df=df[df['Length']<=max_length]
    train_set=df[df[col_name]!=hold_out][['Entry','Sequence',label,'Length',col_name]]
    test_set=df[df[col_name]==hold_out][['Entry','Sequence',label,'Length',col_name]]

    
    train_set.drop_duplicates('Sequence',inplace=True,ignore_index=True)
    test_set.drop_duplicates('Sequence',inplace=True,ignore_index=True)
    if remove_test_data_in_train:
    #removes Sequences in test set that are in the training set
        test_set=test_set[np.logical_not(test_set['Sequence'].isin(train_set['Sequence']))]
    
    return train_set,test_set

PATH_TO_CSV="data/"
df_bac=pd.read_csv(f'{PATH_TO_CSV}uniprot_data.tab',sep='\t')
df_bac.dropna(subset=['Sequence', 'Gene ontology IDs',
                      'Taxonomic lineage (SPECIES)'],inplace=True)

GO_CODE='GO:0003677'
label='dna_binding'
df_bac[label]=df_bac['Gene ontology IDs'].apply(go_id_labeler(GO_CODE))
df_bac=df_bac[np.logical_not(df_bac['Sequence'].isin(sequences_unsure(df_bac)))]

df_not_bac=pd.read_csv(f'{PATH_TO_CSV}uniprot_data_not_bac.tab',sep='\t')
df_not_bac.dropna(subset=['Sequence', 'Gene ontology IDs',
                      'Taxonomic lineage (SPECIES)'],inplace=True)
df_not_bac['dna_binding']=df_not_bac['Gene ontology IDs'].apply(go_id_labeler(GO_CODE))
df_not_bac=df_not_bac[np.logical_not(df_not_bac['Sequence'].isin(sequences_unsure(df_not_bac)))]

names=['Escherichia coli','Salmonella enterica (Salmonella choleraesuis)',
       'Staphylococcus aureus','Mycobacterium tuberculosis','Bacillus subtilis',
       "Streptococcus pyogenes","Bacillus cereus",
      "Streptococcus pneumoniae"]
new_names=['ecoli','sal','staph','tb','bacs','strepy',"bacc","strepn"]

suffix="bac"
a_second_suffix="euk"
for name,new_name in zip(names,new_names):
    train_set,test_set=train_test(df_bac,name)
    train_set.to_csv(f"{PATH_TO_CSV}{new_name}_train_{suffix}.csv",index=False)
    test_set.to_csv(f"{PATH_TO_CSV}{new_name}_test_{suffix}.csv",index=False)
    train_set,test_set=train_test(df_bac,name,remove_test_data_in_train=False)
    train_set=df_not_bac.drop_duplicates('Sequence',ignore_index=True)
    train_set.to_csv(f"{PATH_TO_CSV}{new_name}_train_{a_second_suffix}.csv",index=False)
    test_set.to_csv(f"{PATH_TO_CSV}{new_name}_test_{a_second_suffix}.csv",index=False)

names=['Homo sapiens (Human)','Arabidopsis thaliana (Mouse-ear cress)',
       'Mus musculus (Mouse)','Oryza sativa (Rice)','Rattus norvegicus (Rat)',
       "Saccharomyces cerevisiae (Baker's yeast)","Xenopus laevis (African clawed frog)",
      "Drosophila melanogaster (Fruit fly)"]
new_names=['human','mouse_ear','mouse','rice','rat','yeast',"frog","fly"]

suffix="euk"
a_second_suffix="bac"
for name,new_name in zip(names,new_names):
    train_set,test_set=train_test(df_not_bac,name)
    train_set.to_csv(f"{PATH_TO_CSV}{new_name}_train_{suffix}.csv",index=False)
    test_set.to_csv(f"{PATH_TO_CSV}{new_name}_test_{suffix}.csv",index=False)
    train_set,test_set=train_test(df_not_bac,name,remove_test_data_in_train=False)
    train_set=df_bac.drop_duplicates('Sequence',ignore_index=True)
    train_set.to_csv(f"{PATH_TO_CSV}{new_name}_train_{a_second_suffix}.csv",index=False)
    test_set.to_csv(f"{PATH_TO_CSV}{new_name}_test_{a_second_suffix}.csv",index=False)
