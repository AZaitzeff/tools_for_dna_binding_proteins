from xgboost import XGBClassifier
import pandas as pd
import numpy as np

PATH_TO_RESULTS='../revised_cnn_results/'
PATH_TO_CSV="../revised_protein_data/"
cols=[f"fb_embed_{i}" for i in range(1280)]
df_fb=pd.concat([pd.read_csv(f"../dna_binding_protein_esm/fb_embed_not_bac.csv"),pd.read_csv(f"../dna_binding_protein_esm/fb_embed_bac.csv")])
df_fb.drop_duplicates('Sequence',inplace=True)
suffix="40_1000" #change for different data set
name = "random" #change for different data set

test_set_x=pd.read_csv(f"{PATH_TO_CSV}{name}_test_{suffix}.csv") 
train_set_full_x=pd.read_csv(f"{PATH_TO_CSV}{name}_train_{suffix}.csv") 

train_set_full=pd.merge(train_set_full_x[['dna_binding','Sequence']],df_fb,on="Sequence")
test_set=pd.merge(test_set_x[['dna_binding','Sequence']],df_fb,on="Sequence")

list_of_pred=[]
for n_i,rand_ind in enumerate([42,421,4211]):
    print(name,n_i)
    np.random.seed(rand_ind)
    N=train_set_full.shape[0]
    split=[n for n in range(N)]
    np.random.shuffle(split)
    N90=int(N*.9)
    first90=split[:N90]
    last10=split[N90:]
    train_set=train_set_full.iloc[first90]
    valid_set=train_set_full.iloc[last10]
    Y=train_set['dna_binding'].values
    Yv=valid_set['dna_binding'].values
    X=train_set[cols].values
    Xv=valid_set[cols].values
    Xt=test_set[cols].values
    model=XGBClassifier(n_estimators=1000,n_jobs=8,scale_pos_weight=sum(1-Y) / sum(Y))
    eval_set = [(Xv, Yv)]
    model.fit(X, Y, early_stopping_rounds=10, eval_metric="aucpr", eval_set=eval_set, verbose=True)
    predictions=model.predict_proba(Xt)[:,1]
    list_of_pred.append(predictions)
pred=np.median(np.stack(list_of_pred),axis=0)
np.save(f'{PATH_TO_RESULTS}results_xgb_{name}_{suffix}.npy',pred) 



