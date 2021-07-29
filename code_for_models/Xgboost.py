from xgboost import XGBClassifier
import pandas as pd
import numpy as np

good_species=['escherichia_coli', 'mycobacterium_tuberculosis']
level="species"
PATH_TO_RESULTS='../results/'
PATH_TO_CSV="../data/"
cols=[f"fb_embed_{i}" for i in range(1280)]
df_fb=pd.read_csv(f"{PATH_TO_CSV}fb_embed.csv")
suffix="full" #change for different data set

for name in good_species:
    list_of_pred=[]
    test_set=pd.read_csv(f"{PATH_TO_CSV}{name}_test_{suffix}.csv") 
    train_set=pd.read_csv(f"{PATH_TO_CSV}{name}_train_{suffix}.csv") 
    valid_set=pd.read_csv(f"{PATH_TO_CSV}{name}_valid_{suffix}.csv") 
    
    train_set=pd.merge(train_set,df_fb,how="left",on="Entry")
    valid_set=pd.merge(valid_set,df_fb,how="left",on="Entry")
    test_set=pd.merge(test_set,df_fb,how="left",on="Entry")
    
    Y=train_set['dna_binding'].values
    Yv=valid_set['dna_binding'].values
    X=train_set[cols].values
    Xv=valid_set[cols].values
    Xt=test_set[cols].values
    model=XGBClassifier(n_jobs=8,scale_pos_weight=sum(1-Y) / sum(Y))
    eval_set = [(Xv, Yv)]
    model.fit(X, Y, early_stopping_rounds=5, eval_metric="auc", eval_set=eval_set, verbose=True)
    pred=model.predict_proba(Xt)[:,1]
    np.save(f'{PATH_TO_RESULTS}results_xgb_{name}_{suffix}.npy',pred) 


# In[ ]:




