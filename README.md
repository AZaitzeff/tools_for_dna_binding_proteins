# Tools for DNA-binding proteins

To use the same data sets from the paper, unzip the zip files in the data folder. 

## Generating the species data from the paper
Running the code in `make_data_for_species_test.py` will generate the species data sets for the paper. Combine the bac and euk training sets to get the combined data set. Use the test set correspnding to the species kingdom in this case. 

## Generating the tables of the results
The predictions of the models on the test sets used in the paper are in the results folder. 

This jupyter notebook `see_results.ipynb` contains the code for the metrics in the paper and can output all the figures used in the paper.

## Running the models
For all the models, the data sets are determined by the `name` and `suffix` variables. 

### Nearest Neighbors 
- Run `nearest_nearbor_generation.py` to generate the fasfa files and bash scripts for blast. 
- Run the generated `blast` scripts
- Run NN.py in code_for_models

### LSTM, biLSTM
 Both of them assume that you have TensorFlow 2 

### XGBoost
You need the embedding by ESM in a datafame to run Xgboost.py. For the provided data sets, please contact me for the embeddings and they need to go in your data folder under the name `fb_embed.csv`. 

## How to generate your own data sets

### run get_data.sh

This bash script pulls data from Uniprot into the data folder. The Uniprot programmatic access reference is https://www.uniprot.org/help/api_queries. 
The possible arguments for the query and columns can be found at https://www.uniprot.org/help/uniprotkb_column_names.
I will explain the arguments used in my query
- `taxonomy:2`: Get only proteins from bacteria
- `length:[50 TO 5500]`: Sequences with a length between 50 and 5500
- `reviewed:yes`: Only reviewed proteins
- `columns=id,sequence,go-id,lineage(SPECIES),length`: Which columns to include in the resulting file

### run make_data.py
If you would like to predict a different molecular function
- Change `GO_CODE` if you want to indicate a 
- Should probably also change the `label`

Would like a different split and naming convention
- Change the arguments into `train_valid_test` at the end of the program
- Change `suffix` and 'a_second_suffix'

Now, this makes the full data set. Follow the next steps to generate data sets where the training and validation set only contains sequences that are far away (as measured by BLAST percent identity) from the testing set

### download BLAST into the blast folder

https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download

If you download the software somewhere else change the generated paths the `script` variable in dftofasfa.py.

### Run dftofasfa.py
This function generates 
- fasfa files from the csv files created by make_data.py
- scripts to run the blast code 

### Run blastp
- dftofasfa.py should generate scripts in the blast folder
- Run these files
- The files take about 24 hours to run. 

### Run make_data_train_test_far_from_test.py
- this file generates the training and validation set that only contains sequences that are far away (as measured by BLAST percent identity) from the testing set
- You can change how far away this limit is by changing the `PI_LIMIT` variable
