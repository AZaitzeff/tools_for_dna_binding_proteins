#!/bin/bash
wget -c -O uniprot_data.tab.gz "https://www.uniprot.org/uniprot/?query=taxonomy:2+AND+length:[50 TO 5500]+AND+reviewed:yes&format=tab&columns=id,sequence,go-id,lineage(SPECIES),length&compress=yes"

gunzip uniprot_data.tab.gz

mv uniprot_data.tab data/
