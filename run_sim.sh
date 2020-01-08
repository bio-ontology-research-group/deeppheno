#!/bin/bash

for i in $(seq 5); do
	 groovy GeneDis.groovy fold${i}_data/gene_annotations.tab fold${i}_data/sim_gene_disease.txt
	 groovy GeneDis.groovy fold${i}_data/gene_annotations_iea.tab fold${i}_data/sim_gene_disease_iea.txt
	 groovy GeneDis.groovy fold${i}_data/gene_annotations_exp.tab fold${i}_data/sim_gene_disease_exp.txt
	 groovy GeneDis.groovy fold${i}_data/gene_annotations_all.tab fold${i}_data/sim_gene_disease_all.txt
done
