# DeepPheno: Predicting single gene loss of function phenotypes

DeepPheno is a method for predicting gene-phenotype (HPO classes)
associations from gene functional annotations (GO classes) and gene
expression values.


This repository contains script which were used to build and train the
DeepPheno model together with the scripts for evaluating the model's
performance.

## Dependencies
* The code was developed and tested using python 3.7.
* To install python dependencies run:
  `pip install -r requirements.txt`

## Installation
* To install deeppheno package:
  `pip install deeppheno`

## Data
* https://bio2vec.cbrc.kaust.edu.sa/data/deeppheno/ - Here you can find the data
used to train and evaluate our method.
 * data.tar.gz - data folder with latest dataset
 * data-cafa2.tar.gz - CAFA2 challenge dataset
 * predictions.txt.gz - DeepPheno predictions for human genes

## Running the prediction model
  `deeppheno --data-root <path to data folder> --in-file <input-file>`

## Scripts
The scripts require GeneOntology and Human Phenotype Ontology in OBO Format.
* uni2pandas.py - This script is used to convert data from UniProt
database format to pandas dataframe.
* data.py - This script is used to generate training and
  testing datasets.
* pheno.py - This script is used to train the model
* evaluate_*.py - The scripts are used to compute Fmax, Smin
* GeneDis.groovy - This script is used to compute semantic similarity
  between gene and disease phenotypes

## Training
* Download all the files from https://bio2vec.cbrc.kaust.edu.sa/data/deeppheno/data.tar.gz and place them into data folder
* run `python pheno.py` to start training the model

## Citation
If you use DeepPheno for your research, or incorporate our learning algorithms in your work, please cite:

Maxat Kulmanov, Robert Hoehndorf; DeepPheno: Predicting single gene
knockout phenotypes. BioArxiv, https://doi.org/10.1101/839332

