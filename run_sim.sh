#!/bin/bash

for i in $(seq 5); do
    #groovy GeneDis.groovy fold${i}_exp-data/gene_annotations.tab fold${i}_exp-data/sim_gene_disease.txt &
    #groovy GeneDis.groovy fold${i}_data/gene_annotations_iea.tab fold${i}_data/sim_gene_disease_iea.txt
    #groovy GeneDis.groovy fold${i}_data/gene_annotations_exp.tab fold${i}_data/sim_gene_disease_exp.txt
    #groovy GeneDis.groovy fold${i}_data/gene_annotations_all.tab fold${i}_data/sim_gene_disease_all.txt
    # python evaluate.py -tsdf data/predictions.pkl_flat.pkl -f $i > fold${i}_data/predictions.pkl_flat.pkl.auc.res &
    # python evaluate.py -tsdf data/predictions_exp.pkl_flat.pkl -f $i > fold${i}_data/predictions_exp.pkl_flat.pkl.auc.res &
    # python evaluate.py -tsdf data/predictions_iea.pkl_flat.pkl -f $i > fold${i}_data/predictions_iea.pkl_flat.pkl.auc.res &
    # python evaluate.py -tsdf data/predictions_all.pkl_flat.pkl -f $i > fold${i}_data/predictions_all.pkl_flat.pkl.auc.res &

    # python evaluate.py -tsdf data/predictions.pkl -f $i > fold${i}_data/predictions.pkl.auc.res &
    # python evaluate.py -tsdf data/predictions_exp.pkl -f $i > fold${i}_data/predictions_exp.pkl.auc.res &
    # python evaluate.py -tsdf data/predictions_iea.pkl -f $i > fold${i}_data/predictions_iea.pkl.auc.res &
    # python evaluate.py -tsdf data/predictions_all.pkl -f $i > fold${i}_data/predictions_all.pkl.auc.res &

    python evaluate_cafa2.py -tsdf data-cafa/predictions.pkl -f $i -rc HP:0000118 > fold${i}_data-cafa/predictions.pkl.auc.organ.res &
    python evaluate_cafa2.py -tsdf data-cafa/predictions.pkl -f $i -rc HP:0000004 > fold${i}_data-cafa/predictions.pkl.auc.onset.res &
    python evaluate_cafa2.py -tsdf data-cafa/predictions.pkl -f $i -rc HP:0000005 > fold${i}_data-cafa/predictions.pkl.auc.inher.res &
    python evaluate_cafa2.py -tsdf data-cafa/predictions.pkl_flat.pkl -f $i -rc HP:0000118 > fold${i}_data-cafa/predictions.pkl_flat.pkl.auc.organ.res &
    python evaluate_cafa2.py -tsdf data-cafa/predictions.pkl_flat.pkl -f $i -rc HP:0000004 > fold${i}_data-cafa/predictions.pkl_flat.pkl.auc.onset.res &
    python evaluate_cafa2.py -tsdf data-cafa/predictions.pkl_flat.pkl -f $i -rc HP:0000005 > fold${i}_data-cafa/predictions.pkl_flat.pkl.auc.inher.res &

done
