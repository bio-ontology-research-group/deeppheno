#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, FUNC_DICT
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data-cafa/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--hp-file', '-hf', default='data-cafa/hp.obo',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--hp-annots-file', '-haf', default='data-cafa/HPO-t0/hpoa.hp',
    help='Human Phenotype Ontology annotations')
@ck.option(
    '--deepgo-annots-file', '-daf', default='data-cafa/human.res',
    help='Predicted go annotations with DeepGOPlus')
@ck.option(
    '--data-file', '-df', default='data-cafa/swissprot.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--out-terms-file', '-otf', default='data-cafa/terms.pkl',
    help='Terms for prediction')
@ck.option(
    '--out-data-file', '-odf', default='data-cafa/human.pkl',
    help='Data file')
@ck.option(
    '--min-count', '-mc', default=10,
    help='Min count of HP classes for prediction')
def main(go_file, hp_file, hp_annots_file, deepgo_annots_file,
         data_file, out_data_file, out_terms_file, min_count):
    go = Ontology(go_file, with_rels=True)
    print('GO loaded')
    hp = Ontology(hp_file, with_rels=True)
    print('HP loaded')

    iea_annots = {}
    go_annots = {}
    seqs = {}
    df = pd.read_pickle(data_file)
    df = df[df['orgs'] == '9606']
    
    acc2prot = {}
    for i, row in df.iterrows():
        accs = row['accessions'].split('; ')
        p_id = row['proteins']
        for acc in accs:
            acc2prot[acc] = p_id
        if p_id not in go_annots:
            go_annots[p_id] = set()
        if p_id not in iea_annots:
            iea_annots[p_id] = set()
        go_annots[p_id] |= set(row.exp_annotations)
        iea_annots[p_id] |= set(row.iea_annotations)
        seqs[p_id] = row.sequences

    print('GO Annotations', len(go_annots))

    print('Loading HP annotations')
    hp_annots = {}
    unrev = set()
    with open(hp_annots_file) as f:
        next(f)
        for line in f:
            it = line.strip().split('\t')
            acc_id = it[0]
            hp_id = it[1]
            if acc_id not in acc2prot:
                unrev.add(acc_id)
                continue
            p_id = acc2prot[acc_id]
            if p_id not in hp_annots:
                hp_annots[p_id] = set()
            if hp.has_term(hp_id):
                hp_annots[p_id] |= hp.get_anchestors(hp_id)

    print('HP Annotations', len(hp_annots))
    dg_annots = {}
    gos = set()
    with open(deepgo_annots_file) as f:
        for line in f:
            it = line.strip().split('\t')
            p_id = it[0]
            annots = dg_annots.get(p_id, {})
            for item in it[1:]:
                go_id, score = item.split('|')
                score = float(score)
                annots[go_id] = max(score, annots.get(go_id, 0))
            dg_annots[p_id] = annots
            gos |= set(annots.keys())
    print('DeepGO Annotations', len(dg_annots))
    deepgo_annots = {}
    for g_id, annots in dg_annots.items():
        deepgo_annots[g_id] = [go_id + '|' + str(score) for go_id, score in annots.items()]
    print('Number of GOs', len(gos))
    df = pd.DataFrame({'gos': list(gos)})
    df.to_pickle('data-cafa/gos.pkl')

    logging.info('Processing annotations')
    
    cnt = Counter()
    annotations = list()
    for g_id, annots in hp_annots.items():
        for term in annots:
            cnt[term] += 1
    
    
    deepgo_annotations = []
    go_annotations = []
    iea_annotations = []
    hpos = []
    proteins = []
    sequences = []
    for p_id, phenos in hp_annots.items():
        if p_id not in dg_annots:
            continue
        proteins.append(p_id)
        hpos.append(phenos)
        go_annotations.append(go_annots[p_id])
        iea_annotations.append(iea_annots[p_id])
        deepgo_annotations.append(deepgo_annots[p_id])
        sequences.append(seqs[p_id])
        
    df = pd.DataFrame(
        {'proteins': proteins, 'hp_annotations': hpos,
         'go_annotations': go_annotations, 'iea_annotations': iea_annotations,
         'deepgo_annotations': deepgo_annotations,
         'sequences': sequences})
    df.to_pickle(out_data_file)
    print(f'Number of proteins {len(df)}')

    test_annots = {}
    tar2prot = {}
    with open('data-cafa/tar2prot.txt') as f:
        for line in f:
            it = line[1:].strip().split()
            tar2prot[it[0]] = it[1]

    unknown_prots = set()
    with open('data-cafa/benchmark/groundtruth/leafonly_HPO.txt') as f:
        for line in f:
            it = line.strip().split()
            p_id = tar2prot[it[0]]
            if p_id in hp_annots:
                continue
            unknown_prots.add(it[0])
            hp_id = it[1]
            if p_id not in test_annots:
                test_annots[p_id] = set()
            if hp.has_term(hp_id):
                test_annots[p_id] |= hp.get_anchestors(hp_id)
    with open('data-cafa/noknowledge_targets.txt', 'w') as f:
        for t_id in unknown_prots:
            f.write(t_id + '\n')

    deepgo_annotations = []
    go_annotations = []
    iea_annotations = []
    hpos = []
    proteins = []
    sequences = []
    for p_id, phenos in test_annots.items():
        if p_id not in dg_annots:
            continue
        proteins.append(p_id)
        hpos.append(phenos)
        go_annotations.append(go_annots[p_id])
        iea_annotations.append(iea_annots[p_id])
        deepgo_annotations.append(deepgo_annots[p_id])
        sequences.append(seqs[p_id])
    df = pd.DataFrame(
        {'proteins': proteins, 'hp_annotations': hpos,
         'go_annotations': go_annotations,
         'iea_annotations': iea_annotations,
         'deepgo_annotations': deepgo_annotations,
         'sequences': sequences})
    
    df.to_pickle('data-cafa/human_test.pkl')
    print(f'Number of test proteins {len(df)}')

    # Filter terms with annotations more than min_count
    terms_set = set()
    all_terms = []
    for key, val in cnt.items():
        if key == 'HP:0000001':
            continue
        all_terms.append(key)
        if val >= min_count:
            terms_set.add(key)
    terms = []
    for t_id in hp.get_ordered_terms():
        if t_id in terms_set:
            terms.append(t_id)
    
    logging.info(f'Number of terms {len(terms)}')
    # Save the list of terms
    df = pd.DataFrame({'terms': terms})
    df.to_pickle(out_terms_file)

    df = pd.DataFrame({'terms': all_terms})
    df.to_pickle('data-cafa/all_terms.pkl')



if __name__ == '__main__':
    main()
