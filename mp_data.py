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
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--mp-file', '-mf', default='data/mp.obo',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--mp-annots-file', '-maf', default='data/MGI_GenePheno.rpt',
    help='Mouse Phenotype Ontology annotations')
@ck.option(
    '--deepgo-annots-file', '-daf', default='data/mouse.res',
    help='Predicted go annotations with DeepGOPlus')
@ck.option(
    '--id-mapping-file', '-imf', default='data/MRK_SwissProt.rpt',
    help='Mapping MGI to Swissprot')
@ck.option(
    '--data-file', '-df', default='data/swissprot.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--out-terms-file', '-otf', default='data/terms.pkl',
    help='Terms for prediction')
@ck.option(
    '--out-data-file', '-odf', default='data/mouse.pkl',
    help='Data file')
@ck.option(
    '--min-count', '-mc', default=50,
    help='Min count of HP classes for prediction')
def main(go_file, mp_file, mp_annots_file, deepgo_annots_file, id_mapping_file,
         data_file, out_data_file, out_terms_file, min_count):
    go = Ontology(go_file, with_rels=True)
    logging.info('GO loaded')
    mp = Ontology(mp_file, with_rels=True)
    logging.info('MP loaded')

    logging.info('Load MP2Uniprot mapping')
    prot2gene = {}
    with open(id_mapping_file) as f:
        next(f)
        for line in f:
            it = line.strip().split('\t')
            if it[0] not in gene2prot:
                gene2prot[it[0]] = []
            gene2prot[it[0]] += list(it[6].split())
    logging.info('Loading MP annotations')
    mp_annots = {}
    df = pd.read_pickle(data_file)
    acc2prot = {}
    for row in df.itertuples():
        p_id = row.proteins
        acc_ids = row.accessions.split('; ')
        for acc_id in acc_ids:
            acc2prot[acc_id] = p_id
    with open(mp_annots_file) as f:
        next(f)
        for line in f:
            it = line.strip().split('\t')
            for mgi in it[6].split('|'):
                if mgi not in gene2prot:
                    continue
                prot_ids = gene2prot[mgi]
                mp_id = it[4]
                for prot_id in prot_ids:
                    if prot_id not in acc2prot:
                        continue
                    prot_id = acc2prot[prot_id]
                    if prot_id not in mp_annots:
                        mp_annots[prot_id] = set()
                        if mp.has_term(mp_id):
                            mp_annots[prot_id] |= mp.get_anchestors(mp_id)
    print('MP Annotations', len(mp_annots))
    dg_annots = {}
    gos = set()
    with open(deepgo_annots_file) as f:
        for line in f:
            it = line.strip().split('\t')
            prot_id = it[0]
            annots = []
            for item in it[1:]:
                go_id, score = item.split('|')
                score = float(score)
                annots.append(go_id)
            dg_annots[prot_id] = it[1:]
            gos |= set(annots)
    print('DeepGO Annotations', len(dg_annots))
    print('Number of GOs', len(gos))
    go_df = pd.DataFrame({'gos': list(gos)})
    go_df.to_pickle('data/gos.pkl')
    
    logging.info('Processing annotations')
    
    cnt = Counter()
    annotations = list()
    for prot_id, annots in mp_annots.items():
        for term in annots:
            cnt[term] += 1
    
    
    deepgo_annots = []
    go_annots = []
    mpos = []
    prots = []
    sequences = []
    for row in df.itertuples():
        p_id = row.proteins
        if p_id in mp_annots:
            prots.append(p_id)
            mpos.append(mp_annots[p_id])
            go_annots.append(row.annotations)
            deepgo_annots.append(dg_annots[p_id])
            sequences.append(row.sequences)
            
    prots_set = set(prots)
    for key, val in mp_annots.items():
        if key not in prots_set:
            print(key)
            
    df = pd.DataFrame(
        {'proteins': prots, 'mp_annotations': mpos,
         'go_annotations': go_annots, 'deepgo_annotations': deepgo_annots,
         'sequences': sequences})
    df.to_pickle(out_data_file)
    print(f'Number of proteins {len(df)}')
    
    # Filter terms with annotations more than min_count
    res = {}
    for key, val in cnt.items():
        if key == 'MP:0000001':
            continue
        if val >= min_count:
            ont = key.split(':')[0]
            if ont not in res:
                res[ont] = []
            res[ont].append(key)
    terms = []
    for key, val in res.items():
        print(key, len(val))
        terms += val

    logging.info(f'Number of terms {len(terms)}')
    
    # Save the list of terms
    df = pd.DataFrame({'terms': terms})
    df.to_pickle(out_terms_file)

                


if __name__ == '__main__':
    main()
