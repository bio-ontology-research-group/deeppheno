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
    '--hp-file', '-hf', default='data/hp.obo',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--hp-annots-file', '-haf', default='data/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt',
    help='Human Phenotype Ontology annotations')
@ck.option(
    '--deepgo-annots-file', '-daf', default='data/human.res',
    help='Predicted go annotations with DeepGOPlus')
@ck.option(
    '--id-mapping-file', '-imf', default='data/gene2prot.tab',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--data-file', '-df', default='data/swissprot.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--string-mapping-file', '-smf', default='data/string2uni.tab',
    help='ELEmbeddings')
@ck.option(
    '--expressions-file', '-ef', default='data/E-MTAB-5214-query-results.tpms.tsv',
    help='ELEmbeddings')
@ck.option(
    '--out-terms-file', '-otf', default='data/terms.pkl',
    help='Terms for prediction')
@ck.option(
    '--out-data-file', '-odf', default='data/human.pkl',
    help='Data file')
@ck.option(
    '--min-count', '-mc', default=10,
    help='Min count of HP classes for prediction')
def main(go_file, hp_file, hp_annots_file, deepgo_annots_file, id_mapping_file,
         data_file, string_mapping_file, expressions_file,
         out_data_file, out_terms_file, min_count):
    go = Ontology(go_file, with_rels=True)
    print('GO loaded')
    hp = Ontology(hp_file, with_rels=True)
    print('HP loaded')
    print('Load Gene2prot mapping')
    df = pd.read_pickle(data_file)
    prot2gene = {}
    with open(id_mapping_file) as f:
        next(f)
        for line in f:
            it = line.strip().split('\t')
            prot2gene[it[2]] = it[0]
    # for row in df.itertuples():
    #     if row.genes != '':
    #         prot2gene[row.proteins] = row.genes
    
    print('Loading HP annotations')
    hp_annots = {}
    name2gene = {}
    with open(hp_annots_file) as f:
        next(f)
        for line in f:
            it = line.strip().split('\t')
            gene_id = it[0]
            gene_name = it[1].upper()
            name2gene[gene_name] = gene_id
            hp_id = it[3]
            if gene_id not in hp_annots:
                hp_annots[gene_id] = set()
            if hp.has_term(hp_id):
                hp_annots[gene_id] |= hp.get_anchestors(hp_id)
    total_annots = 0
    for g_id, annots in hp_annots.items():
        total_annots += len(annots)
    print('HP Annotations', len(hp_annots), total_annots, (total_annots / len(hp_annots)))
    dg_annots = {}
    gos = set()
    with open(deepgo_annots_file) as f:
        for line in f:
            it = line.strip().split('\t')
            if it[0] not in prot2gene:
                continue
            gene_id = prot2gene[it[0]]
            annots = dg_annots.get(gene_id, {})
            for item in it[1:]:
                go_id, score = item.split('|')
                score = float(score)
                annots[go_id] = max(score, annots.get(go_id, 0))
            dg_annots[gene_id] = annots
            gos |= set(annots.keys())
    print('DeepGO Annotations', len(dg_annots))
    deepgo_annots = {}
    for g_id, annots in dg_annots.items():
        deepgo_annots[g_id] = [go_id + '|' + str(score) for go_id, score in annots.items()]
    print('Number of GOs', len(gos))
    gos_df = pd.DataFrame({'gos': list(gos)})
    # gos_df.to_pickle('data/gos.pkl')

    go_annots = {}
    iea_annots = {}
    seqs = {}
    
    for i, row in df.iterrows():
        if row.proteins not in prot2gene:
            continue
        g_id = prot2gene[row.proteins]
        if g_id not in go_annots:
            go_annots[g_id] = set()
            iea_annots[g_id] = set()
        go_annots[g_id] |= set(row.exp_annotations)
        iea_annots[g_id] |= set(row.iea_annotations)
        seqs[g_id] = row.sequences

    print('GO Annotations', len(go_annots))
    logging.info('Processing annotations')
    
    cnt = Counter()
    annotations = list()
    for g_id, annots in hp_annots.items():
        for term in annots:
            cnt[term] += 1

    gene_exp = {}
    max_val = 0
    with open(expressions_file) as f:
        for line in f:
            if line.startswith('#') or line.startswith('Gene'):
                continue
            it = line.strip().split('\t')
            gene_name = it[1].upper()
            if gene_name in name2gene:
                exp = np.zeros((53,), dtype=np.float32)
                for i in range(len(it[2:])):
                    exp[i] = float(it[2 + i]) if it[2 + i] != '' else 0.0
                gene_exp[name2gene[gene_name]] = exp / np.max(exp)
                
    print('Expression values', len(gene_exp))
    
    deepgo_annotations = []
    go_annotations = []
    iea_annotations = []
    hpos = []
    genes = []
    sequences = []
    expressions = []
    mis_exp = 0
    for g_id, phenos in hp_annots.items():
        if g_id not in dg_annots:
            continue
        genes.append(g_id)
        hpos.append(phenos)
        go_annotations.append(go_annots[g_id])
        iea_annotations.append(iea_annots[g_id])
        deepgo_annotations.append(deepgo_annots[g_id])
        sequences.append(seqs[g_id])
        if g_id in gene_exp:
            expressions.append(gene_exp[g_id])
        else:
            expressions.append(np.zeros((53,), dtype=np.float32))
            mis_exp += 1
    print('Missing expressions', mis_exp)
    
    # for g_id, gos in dg_annots.items():
    #     genes.append(g_id)
    #     phenos = set()
    #     if g_id in hp_annots:
    #         phenos = hp_annots[g_id]
    #     hpos.append(phenos)
    #     go_annotations.append(go_annots[g_id])
    #     iea_annotations.append(iea_annots[g_id])
    #     deepgo_annotations.append(deepgo_annots[g_id])
    #     sequences.append(seqs[g_id])
    
        
    df = pd.DataFrame(
        {'genes': genes, 'hp_annotations': hpos,
         'go_annotations': go_annotations, 'iea_annotations': iea_annotations,
         'deepgo_annotations': deepgo_annotations,
         'sequences': sequences, 'expressions': expressions})
    df.to_pickle(out_data_file)
    print(f'Number of proteins {len(df)}')
    print(df)
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
    labels = []
    for t_id in hp.get_ordered_terms():
        if t_id in terms_set:
            terms.append(t_id)
            labels.append(hp.get_term(t_id)['name'])
    
    logging.info(f'Number of terms {len(terms)}')
    # Save the list of terms
    # df = pd.DataFrame({'terms': terms, 'labels': labels})
    # df.to_pickle(out_terms_file)
    # df = pd.DataFrame({'terms': all_terms})
    # df.to_pickle('data/all_terms.pkl')
                


if __name__ == '__main__':
    main()
