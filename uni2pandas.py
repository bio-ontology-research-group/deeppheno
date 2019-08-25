#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from utils import Ontology, is_exp_code, FUNC_DICT

logging.basicConfig(level=logging.INFO)

ORGS = set(['HUMAN', 'MOUSE', ])

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--uniprot-file', '-uf', default='data/uniprot_sprot.dat.gz',
    help='UniProt knowledgebase file in text format (archived)')
@ck.option(
    '--filter_exp', '-fe', is_flag=True,
    help='Filter proteins with experimental annotations')
@ck.option(
    '--prop-annots', '-pa', is_flag=True,
    help='Propagate annotations with GO structure')
@ck.option(
    '--out-file', '-o', default='data/swissprot.pkl',
    help='Result file with a list of proteins, sequences and annotations')
def main(go_file, uniprot_file, filter_exp, prop_annots, out_file):
    go = Ontology(go_file, with_rels=True)

    proteins, accessions, sequences, annotations, interpros, orgs = load_data(uniprot_file)
    df = pd.DataFrame({
        'proteins': proteins,
        'accessions': accessions,
        'sequences': sequences,
        'annotations': annotations,
        'interpros': interpros,
        'orgs': orgs
    })

    # Filter proteins
    df = df[df['orgs'] == '9606']
    
    logging.info('Filtering proteins with experimental annotations')
    index = []
    annotations = []
    iea_annotations = []
    for i, row in enumerate(df.itertuples()):
        annots = set()
        iea_annots = set()
        for annot in row.annotations:
            go_id, code = annot.split('|')
            anch_set = go.get_anchestors(go_id)
            if is_exp_code(code):
                annots |= anch_set
            iea_annots |= anch_set
        annots = list(annots)
        iea_annots = list(iea_annots)
        annotations.append(annots)
        iea_annotations.append(iea_annots)
    df['exp_annotations'] = annotations
    df['iea_annotations'] = iea_annotations

    
    df.to_pickle(out_file)
    logging.info('Successfully saved %d proteins' % (len(df),) )
    
def load_data(uniprot_file):
    proteins = list()
    accessions = list()
    sequences = list()
    annotations = list()
    interpros = list()
    orgs = list()
    with gzip.open(uniprot_file, 'rt') as f:
        prot_id = ''
        prot_ac = ''
        seq = ''
        org = ''
        annots = list()
        ipros = list()
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '':
                    proteins.append(prot_id)
                    accessions.append(prot_ac)
                    sequences.append(seq)
                    annotations.append(annots)
                    interpros.append(ipros)
                    orgs.append(org)
                prot_id = items[1]
                annots = list()
                ipros = list()
                seq = ''
            elif items[0] == 'AC' and len(items) > 1:
                prot_ac = items[1]
            elif items[0] == 'OX' and len(items) > 1:
                if items[1].startswith('NCBI_TaxID='):
                    org = items[1][11:]
                    end = org.find(' ')
                    org = org[:end]
                else:
                    org = ''
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    code = items[3].split(':')[0]
                    annots.append(go_id + '|' + code)
                if items[0] == 'InterPro':
                    ipro_id = items[1]
                    ipros.append(ipro_id)
            elif items[0] == 'SQ':
                seq = next(f).strip().replace(' ', '')
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq


        proteins.append(prot_id)
        accessions.append(prot_ac)
        sequences.append(seq)
        annotations.append(annots)
        interpros.append(ipros)
        orgs.append(org)
    return proteins, accessions, sequences, annotations, interpros, orgs


if __name__ == '__main__':
    main()
