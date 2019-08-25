#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd

from utils import Ontology

@ck.command()
@ck.option(
    '--go-mappings-file', '-gmf', default='data/go_mappings.txt',
    help='GO definitions mappings')
@ck.option(
    '--hp-mappings-file', '-hmf', default='data/hp_mappings.txt',
    help='HP definitions mappings')
@ck.option(
    '--out-file', '-of', default='data/go2hp.txt',
    help='Result file with a list of terms for prediction task')
def main(go_mappings_file, hp_mappings_file, out_file):
    w = open(out_file, 'w')
    go_mappings = {}
    with open(go_mappings_file, 'r') as f:
        for line in f:
            it = line.strip().split('\t')
            for t_id in it[1:]:
                t_id = t_id.strip()
                ont = t_id.split('_')[0]
                if ont not in set(['UBERON', 'CL', 'CHEBI']):
                    continue
                if t_id not in go_mappings:
                    go_mappings[t_id] = []
                go_mappings[t_id].append(it[0].strip())
    hp_mappings = {}
    with open(hp_mappings_file, 'r') as f:
        for line in f:
            it = line.strip().split('\t')
            if line.find('PATO_0000470') != -1 or line.find('PATO_0001997') != -1:
                continue
            for t_id in it[1:]:
                t_id = t_id.strip()
                ont = t_id.split('_')[0]
                if ont not in set(['GO', 'UBERON', 'CL', 'CHEBI']):
                    continue
                if ont == 'GO':
                    if t_id not in hp_mappings:
                        hp_mappings[t_id] = set()
                    hp_mappings[t_id].add(it[0].strip())
                elif t_id in go_mappings:
                    print(t_id)
                    for g_id in go_mappings[t_id]:
                        if g_id not in hp_mappings:
                            hp_mappings[g_id] = set()
                        hp_mappings[g_id].add(it[0].strip())
               
                            
    with open(out_file, 'w') as f:
        for go_id, hp_ids in hp_mappings.items():
            w.write(go_id)
            for hp_id in hp_ids:
                w.write('\t' + hp_id)
            w.write('\n')
    
            
                
    

                


if __name__ == '__main__':
    main()
