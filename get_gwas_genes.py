#!/usr/bin/env python

import click as ck
import pandas as pd
import gzip
from bisect import bisect_left


@ck.command()
def main():
    ref = {}
    genes = {}
    with open('data/hg38_kgXref.txt', 'r') as f:
        for line in f:
            it = line.strip().split('\t')
            x = it[0]
            gene = it[4]
            if gene:
                ref[x] = gene
    with open('data/hg38_knownGene.txt', 'r') as f:
        for line in f:
            it = line.strip().split('\t')
            x = it[0]
            chrom = it[1][3:]
            start = int(it[5])
            end = int(it[6])
            if start < end:
                gene = ref[x]
                if chrom not in genes:
                    genes[chrom] = []
                genes[chrom].append((start, end, gene))
        for chrom in genes:
            values = sorted(genes[chrom], key=lambda x: (x[1], x[0]))
            keys = [x[1] for x in values]
            genes[chrom] = {
                'values': values,
                'keys': keys
            }

    def get_gene_name(variant):
        chrom, pos, _, _ = variant.split(':')
        pos = int(pos)
        if chrom not in genes:
            return 'UNKNOWN'
        values = genes[chrom]['values']
        i = bisect_left(genes[chrom]['keys'], pos)
        if i < len(values) and pos >= values[i][0]:
            return values[i][2]
        return 'UNKNOWN'


        
    gene_list = list()
    with open('data/E03.gwas.imputed_v3.both_sexes.tsv') as f:
        next(f)
        for line in f:
            it = line.strip().split()
            chrom = it[0]
            if chrom == '23':
                chrom = 'X'
            elif chrom == '24':
                chrom = 'Y'
            elif chrom == '25':
                chrom = 'XY'
            elif chrom == '26':
                chrom = 'MT'
            # vt = f'{chrom}:{it[2]}'
            vt = it[0]
            p = float(it[11])
            if p <= 5e-8:
                gene_list.append((get_gene_name(vt), p))
    gene_list = sorted(gene_list, key=lambda x: x[1])
    gene_list = list(map(lambda x: x[0], gene_list))
    df = pd.DataFrame({'genes': list(gene_list)})
    print(df)
    df.to_pickle('data/E03_gwas_genes.pkl')
                
if __name__ == '__main__':
    main()
