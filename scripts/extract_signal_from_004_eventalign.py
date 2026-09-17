#!/usr/bin/env python3
'''
extract signal from f5c eventalign output
the output is of the same format as output 
from file extract_signal.py (which takes RNA002 data as input)
'''

import argparse
import sys
import os
import h5py
import numpy as np
import pysam
from collections import defaultdict
import glob

def extract_read_id_from_group_name(group_name):
    if group_name.startswith('read_'):
        return group_name[5:]  
    return None

def build_read_to_fast5_map(fast5_dir):
    mapping = {}
    fast5_files = glob.glob(os.path.join(fast5_dir, '*.fast5'))
    print(f"{len(fast5_files)} fast5 file found")
    
    for fast5_path in fast5_files:
        try:
            with h5py.File(fast5_path, 'r') as f:
                for read_group_name in f.keys():
                    if not read_group_name.startswith('read_'):
                        continue
                    read_id = extract_read_id_from_group_name(read_group_name)
                    if read_id is not None:
                        mapping[read_id] = fast5_path
        except Exception as e:
            print(f"error when processing {fast5_path}: {e}")
            continue
    
    print(f"{len(mapping)} reads in total")
    return mapping

def get_raw_signal_from_multifast5(fast5_path, target_read_id):
    try:
        with h5py.File(fast5_path, 'r') as f:
            for read_group_name in f.keys():
                if not read_group_name.startswith('read_'):
                    continue
                read_id = extract_read_id_from_group_name(read_group_name)
                if read_id == target_read_id:
                    read_group = f[read_group_name]
                    raw_group = read_group.get('Raw')
                    if raw_group is None:
                        return None
                    signal = raw_group.get('Signal')
                    if signal is None:
                        return None
                    return signal[()]
    except Exception as e:
        print(f"读取 {fast5_path} 失败: {e}")
        return None
    return None

def parse_eventalign(eventalign_file):
    # input: f5c eventalign file
    reads = defaultdict(list)
    with open(eventalign_file) as f:
        header = f.readline().strip().split('\t')
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) <= max(col.values()):
                continue
            read_name = parts[col['read_name']]
            contig = parts[col['contig']]
            strand = parts[col['strand']]
            position = int(parts[col['position']])
            kmer = parts[col['reference_kmer']]
            start_idx = int(parts[col['start_idx']])
            end_idx = int(parts[col['end_idx']])
            reads[read_name].append({
                'contig': contig,
                'strand': strand,
                'position': position,
                'kmer': kmer,
                'start': start_idx,
                'end': end_idx
            })

    for read_name in reads:
        reads[read_name] = sorted(reads[read_name], key=lambda x: x['position'])
    return reads

def kmer_to_base_signals(kmer, signal_segment):
    n_bases = len(kmer)
    n_samples = len(signal_segment)
    if n_samples == 0 or n_bases == 0:
        return [[] for _ in range(n_bases)]
    # 平均分配采样点
    base_len = n_samples // n_bases
    base_signals = []
    for i in range(n_bases):
        start = i * base_len
        end = (i + 1) * base_len if i < n_bases - 1 else n_samples
        base_signals.append(signal_segment[start:end].tolist())
    return base_signals

def extract_signal_main(eventalign_file, sam_file, fast5_dir, output_file):
    print("building read-to-fast5map...")
    read_to_fast5 = build_read_to_fast5_map(fast5_dir)
    
    print("reading eventalign file...")
    reads_events = parse_eventalign(eventalign_file)
    
    print("loading SAM file...")
    sam = pysam.AlignmentFile(sam_file, 'r')
    read_info = {}
    for read in sam:
        if read.is_unmapped:
            continue
        qname = read.query_name
        if qname not in read_info:
            read_info[qname] = {
                'sequence': read.query_sequence,
                'qualities': [c - 33 for c in read.query_qualities] if read.query_qualities else [],
                'contig': read.reference_name,
                'start': read.reference_start + 1
            }
    sam.close()
    print(f"{len(read_info)} mapped reads processed")
    
    out_fh = open(output_file, 'w')
    count = 0
    skipped_no_fast5 = 0
    skipped_no_signal = 0
    
    for read_name, edata in reads_events.items():
        if read_name not in read_info:
            continue
        if read_name not in read_to_fast5: 
            # for testing:
            # read_name: 25d8dea9-62be-4339-84ab-3fc4b2c50e13
            # read_to_fast5: {'21ec890c9d58de2f6abd08a1d862568efdd517c1': 'fast5_dir/RNA241003_Pool1.pod5.bc_31.0_0.fast5'}
            skipped_no_fast5 += 1
            continue
        
        fast5_path = read_to_fast5[read_name]
        
        # here multi_fast5 used, you may change to single_fast5 dependes on your case
        raw_signal = get_raw_signal_from_multifast5(fast5_path, read_name)
        if raw_signal is None:
            skipped_no_signal += 1
            continue
        
        seq = read_info[read_name]['sequence']
        quals = read_info[read_name]['qualities']
        contig = read_info[read_name]['contig']
        start_pos = read_info[read_name]['start']
        
        all_base_signals = []
        for evt in edata:
            seg = raw_signal[evt['start']:evt['end']]
            base_sigs = kmer_to_base_signals(evt['kmer'], seg)
            all_base_signals.extend(base_sigs)
        
        if len(all_base_signals) > len(seq):
            all_base_signals = all_base_signals[:len(seq)]
        elif len(all_base_signals) < len(seq):
            all_base_signals.extend([[] for _ in range(len(seq) - len(all_base_signals))])
        
        signal_str = "|".join(
            "*".join(str(round(s, 4)) for s in base_sig) if base_sig else ""
            for base_sig in all_base_signals
        )
        
        qual_str = "|".join(str(q) for q in quals[:len(seq)])
        
        ref_seq = seq
        
        # output：
        # read_id \t contig \t start \t reference_seq \t base_qualities \t sequence \t signal
        out_line = f"{read_name}\t{contig}\t{start_pos}\t{ref_seq}\t{qual_str}\t{seq}\t{signal_str}\n"
        out_fh.write(out_line)
        count += 1
        
        if count % 100 == 0:
            print(f"{count} reads processed.")
    
    out_fh.close()
    print(f"Finished {count} reads, save to {output_file}")
    if skipped_no_fast5:
        print(f"warning: no corresponding fast5 file for {skipped_no_fast5} reads.")
    if skipped_no_signal:
        print(f"warning: no signal in {skipped_no_signal} reads.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eventalign', required=True)
    parser.add_argument('--sam', required=True)
    parser.add_argument('--fast5_dir', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    extract_signal_main(args.eventalign, args.sam, args.fast5_dir, args.output)

if __name__ == '__main__':
    main()

