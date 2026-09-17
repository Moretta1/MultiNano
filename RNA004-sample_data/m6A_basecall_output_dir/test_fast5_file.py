# for checking the structure of input fast5 file 
import h5py
with h5py.File('fast5_dir/RNA241003_Pool1.pod5.bc_31.0_0.fast5', 'r') as f:
    for g in f.keys():
        if g.startswith('read_'):
            grp = f[g]
            print("group:", g)
            print("attrs keys:", list(grp.attrs.keys()))
            if 'read_id' in grp.attrs:
                print("read_id attr:", grp.attrs['read_id'])
            if 'read_id' in grp:
                print("read_id dataset:", grp['read_id'][()])
            break
