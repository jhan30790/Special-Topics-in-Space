import h5py

with h5py.File('ran_craters_175000.hdf5', 'r') as f:
    img = f['img_175001']
    print("columns:", [x.decode() for x in img['axis0'][:]])
    print("sample data:", img['block0_values'][:3])
