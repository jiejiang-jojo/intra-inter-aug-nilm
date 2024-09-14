import h5py
import numpy as np

if __name__ == '__main__':
    test_file_name = 'data_real'
    train_file_name = 'ebike_baseline_12'
    test_file = h5py.File('./data/{}.h5'.format(test_file_name), 'r')
    train_file = h5py.File('./data/{}.h5'.format(train_file_name), 'r')
    output_file = h5py.File('./data/{}_comb.h5'.format(train_file_name), 'w')
    
    for k in train_file.keys():
        house = train_file[k]
        aggregate = house['aggregate'][:]
        em = house['em'][:]

        group = output_file.create_group(k)
        group.create_dataset('aggregate', data=aggregate)
        group.create_dataset('em', data=em)

    for k in test_file.keys():
        house = test_file[k]
        aggregate = house['aggregate'][:]
        em = house['em_data8'][:]

        print(k)

        print(np.mean(aggregate), np.mean(em))
        aggregate += em
        print(np.mean(aggregate))

        group = output_file.create_group(k)
        group.create_dataset('aggregate', data=aggregate)
        group.create_dataset('em', data=em)

    output_file.close()
