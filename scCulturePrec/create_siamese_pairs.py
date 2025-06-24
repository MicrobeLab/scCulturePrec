import numpy as np
import itertools


def create_pairs(npy_fns, half_num_samples, out_prefix, rows_per_file):
	npy_fns = open(npy_fns).read().strip().split('\n')
	per_pos = round(half_num_samples / len(npy_fns))
	combinations = list(itertools.combinations(npy_fns, 2))
	per_neg = round(half_num_samples / len(combinations))

	# create positive pairs 
	positive_pairs = []
	for fn in npy_fns:
		arr = np.load(fn)
		for _ in range(per_pos):
			indices = np.random.choice(len(arr), 2, replace=False)
			pair_arr = arr[indices]
			positive_pairs.append(pair_arr)
	positive_pairs = np.stack(positive_pairs)
	positive_labels = np.ones(len(positive_pairs), dtype=int)

	# create negative pairs
	negative_pairs = []
	for fn_1,fn_2 in combinations:
		arr_1 = np.load(fn_1)
		arr_2 = np.load(fn_2)
		for _ in range(per_neg):
			indices_1 = np.random.choice(len(arr_1), 1, replace=False)[0]
			indices_2 = np.random.choice(len(arr_2), 1, replace=False)[0]
			pair_arr = np.vstack((arr_1[indices_1], arr_2[indices_2]))
			negative_pairs.append(pair_arr)
	negative_pairs = np.stack(negative_pairs)
	negative_labels = np.zeros(len(negative_pairs), dtype=int)

	# combine positive and negative
	both_pairs = np.concatenate((positive_pairs, negative_pairs), axis=0)
	both_labels = np.concatenate((positive_labels, negative_labels))

	# shuffle
	shuffled_indices = np.random.permutation(both_pairs.shape[0])
	both_pairs = both_pairs[shuffled_indices]
	both_labels = both_labels[shuffled_indices]

	# output in blocks
	num_files = int(np.ceil(both_pairs.shape[0] / rows_per_file))

	xls = open('training_paires_' + out_prefix + '.txt', 'w')
	for i in range(num_files):
		start_row = i * rows_per_file
		end_row = min((i+1) * rows_per_file, both_pairs.shape[0])
		x_subset = both_pairs[start_row:end_row, :]
		y_subset = both_labels[start_row:end_row]
		fn_x = f'{out_prefix}_{i+1:03d}_X.npy'
		fn_y = f'{out_prefix}_{i+1:03d}_y.npy'
		np.save(fn_x, x_subset)
		np.save(fn_y, y_subset)
		xls.write(fn_x + '\n')



