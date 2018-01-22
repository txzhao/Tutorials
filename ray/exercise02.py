from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time

ray.init(num_cpus=4, redirect_output=True)

# unused and changed into remote function
def load_data(filename):
	time.sleep(0.1)
	return np.ones((1000, 100))

def normalize_data(data):
	time.sleep(0.1)
	return data - np.mean(data, axis=0)

def extract_features(normalized_data):
	time.sleep(0.1)
	return np.hstack([normalized_data, normalized_data ** 2])

def compute_loss(features):
	num_data, dim = features.shape
	time.sleep(0.1)
	return np.sum((np.dot(features, np.ones(dim)) - np.ones(num_data)) ** 2)

# remote function
@ray.remote
def load_data_remote(filename):
	time.sleep(0.1)
	return np.ones((1000, 100))

@ray.remote
def normalize_data_remote(data):
	time.sleep(0.1)
	return data - np.mean(data, axis=0)

@ray.remote
def extract_features_remote(normalized_data):
	time.sleep(0.1)
	return np.hstack([normalized_data, normalized_data ** 2])

@ray.remote
def compute_loss_remote(features):
	num_data, dim = features.shape
	time.sleep(0.1)
	return np.sum((np.dot(features, np.ones(dim)) - np.ones(num_data)) ** 2)

# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start_time = time.time()

losses = []
for filename in ['file1', 'file2', 'file3', 'file4']:
    data = load_data_remote.remote(filename)
    normalized_data = normalize_data_remote.remote(data)
    features = extract_features_remote.remote(normalized_data)
    loss = compute_loss_remote.remote(features)
    losses.append(loss)

losses = ray.get(losses)
loss = sum(losses)

end_time = time.time()
duration = end_time - start_time

# verification
assert loss == 4000
assert duration < 0.8, ('The loop took {} seconds. This is too slow.'
                        .format(duration))
assert duration > 0.4, ('The loop took {} seconds. This is too fast.'
                        .format(duration))

print('Success! The example took {} seconds.'.format(duration))