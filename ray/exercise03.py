from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time

ray.init(num_cpus=8, redirect_output=True)

# unused and changed into remote function
# This is a proxy for a function which generates some data.
def create_data(i):
	time.sleep(0.3)
	return i * np.ones(10000)

# This is a proxy for an expensive aggregation step (which is also
# commutative and associative so it can be used in a tree-reduce).
def aggregate_data(x, y):
	time.sleep(0.3)
	return x * y

# This is a proxy for a function which generates some data.
@ray.remote
def create_data_remote(i):
	time.sleep(0.3)
	return i * np.ones(10000)

# This is a proxy for an expensive aggregation step (which is also
# commutative and associative so it can be used in a tree-reduce).
@ray.remote
def aggregate_data_remote(x, y):
	time.sleep(0.3)
	return x * y

# Sleep a little to improve the accuracy of the timing measurements below.
time.sleep(2.0)
start_time = time.time()

# EXERCISE: Here we generate some data. Do this part in parallel.
vectors = [create_data_remote.remote(i + 1) for i in range(8)]

# Here we aggregate all of the data repeatedly calling aggregate_data. This
# can be sped up using Ray.
#
# NOTE: A direct translation of the code below to use Ray will not result in
# a speedup because each function call uses the output of the previous function
# call so the function calls must be executed serially.
#
# EXERCISE: Speed up the aggregation below by using Ray. Note that this will
# require restructuring the code to expose more parallelism. First run 4 tasks
# aggregating the 8 values in pairs. Then run 2 tasks aggregating the resulting
# 4 intermediate values in pairs. then run 1 task aggregating the two resulting
# values. Lastly, you will need to call ray.get to retrieve the final result.
result = aggregate_data(vectors[0], vectors[1])
result = aggregate_data(result, vectors[2])
result = aggregate_data(result, vectors[3])
result = aggregate_data(result, vectors[4])
result = aggregate_data(result, vectors[5])
result = aggregate_data(result, vectors[6])
result = aggregate_data(result, vectors[7])

# NOTE: For clarity, the aggregation above is written out as 7 separate function
# calls, but this can be done more easily in a while loop via
#
#     while len(vectors) > 1:
#         vectors = aggregate_data(vectors[0], vectors[1]) + vectors[2:]
#     result = vectors[0]
#
# When expressed this way, the change from serial aggregation to tree-structured
# aggregation can be made simply by appending the result of aggregate_data to the
# end of the vectors list as opposed to the beginning.
#
# EXERCISE: Think about why this is true.

end_time = time.time()
duration = end_time - start_time
