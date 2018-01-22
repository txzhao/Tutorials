from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import time

ray.init(num_cpus=4, redirect_output=True)

# This function is a proxy for a more interesting and computationally
# intensive function.
def slow_function(i):
	time.sleep(1)
	return i

@ray.remote
def remote_function(i):
	time.sleep(1)
	return i

# Sleep a little to improve the accuracy of the timing measurements below.
# We do this because workers may still be starting up in the background.
time.sleep(2.0)
start_time = time.time()

results = []

# # unused - slow
# for i in range(4):
# 	results.append(slow_function(i))

for i in range(4):
	results.append(remote_function.remote(i))

results = ray.get(results)

end_time = time.time()
duration = end_time - start_time

# verification checks
assert results == [0, 1, 2, 3], 'Did you remember to call ray.get?'
assert duration < 1.1, ('The loop took {} seconds. This is too slow.'
                        .format(duration))
assert duration > 1, ('The loop took {} seconds. This is too fast.'
                      .format(duration))

print('Success! The example took {} seconds.'.format(duration))
