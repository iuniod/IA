''' Generates images for the statistics page. '''
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
	# create the directories if they do not exist
	if not os.path.exists('images'):
		os.mkdir('images')

	# create the datasets

	# number of iterations
	y_astar_iter = [146, 158, 2883, 91, 2542]
	y_hc_iter = [106, 71, 76, 5889, 2221]

	# number of states
	y_astar_states = [17073, 183159, 2171276, 12368, 1385784]
	y_hc_states = [9989, 53875, 72621, 1228027, 1603927]

	# times
	y_hc_times = [0.4, 0.34, 0.77, 10.40, 22.47]
	y_astar_times = [0.8, 2.12, 52.49, 0.8, 27.15]

	# restarts
	y_hc_restarts = [3, 1, 1, 100, 20]

	# constraints
	y_astar_constraints = [0, 0, 0, 0, 0]
	y_hc_constraints = [0, 0, 0, 6, 5]

	# create the plots

	# number of iterations with Bar Graph
	plt.figure()
	x = np.arange(len(y_astar_iter))
	plt.bar(x, y_astar_iter, width=0.4, label='A*')
	plt.bar(x + 0.4, y_hc_iter, width=0.4, label='Hill Climbing')
	plt.xlabel('Test number')
	plt.ylabel('Number of iterations')
	plt.title('Number of iterations')
	plt.legend()
	plt.savefig('images/iterations.png')

	# number of states Bar Graph
	plt.figure()
	x = np.arange(len(y_astar_states))
	plt.bar(x, y_astar_states, width=0.4, label='A*')
	plt.bar(x + 0.4, y_hc_states, width=0.4, label='Hill Climbing')
	plt.xlabel('Test number')
	plt.ylabel('Number of states')
	plt.title('Number of states')
	plt.legend()
	plt.savefig('images/states.png')

	# times Bar Graph
	plt.figure()
	x = np.arange(len(y_astar_times))
	plt.bar(x, y_astar_times, width=0.4, label='A*')
	plt.bar(x + 0.4, y_hc_times, width=0.4, label='Hill Climbing')
	plt.xlabel('Test number')
	plt.ylabel('Time (min)')
	plt.title('Time')
	plt.legend()
	plt.savefig('images/times.png')

	# restarts Bar Graph with values on bars
	plt.figure()
	x = np.arange(len(y_hc_restarts))
	plt.bar(x, y_hc_restarts, width=0.4, label='Hill Climbing')
	plt.xlabel('Test number')
	plt.ylabel('Number of restarts')
	plt.title('Number of restarts')
	plt.legend()
	for i, v in enumerate(y_hc_restarts):
		plt.text(i - 0.1, v + 2, str(v))
	plt.savefig('images/restarts.png')

	# constraints Bar Graph with values on bars
	plt.figure()
	x = np.arange(len(y_astar_constraints))
	plt.bar(x, y_astar_constraints, width=0.4, label='A*')
	plt.bar(x + 0.4, y_hc_constraints, width=0.4, label='Hill Climbing')
	plt.xlabel('Test number')
	plt.ylabel('Number of constraints')
	plt.title('Number of constraints')
	plt.legend()
	for i, v in enumerate(y_astar_constraints):
		plt.text(i - 0.1, v + 0.2, str(v))
	for i, v in enumerate(y_hc_constraints):
		plt.text(i + 0.3, v + 0.2, str(v))
	plt.savefig('images/constraints.png')