""" This is the implementation of the Hill Climbing algorithm for the Class Scheduling problem.
	More information about the Hill Climbing algorithm can be found in the README.pdf file. """
from __future__ import annotations
from typing import Callable
from copy import copy

import numpy as np
import random
import os
from schedule import State

def stochastic_hill_climbing(initial: State, max_iters: int = 1000) -> Result:
	iters, states = 0, 0
	state = initial.clone()
	
	while iters < max_iters:
		iters += 1

		if state.is_final():
			break

		next_states = list(state.get_next_states())
		best_states = list(filter(lambda s: s.conflicts() < state.conflicts(), next_states))
		states += len(next_states)

		if len(best_states) == 0:
			break
			
		best_state = random.choice(best_states)
		state = best_state

	return state.is_final(), iters, states, state

def random_restart_hill_climbing(
	initial: State,
	max_restarts: int = 100, 
	run_max_iters: int = 100, 
) -> Result:
	
	total_iters, total_states = 0, 0

	for index in range(max_restarts):
		state = index > 0 and State(initial.size, seed=random.random()) or initial.clone()
		is_final, iters, states, state = stochastic_hill_climbing(state, run_max_iters)
		total_iters += iters
		total_states += states

		if is_final:
			break
	
	return is_final, total_iters, total_states, state

if __name__ == '__main__':
	n_trials = 3000
	size = 8
	kwargs = {}
	wins, fails = 0, 0
	total_iters, total_states, distance = 0, 0, 0
	random.seed(42)
	
	initials = []
	for _ in range(n_trials):
		initials.append(State(size, seed=random.random()))
	
	for initial in initials:
		is_final, iters, states, state = random_restart_hill_climbing(initial, **kwargs)
		
		if is_final: 
			wins += 1
			total_iters += iters
			total_states += states
		else:
			fails += 1
			distance += state.conflicts()
	
	padding = ' ' * (30 - len('Random Restart'))
	win_percentage = (wins / n_trials) * 100.
	print(f"Success rate for {'Random Restart'}: {padding}{wins} / {n_trials} ({win_percentage:.2f}%)")
	print(f"Average number of iterations (for wins): {' ':8}{(total_iters / wins):.2f}")
	print(f"Total number of states (for wins): {' ':>14}{total_states:,}")
	stat = {
		"wins": win_percentage,
		"iter": total_iters / wins,
		"nums": total_states
	}
	if fails > 0:
		print(f"Average distance to target (for fails): {' ':>9}{(distance / fails):.2f}")
		stat["dist"] = distance / fails