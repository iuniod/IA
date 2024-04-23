""" This is the implementation of the Hill Climbing algorithm for the Class Scheduling problem.
    More information about the Hill Climbing algorithm can be found in the README.pdf file. """
from __future__ import annotations

import random
from schedule import State

Result = tuple[bool, int, int, State]

def stochastic_hill_climbing(initial_state: State, max_iters: int = 1000) -> Result:
    ''' Stoachastic Hill Climbing algorithm. '''
    iters, states = 0, 0
    state = initial_state.clone()

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
    initial_state: State,
    max_restarts: int = 100,
    run_max_iters: int = 100,
) -> Result:
    ''' Random Restart Hill Climbing algorithm with stochastic hill climbing at each restart. '''
    total_iters, total_states = 0, 0

    for index in range(max_restarts):
        state = State(initial_state.size, seed=random.random()) if index > 0 \
                                                                else initial_state.clone()
        is_final, iters, states, state = stochastic_hill_climbing(state, run_max_iters)
        total_iters += iters
        total_states += states

        if is_final:
            break

    return is_final, total_iters, total_states, state

if __name__ == '__main__':
    N_TRIALS = 3000
    size = (5, 5) #TODO: Get the size of the schedule from the input file
    wins, fails = 0, 0
    total_iters, total_states, distance = 0, 0, 0
    random.seed(42)

    initials = []
    for _ in range(N_TRIALS):
        initials.append(State(size, seed=random.random()))

    for initial in initials:
        is_final, iters, states, state = random_restart_hill_climbing(initial)

        if is_final:
            wins += 1
            total_iters += iters
            total_states += states
        else:
            fails += 1
            distance += state.conflicts()

    PADDING = ' ' * (30 - len('Random Restart'))
    WIN_PERCENTAGE = (wins / N_TRIALS) * 100.
    print(f"Success rate for {'Random Restart'}: {PADDING}{wins} / {N_TRIALS} ({WIN_PERCENTAGE:.2f}%)")
    print(f"Average number of iterations (for wins): {' ':8}{(total_iters / wins):.2f}")
    print(f"Total number of states (for wins): {' ':>14}{total_states:,}")
    stat = {
        "wins": WIN_PERCENTAGE,
        "iter": total_iters / wins,
        "nums": total_states
    }
    if fails > 0:
        print(f"Average distance to target (for fails): {' ':>9}{(distance / fails):.2f}")
        stat["dist"] = distance / fails
