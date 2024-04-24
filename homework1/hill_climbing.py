""" This is the implementation of the Hill Climbing algorithm for the Class Scheduling problem.
    More information about the Hill Climbing algorithm can be found in the README.pdf file. """
from __future__ import annotations

import random
import threading
from schedule import State

Result = tuple[bool, int, int, State]
N_TRIALS = 10

def stochastic_hill_climbing(initial_state: State, output_file: str, max_iters: int = 1000) -> Result:
    ''' Stoachastic Hill Climbing algorithm. '''
    iters, states = 0, 0
    state = initial_state.clone()
    intial_conflicts = state.conflicts()

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

    with open(output_file, 'a') as file:
        file.write(f"Number of final conflicts: {state.conflicts()} / {intial_conflicts}\n")
    return state.is_final(), iters, states, state

def random_restart_hill_climbing(
    initial_state: State,
    output_file: str,
    max_restarts: int = 50,
    run_max_iters: int = 10000,
) -> Result:
    ''' Random Restart Hill Climbing algorithm with stochastic hill climbing at each restart. '''
    total_iters, total_states = 0, 0

    for index in range(max_restarts):
        state = State(initial_state.file_name, initial_state.size, seed=random.random()) if index > 0 \
                                                                else initial_state.clone()
        is_final, iters, states, state = stochastic_hill_climbing(state, output_file, run_max_iters)
        total_iters += iters
        total_states += states

        if is_final:
            break

    return is_final, total_iters, total_states, state

def run_test(input_file: str, output_file: str):
    ''' Run the test for the Hill Climbing algorithm. '''
    size = (6, 5) #TODO: Get the size of the schedule from the input file
    wins, fails = 0, 0
    total_iters, total_states, distance = 0, 0, 0
    random.seed(7)

    initials = []
    for _ in range(N_TRIALS):
        initials.append(State(input_file, size, seed=random.random()))

    for initial in initials:
        print(f"Initial state: {initial.file_name}")
        is_final, iters, states, state = random_restart_hill_climbing(initial, output_file)

        if is_final:
            wins += 1
            total_iters += iters
            total_states += states
        else:
            fails += 1
            distance += state.conflicts()

    return wins, fails, total_iters, total_states, distance

def print_statistics(wins: int, fails: int, total_iters: int, total_states: int, distance: int, output_file: str):
    ''' Print the statistics of the Hill Climbing algorithm. '''
    PADDING = ' ' * (30 - len('Random Restart'))
    WIN_PERCENTAGE = (wins / N_TRIALS) * 100.
    with open(output_file, 'a') as file:
        file.write(f"Success rate for {'Random Restart'}: {PADDING}{wins} / {N_TRIALS} ({WIN_PERCENTAGE:.2f}%)\n")
        file.write(f"Average number of iterations (for wins): {' ':8}{(total_iters / wins):.2f}\n")
        file.write(f"Total number of states (for wins): {' ':>14}{total_states:,}\n")
    stat = {
        "wins": WIN_PERCENTAGE,
        "iter": total_iters / wins,
        "nums": total_states
    }
    if fails > 0:
        with open(output_file, 'a') as file:
            file.write(f"Average distance to target (for fails): {' ':>9}{(distance / fails):.2f}\n")
        stat["dist"] = distance / fails

def thread_function(name_of_input_file: str, name_of_output_file: str):
    print_statistics(*run_test(name_of_input_file, name_of_output_file), name_of_output_file)

if __name__ == '__main__':
    threads = []
    name_of_input_files = ['inputs/orar_bonus_exact.yaml',
                           'inputs/orar_mic_exact.yaml',
                           'inputs/orar_mediu_relaxat.yaml',
                           'inputs/orar_mare_relaxat.yaml',
                           'inputs/orar_constrans_incalcat.yaml']
    name_of_output_files = ['outputs/orar_bonus_exact.txt',
                            'outputs/orar_mic_exact.txt',
                            'outputs/orar_mediu_relaxat.txt',
                            'outputs/orar_mare_relaxat.txt',
                            'outputs/orar_constrans_incalcat.txt']

    for i in range(5):
        thread = threading.Thread(target=thread_function, args=(name_of_input_files[i], name_of_output_files[i]))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # For testing purposes
    # FILENAME = 'inputs/orar_mic_exact.yaml'

    # state = State(FILENAME, seed=40)
    # state.display()
    # print(f"Number of conflicts: {state.conflicts()}")

    # new_state = state.change_teacher('Luni', (8, 10), 'PL', 'Andrei Moldovan')
    # new_state.display()
    # print(f"Number of conflicts: {new_state.conflicts()}")

    # new_states = state.get_next_states()
    # print(len(new_states))