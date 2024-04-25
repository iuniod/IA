""" This is the implementation of the Hill Climbing algorithm for the Class Scheduling problem.
    More information about the Hill Climbing algorithm can be found in the README.pdf file. """
from __future__ import annotations

import random
import threading
from schedule import State

Result = tuple[bool, int, int, State]
N_TRIALS = 1

def stochastic_hill_climbing(initial_state: State, statistics_file: str, max_iters: int = 1000) -> Result:
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

    with open(statistics_file, 'a') as file:
        file.write(f"Number of final conflicts: {state.conflicts()} / {intial_conflicts}\n")
    
    return state.is_final(), iters, states, state

def random_restart_hill_climbing(
    initial_state: State,
    statistics_file: str,
    output_file: str,
    max_restarts: int = 100,
    run_max_iters: int = 3000,
) -> Result:
    ''' Random Restart Hill Climbing algorithm with stochastic hill climbing at each restart. '''
    total_iters, total_states = 0, 0
    best_state = None

    for index in range(max_restarts):
        state = State(initial_state.file_name, initial_state.size, seed=random.random()) if index > 0 \
                                                                else initial_state.clone()
        with open(statistics_file, 'a') as file:
            file.write(f"Restarting test {index + 1}...\n")
        is_final, iters, states, state = stochastic_hill_climbing(state, statistics_file, run_max_iters)
        total_iters += iters
        total_states += states

        if best_state is None or state.conflicts() < best_state.conflicts():
            best_state = state

        if is_final:
            break

    best_state.display(output_file)
    return is_final, total_iters, total_states, state

def run_test(input_file: str, statistics_file: str, output_file: str):
    ''' Run the test for the Hill Climbing algorithm. '''
    size = (6, 5) #TODO: Get the size of the schedule from the input file
    wins, fails = 0, 0
    total_iters, total_states, distance = 0, 0, 0
    random.seed(47)

    initials = []
    for _ in range(N_TRIALS):
        initials.append(State(input_file, size, seed=random.random()))

    for initial, step in zip(initials, range(N_TRIALS)):
        with open(statistics_file, 'a') as file:
            file.write(f"Starting test {step + 1}...\n")
        is_final, iters, states, state = random_restart_hill_climbing(initial, statistics_file, output_file)

        if is_final:
            wins += 1
            total_iters += iters
            total_states += states
        else:
            fails += 1
            distance += state.conflicts()

    return wins, fails, total_iters, total_states, distance

def print_statistics(wins: int, fails: int, total_iters: int, total_states: int, distance: int, statistics_file: str):
    ''' Print the statistics of the Hill Climbing algorithm. '''
    PADDING = ' ' * (30 - len('Random Restart'))
    WIN_PERCENTAGE = (wins / N_TRIALS) * 100.
    with open(statistics_file, 'a') as file:
        file.write(f"Success rate for {'Random Restart'}: {PADDING}{wins} / {N_TRIALS} ({WIN_PERCENTAGE:.2f}%)\n")
        file.write(f"Total number of states (for wins): {' ':>14}{total_states:,}\n")
    if wins > 0:
        with open(statistics_file, 'a') as file:
            file.write(f"Average number of iterations (for wins): {' ':8}{(total_iters / wins):.2f}\n")
    # stat = {
    #     "wins": WIN_PERCENTAGE,
    #     "iter": total_iters / wins,
    #     "nums": total_states
    # }
    if fails > 0:
        with open(statistics_file, 'a') as file:
            file.write(f"Average distance to target (for fails): {' ':>9}{(distance / fails):.2f}\n")
        # stat["dist"] = distance / fails

def thread_function(name_of_input_file: str, name_of_statistics_file: str, name_of_output_file: str):
    with open(name_of_statistics_file, 'w') as file:
        file.write(f"Statistics for the Hill Climbing algorithm\n\n")
    print_statistics(*run_test(name_of_input_file, name_of_statistics_file, name_of_output_file), name_of_statistics_file)

if __name__ == '__main__':
    threads = []
    nfiles = [ 'orar_mic_exact.yaml',
                'orar_mediu_relaxat.yaml',
                'orar_mare_relaxat.yaml',
                'orar_constrans_incalcat.yaml',
                'orar_bonus_exact.yaml']

    for i in range(5):
        input_file = f'inputs/{nfiles[i]}'
        statistics_file = f'statistics/{nfiles[i].replace(".yaml", "_statistics.txt")}'
        output_file = f'my_outputs/{nfiles[i].replace(".yaml", ".txt")}'
        thread = threading.Thread(target=thread_function, args=(input_file, statistics_file, output_file))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # # For testing purposes
    # FILENAME = 'inputs/orar_mare_relaxat.yaml'
    # OUTPUT_FILE = 'my_outputs/orar_mare_relaxat.txt'

    # state = State(FILENAME, seed=40)
    # state.display(OUTPUT_FILE)
    # print(f"Number of conflicts: {state.conflicts()}")

    # new_state = state.change_slot('Luni', (8, 10), 'Luni', (10, 12), 'ED020', 'PCom', 'Ioana Scarlatescu')
    # new_state.display()
    # print(f"Number of conflicts: {new_state.conflicts()}")

    # new_state = state.change_teacher('Luni', (8, 10), 'PL', 'Andrei Moldovan')
    # new_state.display()
    # print(f"Number of conflicts: {new_state.conflicts()}")

    # new_states = state.get_next_states()
    # print(len(new_states))