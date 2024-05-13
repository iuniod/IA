""" This is the implementation of the Hill Climbing algorithm for the Class Scheduling problem.
    More information about the Hill Climbing algorithm can be found in the README.pdf file. """
from __future__ import annotations

import random
import threading
import sys
from state import State

Result = tuple[bool, int, int, State]
N_TRIALS = 1
HELP_MESSAGE = '\n Se specifica tipul de stare initiala\nExemplu: python3 hill_climbing.py random\n\
    Exemplu: python3 hill_climbing.py empty\n'

def stochastic_hill_climbing(
    initial_state: State, statistics_file: str, max_iters: int = 1000
    ) -> Result:
    ''' Stoachastic Hill Climbing algorithm. '''
    iters, states = 0, 0
    state = initial_state.clone()
    intial_eval = state.eval()

    while iters < max_iters:
        iters += 1

        if state.is_final():
            break

        next_states = state.get_next_states()
        states += len(next_states)

        if len(next_states) == 0:
            break

        best_state = random.choice(next_states)
        state = best_state

    with open(statistics_file, 'a') as file:
        file.write(f"Final evaluation: {state.eval()} / {intial_eval}\n")

    return state.is_final(), iters, states, state

def random_restart_hill_climbing(
    initial_state: State,
    statistics_file: str,
    output_file: str,
    seed: int,
    max_restarts: int = 20,
    run_max_iters: int = 1000
) -> Result:
    ''' Random Restart Hill Climbing algorithm with stochastic hill climbing at each restart. '''
    total_iters, total_states = 0, 0
    best_state = None

    # TODO use seed
    random.seed(seed)

    for index in range(max_restarts):
        state = State(initial_state.file_name) if index > 0 else initial_state.clone()
        with open(statistics_file, 'a', encoding='utf-8') as file:
            file.write(f"Restarting test {index + 1}...\n")
        is_final, iters, states, state = \
            stochastic_hill_climbing(state, statistics_file, run_max_iters)
        total_iters += iters
        total_states += states

        with open(statistics_file, 'a', encoding='utf-8') as file:
            if best_state is not None:
                file.write(f"Final state vs best state evaluation: {state.eval()} / {best_state.eval()}\n")
                file.write(f"Final state vs best state constraints: {state.no_constraints()} / {best_state.no_constraints()}\n")

        if is_final:
            if best_state is None or (state.eval() == 0 and \
                state.no_constraints() < best_state.no_constraints()):
                best_state = state
            if state.check_constraints():
                break

    best_state.display(output_file)
    best_state.display()
    return is_final, total_iters, total_states, best_state

def run_test(
    name_of_input_file: str, name_of_statistics_file: str, name_of_output_file: str):
    ''' Function that runs the Hill Climbing algorithm in a separate thread. '''
    with open(name_of_statistics_file, 'w', encoding='utf-8') as file:
        file.write("Statistics for the Hill Climbing algorithm\n\n")
    is_final, iters, states, state = random_restart_hill_climbing(
        State(name_of_input_file), name_of_statistics_file, name_of_output_file, 47)

    with open(name_of_statistics_file, 'a', encoding='utf-8') as file:
        file.write(f"Final evaluation: {is_final}\n")
        file.write(f"Total iterations: {iters}\n")
        file.write(f"Total states: {states}\n")
        if state is not None:
            file.write(f"Final state constraints: {state.no_constraints()}\n")

if __name__ == '__main__':
    threads = []
    nfiles = [ 'orar_mic_exact',
                'orar_mediu_relaxat',
                'orar_mare_relaxat',
                'orar_constrans_incalcat',
                'orar_bonus_exact']

    for nfile in nfiles:
        input_file = f'inputs/{nfile}.yaml'
        statistics_file = f'statistics/hc/{nfile}.txt'
        output_file = f'outputs/hc/{nfile}.txt'
        thread = threading.Thread(target=run_test,
            args=(input_file, statistics_file, output_file))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
