from __future__ import annotations

import threading
import json
import utils
from heapq import heappush, heappop
from state import State

Result = tuple[bool, int, int, State]

def astar(input_file: str, statistics_file: str, output_file: str) -> Result:
    frontier = []
    initial = State(input_file)
    heappush(frontier, (initial.g() + initial.h(), initial))

    students_per_subject = str(initial.students_per_subject)
    discovered = {students_per_subject: 0}
    final = None

    total_iters, total_states = 0, 0

    while frontier:
        current = heappop(frontier)[1]
        total_iters += 1
        if current.is_final():
            final = current
            break
        next_states = current.get_next_states()
        total_states += len(next_states)
        with open(statistics_file, 'a', encoding='utf-8') as file:
            file.write(f"Current state: {current.students_per_subject}\n")
        for neighbour in next_states:
            new_cost = neighbour.g()
            students = str(neighbour.students_per_subject)
            if students not in discovered or new_cost < discovered[students]:
                discovered[students] = new_cost
                heappush(frontier, (new_cost, neighbour))

    if final is None:
        with open(statistics_file, 'a', encoding='utf-8') as file:
            file.write("No solution found.\n")
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write("No solution found.\n")
        return False, total_iters, total_states, None

    s = utils.pretty_print_timetable(final.schedule, input_file)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(s)

    print(s)

    return final.check_constraints(), total_iters, total_states, final

def run_test(
    name_of_input_file: str, name_of_statistics_file: str, name_of_output_file: str):
    ''' Function that runs the Hill Climbing algorithm in a separate thread. '''
    with open(name_of_statistics_file, 'w', encoding='utf-8') as file:
        file.write("Statistics for the A* algorithm\n\n")
    is_final, iters, states, final = astar(name_of_input_file,
        name_of_statistics_file, name_of_output_file)

    with open(name_of_statistics_file, 'a', encoding='utf-8') as file:
        file.write(f"Final evaluation: {is_final}\n")
        file.write(f"Total iterations: {iters}\n")
        file.write(f"Total states: {states}\n")
        if final is not None:
            file.write(f"Final state constraints: {final.no_constraints()}\n")

if __name__ == '__main__':
    threads = []
    nfiles = [ 'orar_mic_exact',
                'orar_mediu_relaxat',
                'orar_mare_relaxat',
                'orar_constrans_incalcat',
                'orar_bonus_exact']

    for i, nfile in enumerate(nfiles):
        input_file = f'inputs/{nfile}.yaml'
        statistics_file = f'statistics/astar/{nfile}.txt'
        output_file = f'outputs/astar/{nfile}.txt'
        thread = threading.Thread(target=run_test,
            args=(input_file, statistics_file, output_file))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()