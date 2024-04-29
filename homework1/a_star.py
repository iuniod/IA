from __future__ import annotations

import threading
import json
import utils
from heapq import heappush, heappop
from state import State

def astar(size: (int, int), input_file: str, statistics_file: str, output_file: str) -> None:
    frontier = []
    initial = State(input_file, size)
    heappush(frontier, (initial.g() + initial.h(), initial))

    students_per_subject = str(initial.students_per_subject)
    discovered = {students_per_subject: (None, 0)}
    final = None

    while frontier:
        current = heappop(frontier)[1]
        if current.is_final():
            final = current
            break
        next_states = current.get_next_states_astar()
        with open(statistics_file, 'a', encoding='utf-8') as file:
            file.write(f"Current state: {current.students_per_subject}\n")
            # file.write(f"Next states: {[neighbour.students_per_subject for neighbour in next_states]}\n")
        for neighbour in next_states:
            new_cost = neighbour.g()
            students = str(neighbour.students_per_subject)
            if students not in discovered or new_cost < discovered[students][1]:
                discovered[students] = (current, new_cost)
                heappush(frontier, (new_cost + neighbour.h(), neighbour))

    if final is None:
        with open(statistics_file, 'a', encoding='utf-8') as file:
            file.write("No solution found.\n")
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write("No solution found.\n")
        return

    s = utils.pretty_print_timetable(final.schedule, input_file)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(s)

def thread_function(
    name_of_input_file: str, name_of_statistics_file: str, name_of_output_file: str):
    ''' Function that runs the Hill Climbing algorithm in a separate thread. '''
    with open(name_of_statistics_file, 'w', encoding='utf-8') as file:
        file.write("Statistics for the A* algorithm\n\n")
    size = (6, 5) #TODO: Get the size of the schedule from the input file
    astar(size, name_of_input_file, name_of_statistics_file, name_of_output_file)

if __name__ == '__main__':
    threads = []
    nfiles = [ 'orar_mic_exact',
                'orar_mediu_relaxat',
                'orar_mare_relaxat',
                'orar_constrans_incalcat',
                'orar_bonus_exact']

    for nfile in nfiles:
        input_file = f'inputs/{nfiles}.yaml'
        statistics_file = f'statistics/astar/{nfiles}.txt'
        output_file = f'outputs/astar/{nfiles}.txt'
        thread = threading.Thread(target=thread_function,
            args=(input_file, statistics_file, output_file))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()