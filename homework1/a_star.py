from __future__ import annotations

import random
import threading
import json
from heapq import heappush, heappop
from state import State

def astar(size: (int, int), input_file: str, statistics_file: str, output_file: str) -> None:
    frontier = []
    initial = State(input_file, size)
    heappush(frontier, (initial.g() + initial.h(), initial))

    schedule = initial.serialize_schedule()
    discovered = {schedule: (None, 0)}
    final = None

    while frontier:
        current = heappop(frontier)[1]
        if current.is_final():
            final = current.get_schedule()
            break
        next_states = current.get_next_states()
        for neighbour in next_states:
            new_cost = neighbour.g()
            new_schedule = neighbour.serialize_schedule()
            if new_schedule not in discovered or new_cost < discovered[new_schedule][1]:
                discovered[new_schedule] = (current, new_cost)
                heappush(frontier, (new_cost + neighbour.h(), neighbour))

    if final is None:
        with open(statistics_file, 'a', encoding='utf-8') as file:
            file.write("No solution found.\n")
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write("No solution found.\n")
        return

    final.display(output_file)
    with open(statistics_file, 'a', encoding='utf-8') as file:
        file.write(f"Founds solution with cost {final.g()}.\n")

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

    for i in range(0, 1):
        input_file = f'inputs/{nfiles[i]}.yaml'
        statistics_file = f'statistics/astar/{nfiles[i]}.txt'
        output_file = f'outputs/astar/{nfiles[i]}.txt'
        thread = threading.Thread(target=thread_function,
            args=(input_file, statistics_file, output_file))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()