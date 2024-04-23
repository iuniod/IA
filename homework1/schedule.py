""" Module that contains the implementation of a state. """
from __future__ import annotations
from copy import copy

import random


class State:
    ''' Class that represents a state of the problem. '''
    def __init__(
        self,
        size: (int, int) = (6, 5),
        schedule: list[list[list[(str, str, str)]]] = None,# [(subject, room, teacher) - all rooms]
        conflicts: int | None = None,
        seed: int = 42
    ) -> None:

        self.size = size
        self.schedule = schedule if schedule is not None else State.generate_schedule(size, seed)
        self.nconflicts = conflicts if conflicts is not None \
            else State.__compute_conflicts(self.size, self.schedule)

    def apply_move(self, queen: int, new_row: int) -> State:
        '''
        Construiește o stare vecină în care dama queen este mutată pe linia new_row.
        Numărul de conflicte este calculat prin diferența față de starea originală.
        '''
        return None

    def change_teacher(self, slot: int, new_teacher: str) -> State:
        ''' Change the teacher in a slot. '''
        new_state = copy(self.schedule)
        # TODO: Implement the change of the teacher in a slot
        return State(self.size, new_state)

    @staticmethod
    def generate_schedule(size: int, seed: int) -> list[int]:
        ''' Generate a configuration of a schedule with the given size.
            Add random subjects and teachers to the schedule. '''
        random.seed(seed)
        schedule = []

        # TODO: Implement the generation of a schedule

        return schedule

    @staticmethod
    def __compute_conflicts(size: int, schedule: list[int]) -> int:
        ''' Computes the number of conflicts in the given schedule. '''
        _conflicts = 0

        # TODO: Implement the computation of the number of conflicts

        return _conflicts

    def conflicts(self) -> int:
        ''' Returns the number of conflicts in the current schedule. '''
        return self.nconflicts

    def is_final(self) -> bool:
        ''' Returns True if the current schedule is a final one. '''
        return self.nconflicts == 0

    def get_next_states(self) -> list[State]:
        ''' Returns a list of all possible states that can be reached from the current state. '''
        # return (self.apply_move(col, row) for col in range(self.size)
        # 		for row in range(self.size) if row != self.board[col])
        # TODO: Implement the generation of the next possible states
        return []

    def __str__(self) -> str:
        # TODO: use the pretty print of the schedule from utils
        return ''

    def display(self) -> None:
        ''' Print the current schedule.'''
        print(self)

    def clone(self) -> State:
        ''' Returns a copy of the current schedule.	'''
        return State(self.size, copy(self.schedule), self.nconflicts)
