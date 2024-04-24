""" Module that contains the implementation of a state. """
from __future__ import annotations
from copy import copy

import random
import utils

DAYS_OF_THE_WEEK = ['Luni', 'Marti', 'Miercuri', 'Joi', 'Vineri']
TIME_SLOTS = [(8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20)]

class State:
    ''' Class that represents a state of the problem. '''
    def __init__(
        self,
        file_name: str,
        # (slots, days)
        size: (int, int) = (6, 5),
        # {Day: {Slot: {Classroom: (Teacher, Subject)}}}
        schedule: dict[str, dict[str, dict[str, tuple[str, str]]]] | None = None,
        conflicts: int | None = None,
        seed: int = 42
    ) -> None:

        self.size = size
        self.file_name = file_name
        self.schedule = schedule if schedule is not None else State.generate_schedule(self, size, seed)
        self.nconflicts = conflicts if conflicts is not None \
            else State.__compute_conflicts(self, self.size, self.schedule)

    def apply_move(self, queen: int, new_row: int) -> State:
        '''
        Construiește o stare vecină în care dama queen este mutată pe linia new_row.
        Numărul de conflicte este calculat prin diferența față de starea originală.
        '''
        return None

    def change_teacher(self, day: str, slot: str, subject: str, new_teacher: str) -> State:
        ''' Change the teacher in a slot. '''
        new_state = copy(self.schedule)
        # TODO: Implement the change of the teacher in a slot
        classroom = None
        for classroom in new_state[day][slot].keys():
            if new_state[day][slot][classroom] is not None and new_state[day][slot][classroom][1] == subject:
                break

        new_state[day][slot][classroom] = (new_teacher, subject)

        return State(self.file_name, self.size, new_state)

    def __generate_empty_schedule(
        self,
        size: (int, int),
        classrooms: list[str]
        ) -> dict[str, dict[str, dict[str, tuple[str, str]]]]:
        ''' Generate an empty schedule. '''
        schedule = {}
        # for the first size[0] slots and size[1] days
        for day in DAYS_OF_THE_WEEK[:size[1]]:
            schedule[day] = {}
            for slot in TIME_SLOTS[:size[0]]:
                schedule[day][slot] = {}
                for classroom in classrooms:
                    schedule[day][slot][classroom] = None

        return schedule

    def __add_classroom(
        self,
        size: (int, int),
        students_per_subject: dict[str, int],
        classrooms_info: dict[str, int],
        teachers_info: dict[str, list[str]],
        classrooms: list[str],
        teachers: list[str]
        ) -> None:
        ''' Add a classroom to the schedule. '''
        # get a random subject that has students
        subject = random.choice([subject for subject, students in students_per_subject.items() if students > 0])

        # get a random day and slot
        day = random.choice(DAYS_OF_THE_WEEK[:size[1]])
        slot = random.choice(TIME_SLOTS[:size[0]])

        # get a random classroom that is free in the given day and slot
        potential_classrooms = [classroom for classroom, info in self.schedule[day][slot].items() if info is None]
        # check if the classroom is for the given subject
        potential_classrooms = [classroom for classroom in potential_classrooms if subject in classrooms_info[classroom][utils.SUBJECTS]]

        if not potential_classrooms:
            return

        classroom = random.choice(potential_classrooms)
        students_per_subject[subject] -= classrooms_info[classroom][utils.CAPACITY]

        # get a random teacher that can teach the subject and is free in the given day and slot
        potential_teachers = [teacher for teacher in teachers if subject in teachers_info[teacher][utils.SUBJECTS]]

        # check if the teacher is not in another classroom at the same time
        potential_teachers = [teacher for teacher in potential_teachers \
            if all(self.schedule[day][slot][classroom] is None or \
                self.schedule[day][slot][classroom][0] != teacher for classroom in classrooms)]

        # check if the number of classes that a teacher has is not exceeded 7
        potential_teachers = [teacher for teacher in potential_teachers \
            if len([info for info in self.schedule[day][slot].values() if info is not None and info[0] == teacher]) < 7]

        if not potential_teachers:
            return

        teacher = random.choice(potential_teachers)

        # add the subject to the schedule
        self.schedule[day][slot][classroom] = (teacher, subject)

    @staticmethod
    def generate_schedule(
        self,
        size: (int, int),
        seed: int
        ) -> dict[str, dict[str, dict[str, tuple[str, str]]]]:
        ''' Generate a configuration of a schedule with the given size.
            Add random subjects and teachers to the schedule. '''
        random.seed(seed)

        yaml_dict = utils.read_yaml_file(self.file_name)
        students_per_subject = yaml_dict[utils.SUBJECTS]
        classrooms_info = yaml_dict[utils.CLASSROOMS]
        teachers_info = yaml_dict[utils.TEACHERS]

        teachers = [teacher for teacher in teachers_info.keys()]
        classrooms = [classroom for classroom in classrooms_info.keys()]

        self.schedule = self.__generate_empty_schedule(size, classrooms)

        while not all (students <= 0 for students in students_per_subject.values()):
            self.__add_classroom(size, students_per_subject, classrooms_info, teachers_info, classrooms, teachers)

        return self.schedule

    def __compute_conflicts_for_timeslot(self, day: str, slot: str, teacher: str, teachers_info: dict[str, list[str]]) -> int:
        ''' Computes the number of conflicts for a teacher in a given timeslot. '''
        _conflicts = 0

        # TODO: Implement the computation of the number of conflicts for a teacher in a given timeslot
        # check if the day is in their preferences
        if day not in teachers_info[teacher][utils.DAYS]:
            _conflicts += 1

        # check if the slot is in their preferences
        if slot not in teachers_info[teacher][utils.SLOTS]:
            _conflicts += 1

        # check if the teacher has a break more than they want
        # get the teacher's next class slot
        next_class = None
        for next_slot in TIME_SLOTS[TIME_SLOTS.index(slot) + 1:]:
            for classroom in self.schedule[day][next_slot].keys():
                if self.schedule[day][next_slot][classroom] is not None and self.schedule[day][next_slot][classroom][0] == teacher:
                    next_class = next_slot
                break

        if next_class is not None and teachers_info[teacher][utils.BREAK] is not None:
            # check if the teacher has a break more than they want
            if 2 * (next_class[0] - slot[0] - 1) > teachers_info[teacher][utils.BREAK]:
                _conflicts += 1
        return _conflicts

    @staticmethod
    def __compute_conflicts(self, size: int, schedule: list[int]) -> int:
        ''' Computes the number of conflicts in the given schedule. '''
        _conflicts = 0
        teachers_info = utils.read_yaml_file(self.file_name)[utils.TEACHERS]

        for day in DAYS_OF_THE_WEEK[:size[1]]:
            for slot in TIME_SLOTS[:size[0]]:
                for classroom in schedule[day][slot].keys():
                    if schedule[day][slot][classroom] is not None:
                        teacher, subject = schedule[day][slot][classroom]
                        _conflicts += self.__compute_conflicts_for_timeslot(day, slot, teacher, teachers_info)

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

    def display(self) -> None:
        ''' Print the current schedule.'''
        print(utils.pretty_print_timetable(self.schedule, self.file_name))

    def clone(self) -> State:
        ''' Returns a copy of the current schedule.	'''
        return State(self.size, copy(self.schedule), self.nconflicts)
