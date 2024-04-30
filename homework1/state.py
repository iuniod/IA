""" Module that contains the implementation of a state. """
from __future__ import annotations
import copy
import utils
import json
from collections import deque

##################### MACROS #####################
DAYS_OF_THE_WEEK = ['Luni', 'Marti', 'Miercuri', 'Joi', 'Vineri']
TIME_SLOTS = [(8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20)]
MAX_SLOTS = 7
#################################################

class State:
    ''' Class that represents a state of the problem. '''
    def __init__(
        self,
        file_name: str,
        size: (int, int) = None,
        yaml_dict: dict[str, dict[str, dict[str, str]]] = None,
        schedule: dict[str, dict[str, dict[str, tuple[str, str]]]] | None = None,
        count_techer_slots: dict[str, int] = None,
        students_per_subject: dict[str, int] = None,
        trade_off: int = None
    ) -> None:
        self.file_name = file_name
        self.yaml_dict = utils.read_yaml_file_for_hc(file_name) if yaml_dict is None else yaml_dict
        self.size = size if size is not None else (len(self.yaml_dict[utils.INTERVALS]), len(self.yaml_dict[utils.DAYS]))
        self.schedule = schedule if schedule is not None \
            else self.generate_empty_schedule(self.size, self.yaml_dict[utils.CLASSROOMS])
        self.students_per_subject = students_per_subject if students_per_subject is not None \
            else copy.deepcopy(self.yaml_dict[utils.SUBJECTS])
        self.count_techer_slots = count_techer_slots if count_techer_slots is not None \
            else {teacher: 0 for teacher in self.yaml_dict[utils.TEACHERS]}
        self.trade_off = trade_off if trade_off is not None else 0

    @staticmethod
    def generate_empty_schedule(
        size: (int, int),
        classrooms: list[str]
        ) -> dict[str, dict[str, dict[str, tuple[str, str]]]]:
        ''' Generate an empty schedule. '''
        schedule = {}

        for day in DAYS_OF_THE_WEEK[:size[1]]:
            schedule[day] = {}
            for slot in TIME_SLOTS[:size[0]]:
                schedule[day][slot] = {}
                for classroom in classrooms:
                    schedule[day][slot][classroom] = None

        return schedule

    def eval(self) -> int:
        ''' Returns the evaluation of the current schedule. '''
        return sum(self.students_per_subject.values())

    def is_final(self) -> bool:
        ''' Returns True if the current schedule is a final one. '''
        return self.eval() == 0

    def no_constraints(self) -> int:
        ''' Returns the number of constraints that are not satisfied. '''
        constraints = 0
        for day in DAYS_OF_THE_WEEK[:self.size[1]]:
            for slot in TIME_SLOTS[:self.size[0]]:
                for classroom in self.yaml_dict[utils.CLASSROOMS]:
                    if self.schedule[day][slot][classroom] is not None:
                        teacher, subject = self.schedule[day][slot][classroom]

                        if day not in self.yaml_dict[utils.TEACHERS][teacher][utils.DAYS]:
                            constraints += 1

                        if slot not in self.yaml_dict[utils.TEACHERS][teacher][utils.SLOTS]:
                            constraints += 1

                        if self.yaml_dict[utils.TEACHERS][teacher][utils.BREAK] is not None:
                            # find previous slot in the day in which the teacher has a class
                            prev_slot = None
                            for s in TIME_SLOTS[:TIME_SLOTS.index(slot)][::-1]:
                                if prev_slot is not None:
                                    break
                                for c in self.schedule[day][s].keys():
                                    if self.schedule[day][s][c] is not None and \
                                        self.schedule[day][s][c][0] == teacher:
                                        prev_slot = s
                                        break

                            # find next slot in the day in which the teacher has a class
                            next_slot = None
                            for s in TIME_SLOTS[TIME_SLOTS.index(slot) + 1:]:
                                if next_slot is not None:
                                    break
                                for c in self.schedule[day][s].keys():
                                    if self.schedule[day][s][c] is not None and \
                                        self.schedule[day][s][c][0] == teacher:
                                        next_slot = s
                                        break

                            # check if the teacher has a break more than they want
                            teacher_constraints = self.yaml_dict[utils.TEACHERS][teacher]
                            if prev_slot is not None:
                                if slot[0] - prev_slot[0] > 2 * teacher_constraints[utils.BREAK]:
                                    constraints += 1
                            
                            if next_slot is not None:
                                if next_slot[0] - slot[0] > 2 * teacher_constraints[utils.BREAK]:
                                    constraints += 1

        return constraints

    def check_constraints(self) -> bool:
        ''' Returns True if all the constraints are satisfied. '''
        return self.no_constraints() == 0

    def h(self):
        ''' Returns the heuristic value for the current state. '''
        return self.eval()

    def g(self):
        ''' Returns the cost to reach the current state. '''
        return 100 * self.no_constraints() + self.trade_off

    def get_schedule(self) -> dict[str, dict[str, dict[str, tuple[str, str]]]]:
        ''' Returns the current schedule. '''
        return self.schedule

    def _check_teacher_constraints(
        self, day: str, slot: (int, int),
        classroom: str, subject:str, teacher: str,
        soft_constraints: bool
        ) -> bool:
        ''' Check if the teacher can teach in the given day, slot and classroom. '''
        if subject not in self.yaml_dict[utils.TEACHERS][teacher][utils.SUBJECTS]:
            return False

        if self.count_techer_slots[teacher] >= MAX_SLOTS:
            return False

        if soft_constraints:
            if day not in self.yaml_dict[utils.TEACHERS][teacher][utils.DAYS]:
                return False

            if slot not in self.yaml_dict[utils.TEACHERS][teacher][utils.SLOTS]:
                return False

            if self.yaml_dict[utils.TEACHERS][teacher][utils.BREAK] is not None:
                # find previous slot in the day in which the teacher has a class
                prev_slot = None
                for s in TIME_SLOTS[:TIME_SLOTS.index(slot)][::-1]:
                    if prev_slot is not None:
                        break
                    for c in self.schedule[day][s].keys():
                        if self.schedule[day][s][c] is not None and \
                            self.schedule[day][s][c][0] == teacher:
                            prev_slot = s
                            break
                
                # find next slot in the day in which the teacher has a class
                next_slot = None
                for s in TIME_SLOTS[TIME_SLOTS.index(slot) + 1:]:
                    if next_slot is not None:
                        break
                    for c in self.schedule[day][s].keys():
                        if self.schedule[day][s][c] is not None and \
                            self.schedule[day][s][c][0] == teacher:
                            next_slot = s
                            break

                # check if the teacher has a break more than they want
                teacher_constraints = self.yaml_dict[utils.TEACHERS][teacher]
                if prev_slot is not None:
                    if slot[0] - prev_slot[0] > 2 * teacher_constraints[utils.BREAK]:
                        return False
                
                if next_slot is not None:
                    if next_slot[0] - slot[0] > 2 * teacher_constraints[utils.BREAK]:
                        return False
        return True

    def _get_neighbours(self, soft_constraints: bool) -> list[State]:
        new_states = []
        for day in DAYS_OF_THE_WEEK[:self.size[1]]:
            for slot in TIME_SLOTS[:self.size[0]]:
                for subject, students in self.students_per_subject.items():
                    if students == 0:
                        continue

                    for classroom in self.yaml_dict[utils.CLASSROOMS]:
                        if self.schedule[day][slot][classroom] is not None:
                            continue

                        if subject not in \
                            self.yaml_dict[utils.CLASSROOMS][classroom][utils.SUBJECTS]:
                            continue

                        for teacher in self.yaml_dict[utils.TEACHERS]:
                            if not self._check_teacher_constraints(day, slot, classroom, subject,
                                teacher, soft_constraints):
                                continue

                            current_teachers = []
                            for c in self.schedule[day][slot].values():
                                if c is not None:
                                    current_teachers.append(c[0])
                            if teacher in current_teachers:
                                continue

                            new_state = self.clone()
                            # number of students unassigned to the subject
                            no_unassigned = new_state.students_per_subject[subject]
                            # number of empty seats in all the classrooms that can host the subject
                            no_empty_capacity = 0
                            for d in DAYS_OF_THE_WEEK[:self.size[1]]:
                                for s in TIME_SLOTS[:self.size[0]]:
                                    for c in self.yaml_dict[utils.CLASSROOMS]:
                                        if subject in self.yaml_dict[utils.CLASSROOMS][c][utils.SUBJECTS] and \
                                            self.schedule[d][s][c] is None:
                                            no_empty_capacity += self.yaml_dict[utils.CLASSROOMS][c][utils.CAPACITY]
                            no_classrooms_for_subject = len([c for c in self.yaml_dict[utils.CLASSROOMS] \
                                if subject in self.yaml_dict[utils.CLASSROOMS][c][utils.SUBJECTS]])
                            no_classrooms = len(self.yaml_dict[utils.CLASSROOMS])
                            new_state.trade_off = 1 - (no_unassigned / no_empty_capacity) + \
                                (no_classrooms_for_subject / no_classrooms)
                            new_state.schedule[day][slot][classroom] = (teacher, subject)
                            new_state.students_per_subject[subject] -= \
                                self.yaml_dict[utils.CLASSROOMS][classroom][utils.CAPACITY]
                            new_state.students_per_subject[subject] = \
                                max(0, new_state.students_per_subject[subject])
                            new_state.count_techer_slots[teacher] += 1

                            new_states.append(new_state)
        return new_states

    def get_next_states(self) -> deque[State]:
        ''' Returns a deque of all possible states that can be reached from the current state. '''
        new_states = self._get_neighbours(True)
        if len(new_states) == 0:
            new_states = self._get_neighbours(False)

        return new_states

    def display(self, output_file: str = None) -> None:
        ''' Print the current schedule.'''
        if output_file is not None:
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(utils.pretty_print_timetable(self.schedule,self.file_name))
        else :
            print(utils.pretty_print_timetable(self.schedule, self.file_name))

    def clone(self) -> State:
        ''' Returns a copy of the current schedule.	'''
        return State(self.file_name, self.size, copy.deepcopy(self.yaml_dict),
            copy.deepcopy(self.schedule), copy.deepcopy(self.count_techer_slots),
            copy.deepcopy(self.students_per_subject))

    def __lt__(self, other: State) -> bool:
        return self.g() + self.h() < other.g() + other.h()
