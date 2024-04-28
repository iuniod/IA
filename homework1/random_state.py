""" Module that contains the implementation of a state. """
from __future__ import annotations
import copy
import random
import utils

##################### MACROS #####################
DAYS_OF_THE_WEEK = ['Luni', 'Marti', 'Miercuri', 'Joi', 'Vineri']
TIME_SLOTS = [(8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20)]

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

def generate_hard_schedule(
    yaml_dict: dict[str, dict[str, list[str]]],
    size: (int, int),
    seed: int
    ) -> dict[str, dict[str, dict[str, tuple[str, str]]]] | None:
    ''' Generate a configuration of a schedule that respects all hard constraints. '''
    random.seed(seed)

    classrooms_info = yaml_dict[utils.CLASSROOMS]
    teachers_info = yaml_dict[utils.TEACHERS]
    schedule = generate_empty_schedule(size, classrooms_info)

    students_per_subject = copy.deepcopy(yaml_dict[utils.SUBJECTS])

    iterations = 0
    max_iterations = size[0] * size[1] * len(classrooms_info) * len(teachers_info)
    while len([subject for subject, students in students_per_subject.items() if students > 0]) > 0 \
        and iterations < max_iterations:
        # get a random day, slot and subject that has students
        day = random.choice(DAYS_OF_THE_WEEK[:size[1]])
        slot = random.choice(TIME_SLOTS[:size[0]])
        possible_subjects = [subject for subject, students in students_per_subject.items() \
            if students > 0]
        if not possible_subjects:
            break
        subject = random.choice(possible_subjects)

        # get a random classroom that can hold the subject
        possible_classrooms = [classroom for classroom in schedule[day][slot].keys() \
            if schedule[day][slot][classroom] is None and \
                subject in classrooms_info[classroom][utils.SUBJECTS]]
        if not possible_classrooms:
            iterations += 1
            continue
        classroom = random.choice(possible_classrooms)

        # get a random teacher that can teach the subject, is free in the slot and has less than 7
        possible_teachers = []
        for t in teachers_info.keys():
            if subject in teachers_info[t][utils.SUBJECTS]:
                count = 0
                for c in schedule[day][slot].keys():
                    if schedule[day][slot][c] is not None and schedule[day][slot][c][0] == t:
                        count += 1
                if count == 0:
                    new_count = 0
                    for d in DAYS_OF_THE_WEEK[:size[1]]:
                        for s in TIME_SLOTS[:size[0]]:
                            for c in schedule[d][s].keys():
                                if schedule[d][s][c] is not None and schedule[d][s][c][0] == t:
                                    new_count += 1
                    if new_count < 7:
                        possible_teachers.append(t)
        if not possible_teachers:
            iterations += 1
            continue
        teacher = random.choice(possible_teachers)

        students_per_subject[subject] -= classrooms_info[classroom][utils.CAPACITY]
        schedule[day][slot][classroom] = (teacher, subject)

    if iterations == max_iterations:
        return None
    return schedule

def generate_schedule(
    yaml_dict: dict[str, dict[str, list[str]]],
    size: (int, int),
    seed: int
    ) -> dict[str, dict[str, dict[str, tuple[str, str]]]]:
    ''' Generate a schedule that respects all hard constraints, without stucking. '''
    new_schedule = None
    while new_schedule is None:
        new_schedule = generate_hard_schedule(yaml_dict, size, seed)
        seed += 1
    return new_schedule

class State:
    ''' Class that represents a state of the problem. '''
    def __init__(
        self,
        file_name: str,
        size: (int, int) = (6, 5),
        yaml_dict: dict[str, dict[str, list[str]]] = None,
        schedule: dict[str, dict[str, dict[str, tuple[str, str]]]] | None = None,
        count_teacher_slots: dict[str, int] = None,
        constraints: int | None = None,
        seed: int = 42
    ) -> None:
        self.size = size
        self.file_name = file_name
        self.yaml_dict = utils.read_yaml_file_for_hc(file_name) if yaml_dict is None else yaml_dict
        self.schedule = schedule if schedule is not None \
            else generate_schedule(self.yaml_dict, size, seed)
        self.nconstraints = constraints if constraints is not None \
            else State.__compute_constraints(self, self.size, self.schedule)
        self.count_teacher_slots = count_teacher_slots if count_teacher_slots is not None \
            else {teacher: sum([1 for day in DAYS_OF_THE_WEEK[:size[1]] \
                for slot in TIME_SLOTS[:size[0]] \
                for classroom in self.schedule[day][slot].keys() \
                if self.schedule[day][slot][classroom] is not None and \
                    self.schedule[day][slot][classroom][0] == teacher]) \
                        for teacher in self.yaml_dict[utils.TEACHERS].keys()}
    def change_slot(self, old_day: str, old_slot: str, new_day: str, new_slot: str,
        classroom: str, subject: str, teacher: str
        ) -> State:
        ''' Change the slot of a subject. '''
        new_state = self.clone()

        new_state.schedule[new_day][new_slot][classroom] = (teacher, subject)
        new_state.schedule[old_day][old_slot][classroom] = None
        new_state.nconstraints = \
            State.__compute_constraints(new_state, new_state.size, new_state.schedule)

        return new_state

    def change_teacher(
        self, day: str, slot: str, subject: str, teacher: str, new_teacher: str
        ) -> State:
        ''' Change the teacher in a slot. '''
        new_state = self.clone()
        classroom = None
        for classroom in new_state.schedule[day][slot].keys():
            if new_state.schedule[day][slot][classroom] is not None and \
                new_state.schedule[day][slot][classroom][1] == subject:
                break

        new_state.schedule[day][slot][classroom] = (new_teacher, subject)
        new_state.count_teacher_slots[teacher] -= 1
        new_state.count_teacher_slots[new_teacher] += 1
        new_state.nconstraints = \
            State.__compute_constraints(new_state, new_state.size, new_state.schedule)

        return new_state

    def __compute_constraints_for_break(self, day: str, slot: str, teacher: str) -> int:
        ''' Computes the number of constraints for a teacher in a given break. '''
        teacher_constraints = self.yaml_dict[utils.TEACHERS][teacher]
        count = 0

        # find previous slot in the day in which the teacher has a class
        prev_slot = None
        for s in TIME_SLOTS[:TIME_SLOTS.index(slot)][::-1]:
            if prev_slot is not None:
                break
            for c in self.schedule[day][s].keys():
                if self.schedule[day][s][c] is not None and self.schedule[day][s][c][0] == teacher:
                    prev_slot = s
                    break
        
        # find next slot in the day in which the teacher has a class
        next_slot = None
        for s in TIME_SLOTS[TIME_SLOTS.index(slot) + 1:]:
            if next_slot is not None:
                break
            for c in self.schedule[day][s].keys():
                if self.schedule[day][s][c] is not None and self.schedule[day][s][c][0] == teacher:
                    next_slot = s
                    break

        # check if the teacher has a break more than they want
        if prev_slot is not None:
            if slot[0] - prev_slot[0] > 2 * teacher_constraints[utils.BREAK]:
                count += 1
        
        if next_slot is not None:
            if next_slot[0] - slot[0] > 2 * teacher_constraints[utils.BREAK]:
                count += 1

        return count

    def __compute_constraints_for_timeslot(self, day: str, slot: str, teacher: str) -> int:
        ''' Computes the number of constraints for a teacher in a given timeslot. '''
        _constraints = 0
        teacher_constraints = self.yaml_dict[utils.TEACHERS][teacher]

        # check if the day is in their preferences
        if day not in teacher_constraints[utils.DAYS]:
            _constraints += 1

        # check if the slot is in their preferences
        if slot not in teacher_constraints[utils.SLOTS]:
            _constraints += 1

        # check if the teacher has a break more than they want
        if teacher_constraints[utils.BREAK] is not None:
            _constraints += self.__compute_constraints_for_break(day, slot, teacher)

        return _constraints

    def __compute_constraints(self, size: int, schedule: list[int]) -> int:
        ''' Computes the number of constraints in the given schedule. '''
        _constraints = 0

        for day in DAYS_OF_THE_WEEK[:size[1]]:
            for slot in TIME_SLOTS[:size[0]]:
                for classroom in schedule[day][slot].keys():
                    if schedule[day][slot][classroom] is not None:
                        teacher, _ = schedule[day][slot][classroom]
                        _constraints += self.__compute_constraints_for_timeslot(day, slot, teacher)

        return _constraints

    def no_constraints(self) -> int:
        ''' Returns the number of constraints in the current schedule. '''
        return self.nconstraints

    def check_constraints(self) -> bool:
        ''' Returns True if the current schedule respects all constraints. '''
        return self.nconstraints == 0

    def eval(self) -> int:
        ''' Returns the evaluation of the current schedule. '''
        return self.nconstraints

    def is_final(self) -> bool:
        ''' Returns True if the current schedule is a final one. '''
        return self.nconstraints == 0

    def __possible_teacher_replacements(
        self, day: str, slot: str, classroom: str, replacements: list[State]
        ) -> list[State]:
        ''' Returns a list of teachers that can replace the current teacher. '''
        teacher, subject = self.schedule[day][slot][classroom]

        teachers_info = self.yaml_dict[utils.TEACHERS]
        for new_teacher in teachers_info.keys():
            new_teacher_constraints = teachers_info[new_teacher]
            if subject in new_teacher_constraints[utils.SUBJECTS]:
                # teacher can teach the subject
                count = 0
                for c in self.schedule[day][slot].keys():
                    if self.schedule[day][slot][c] is not None and \
                       self.schedule[day][slot][c][0] == new_teacher:
                        count += 1
                if count == 0:
                    # teacher is not in another classroom at the same time
                    if day in new_teacher_constraints[utils.DAYS] and \
                       slot in new_teacher_constraints[utils.SLOTS]:
                        # teacher accepts the day or slot
                        if self.count_teacher_slots[new_teacher] < 7:
                            # teacher has less than 7 hours per week
                            replacements.append(self.change_teacher(day, slot,
                                subject, teacher, new_teacher))

        return replacements

    def __possible_timeslot_replacements(
        self, day: str, slot: str, classroom: str, replacements: list[State]
        ) -> list[State]:
        ''' Returns a list of possible timeslots for the current subject. '''
        teacher, subject = self.schedule[day][slot][classroom]
        teacher_constraints = self.yaml_dict[utils.TEACHERS][teacher]

        for new_day in DAYS_OF_THE_WEEK[:self.size[1]]:
            for new_slot in TIME_SLOTS[:self.size[0]]:
                if self.schedule[new_day][new_slot][classroom] is None and \
                    (new_day in teacher_constraints[utils.DAYS] and \
                    new_slot in teacher_constraints[utils.SLOTS]):
                    count = 0
                    for c in self.schedule[new_day][new_slot].keys():
                        if self.schedule[new_day][new_slot][c] is not None and \
                            self.schedule[new_day][new_slot][c][0] == teacher:
                            count += 1
                    if count == 0:
                        replacements.append(self.change_slot(day, slot, new_day, new_slot,
                            classroom, subject, teacher))

        return replacements

    def get_next_states(self) -> list[State]:
        ''' Returns a list of all possible states that can be reached from the current state. '''
        new_states = []

        for day in DAYS_OF_THE_WEEK[:self.size[1]]:
            for slot in TIME_SLOTS[:self.size[0]]:
                for classroom in self.schedule[day][slot].keys():
                    if self.schedule[day][slot][classroom] is not None:
                        teacher, _ = self.schedule[day][slot][classroom]
                        no_constraints = self.__compute_constraints_for_timeslot(day, slot, teacher)
                        if no_constraints > 0:
                            new_states = self.__possible_teacher_replacements(day, slot,
                                classroom, new_states)

                            new_states = self.__possible_timeslot_replacements(day, slot,
                                classroom, new_states)

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
            copy.deepcopy(self.schedule), copy.deepcopy(self.count_teacher_slots),
            self.nconstraints)
