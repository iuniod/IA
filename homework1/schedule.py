""" Module that contains the implementation of a state. """
from __future__ import annotations
import copy
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
        self.yaml_dict = utils.read_yaml_file_for_hc(file_name)
        self.file_name = file_name
        self.schedule = schedule if schedule is not None else State.generate_schedule(self.yaml_dict, size, seed)
        self.nconflicts = conflicts if conflicts is not None \
            else State.__compute_conflicts(self, self.size, self.schedule)

    def change_slot(self, old_day: str, old_slot: str, new_day: str, new_slot: str, classroom: str, subject: str, teacher: str) -> State:
        ''' Change the slot of a subject. '''
        new_state = copy.deepcopy(self.schedule)
        new_state[new_day][new_slot][classroom] = (teacher, subject)
        new_state[old_day][old_slot][classroom] = None

        return State(self.file_name, self.size, new_state)

    def change_teacher(self, day: str, slot: str, subject: str, new_teacher: str) -> State:
        ''' Change the teacher in a slot. '''
        new_state = copy.deepcopy(self.schedule)

        classroom = None
        for classroom in new_state[day][slot].keys():
            if new_state[day][slot][classroom] is not None and new_state[day][slot][classroom][1] == subject:
                break

        new_state[day][slot][classroom] = (new_teacher, subject)

        return State(self.file_name, self.size, new_state)

    def split_classroom(
        self, day: str, slot: str, classroom: str,
        new_day1: str, new_slot1: str, new_classroom1: str,
        new_day2: str, new_slot2: str, new_classroom2: str
        ) -> State:
        ''' Split a classroom into two. '''
        new_state = copy.deepcopy(self.schedule)

        teacher, subject = new_state[day][slot][classroom]
        new_state[new_day1][new_slot1][new_classroom1] = (teacher, subject)
        new_state[new_day2][new_slot2][new_classroom2] = (teacher, subject)
        new_state[day][slot][classroom] = None

        return State(self.file_name, self.size, new_state)

    @staticmethod
    def __generate_empty_schedule(
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

    @staticmethod
    def generate_schedule(
        yaml_dict: dict[str, dict[str, list[str]]],
        size: (int, int),
        seed: int
        ) -> dict[str, dict[str, dict[str, tuple[str, str]]]]:
        ''' Generate a configuration of a schedule with the given size.
            Add random subjects and teachers to the schedule. '''
        random.seed(seed)

        schedule = State.__generate_empty_schedule(size, yaml_dict[utils.CLASSROOMS])

        students_per_subject = yaml_dict[utils.SUBJECTS]

        # as long as the number of students for each subject is not 0
        iterations = 0
        while any(students_per_subject.values()):
            # get a random day, slot and subject that has students
            day = random.choice(DAYS_OF_THE_WEEK[:size[1]])
            slot = random.choice(TIME_SLOTS[:size[0]])
            possible_subjects = [subject for subject, students in students_per_subject.items() if students > 0]
            if not possible_subjects:
                break
            subject = random.choice(possible_subjects)

            # get a random classroom that can hold the subject
            possible_classrooms = [classroom for classroom in schedule[day][slot].keys() if schedule[day][slot][classroom] is None and subject in yaml_dict[utils.CLASSROOMS][classroom][utils.SUBJECTS]]
            if not possible_classrooms:
                iterations += 1
                continue
            classroom = random.choice(possible_classrooms)

            # get a random teacher that can teach the subject
            possible_teachers = []
            for t in yaml_dict[utils.TEACHERS].keys():
                # check if he can teach the subject
                if subject in yaml_dict[utils.TEACHERS][t][utils.SUBJECTS]:
                    # check if he is not already teaching in the same slot
                    count = 0
                    for c in schedule[day][slot].keys():
                        if schedule[day][slot][c] is not None and schedule[day][slot][c][0] == t:
                            count += 1
                    if count == 0:
                        # check if he teach more than 7 times in a week
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

            students_per_subject[subject] -= yaml_dict[utils.CLASSROOMS][classroom][utils.CAPACITY]
            schedule[day][slot][classroom] = (teacher, subject)

        return schedule

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

    def __compute_conflicts(self, size: int, schedule: list[int]) -> int:
        ''' Computes the number of conflicts in the given schedule. '''
        _conflicts = 0
        teachers_info = self.yaml_dict[utils.TEACHERS]

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

    def __possible_teacher_replacements(self, day: str, slot: str, classroom: str) -> list[str]:
        ''' Returns a list of teachers that can replace the current teacher. '''
        teachers_info = self.yaml_dict[utils.TEACHERS]
        current_eacher, subject = self.schedule[day][slot][classroom]
        replacements = []

        for new_teacher in teachers_info.keys():
            if subject in teachers_info[new_teacher][utils.SUBJECTS]:
                # teacher can teach the subject
                count = 0
                for c in self.schedule[day][slot].keys():
                    if self.schedule[day][slot][c] is not None and self.schedule[day][slot][c][0] == new_teacher:
                        count += 1
                if count == 0:
                    # teacher is not in another classroom at the same time
                    if day in teachers_info[new_teacher][utils.DAYS] and \
                       slot in teachers_info[new_teacher][utils.SLOTS]:
                        # teacher accepts the day or slot
                        if sum(1 for d in DAYS_OF_THE_WEEK[:self.size[1]]
                               for s in TIME_SLOTS[:self.size[0]]
                               for c in self.schedule[d][s].keys()
                               if self.schedule[d][s][c] is not None and self.schedule[d][s][c][0] == new_teacher) < 7:
                            # teacher has less than 7 hours per week
                            replacements.append(new_teacher)

        return replacements

    def __possible_timeslot_replacements(self, day: str, slot: str, classroom: str) -> list[(str, str)]:
        ''' Returns a list of possible timeslots for the current subject. '''
        teachers_info = self.yaml_dict[utils.TEACHERS]
        teacher, subject = self.schedule[day][slot][classroom]
        replacements = []

        for new_day in DAYS_OF_THE_WEEK[:self.size[1]]:
            for new_slot in TIME_SLOTS[:self.size[0]]:
                if self.schedule[new_day][new_slot][classroom] is None and \
                    (new_day in teachers_info[teacher][utils.DAYS] and \
                    new_slot in teachers_info[teacher][utils.SLOTS]):
                    count = 0
                    for c in self.schedule[new_day][new_slot].keys():
                        if self.schedule[new_day][new_slot][c] is not None and self.schedule[new_day][new_slot][c][0] == teacher:
                            count += 1
                    if count == 0:
                        replacements.append((new_day, new_slot))

        return replacements

    def get_next_states(self) -> list[State]:
        ''' Returns a list of all possible states that can be reached from the current state. '''
        teachers_info = self.yaml_dict[utils.TEACHERS]
        new_states = []

        for day in DAYS_OF_THE_WEEK[:self.size[1]]:
            for slot in TIME_SLOTS[:self.size[0]]:
                for classroom in self.schedule[day][slot].keys():
                    if self.schedule[day][slot][classroom] is not None:
                        teacher, subject = self.schedule[day][slot][classroom]
                        no_conflicts = self.__compute_conflicts_for_timeslot(day, slot, teacher, teachers_info)
                        if no_conflicts > 0:
                            # TODO: Make a list with all the possible teachers that can teach instead of the current one
                            replacements = self.__possible_teacher_replacements(day, slot, classroom)
                            for new_teacher in replacements:
                                new_states.append(self.change_teacher(day, slot, subject, new_teacher))
                            
                            replacements = self.__possible_timeslot_replacements(day, slot, classroom)
                            for new_day, new_slot in replacements:
                                new_states.append(self.change_slot(day, slot, new_day, new_slot, classroom, subject, teacher))
        return new_states

    def display(self, output_file: str = None) -> None:
        ''' Print the current schedule.'''
        if output_file is not None:
            with open(output_file, 'a') as file:
                file.write(utils.pretty_print_timetable(self.schedule, self.file_name))
        else :
            print(utils.pretty_print_timetable(self.schedule, self.file_name))

    def clone(self) -> State:
        ''' Returns a copy of the current schedule.	'''
        return State(self.file_name, self.size, copy.deepcopy(self.schedule), self.nconflicts)
