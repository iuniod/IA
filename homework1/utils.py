""" This module constains parsing functions for the input data and
    pretty printing functions for the output data. """
import yaml

##################### MACROS #####################
INTERVALS = 'Intervale'
DAYS = 'Zile'
SUBJECTS = 'Materii'
TEACHERS = 'Profesori'
CLASSROOMS = 'Sali'
CONSTRAINTS = 'Constrangeri'
CAPACITY = 'Capacitate'

def process_teacher_constraints(constraints: list) -> (list, int):
    ''' Process the constraints of a teacher and return the days, slots and break limit. '''
    days = []
    slots = []
    break_limit = -1

    for i, constraint in enumerate(constraints):
        if i < 5:
            if constraint[0] != '!':
                days.append(constraint)
        else:
            if constraint[0:6] == '!Pauza':
                break_limit = int(constraint[8:])
            elif constraint[0] != '!':
                index = constraint.index('-')
                first = int(constraint[0:index])
                last = int(constraint[index + 1:])
                while first < last:
                    slots.append((first, first + 2))
                    first += 2
        slots.sort()

    return days, slots, break_limit if break_limit != -1 else None

def read_yaml_file(file_path: str) -> dict:
    ''' Read a yaml file and return its content as a dictionary.'''
    parse_file = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        parse_file = yaml.safe_load(file)

    # code all the constraints in a more accessible way to check them
    for teacher in parse_file[TEACHERS]:
        days, slots, break_limit = process_teacher_constraints(parse_file[TEACHERS][teacher][CONSTRAINTS])

        subjects = parse_file[TEACHERS][teacher]['Materii']
        parse_file[TEACHERS][teacher] = { 'Zile': days,
                                   'Sloturi': slots,
                                   'Materii': subjects,
                                   'Pauza': break_limit if break_limit != -1 else None}
    return parse_file

def acces_yaml_attributes(yaml_dict: dict):
    ''' Print the data from the yaml dictionary (in Romanian).'''

    print('Zilele din orar sunt:', yaml_dict[DAYS])
    print()
    print('Intervalele orarului sunt:', yaml_dict[INTERVALS])
    print()
    print('Materiile sunt:', yaml_dict[SUBJECTS])
    print()
    print('Profesorii sunt:', end=' ')
    print(*list(yaml_dict[TEACHERS].keys()), sep=', ')
    print()
    print('Sălile sunt:', end=' ')
    print(*list(yaml_dict[CLASSROOMS].keys()), sep=', ')
    print("Constrangerile fiecarui profesor sunt:")
    for teacher in yaml_dict[TEACHERS]:
        print(f"{teacher}: {yaml_dict[TEACHERS][teacher]}")


def get_teachers_initials(teachers: list) -> dict:
    ''' Get a list of teachers and return two dictionaries:
    - one that has the names of the teachers as keys and their initials as values:
        (teacher_to_initials[teacher] = initials)
    - one that has the initials of the teachers as keys and their names as values:
        (initials_to_teacher[initials] = teacher) '''

    initials_to_teacher = {}
    teacher_to_initials = {}
    initials_count = {}

    for teacher in teachers:
        name_components = teacher.split(' ')
        initials = name_components[0][0] + name_components[1][0]

        if initials in initials_count:
            initials_count[initials] += 1
            initials += str(initials_count[initials])
        else:
            initials_count[initials] = 1

        initials_to_teacher[initials] = teacher
        teacher_to_initials[teacher] = initials

    return teacher_to_initials, initials_to_teacher


def allign_string_with_spaces(s: str, max_len: int, allignment_type: str = 'center') -> str:
    ''' Receives a string and an integer, and returns the string, filled with spaces
        until it reaches the given length. '''

    len_str = len(s)

    if len_str >= max_len:
        raise ValueError('Length of the string is greater than the given maximum length')


    if allignment_type == 'left':
        s = 6 * ' ' + s
        s += (max_len - len(s)) * ' '

    elif allignment_type == 'center':
        if len_str % 2 == 1:
            s = ' ' + s
        s = s.center(max_len, ' ')

    return s


def pretty_print_timetable_aux_zile(
    timetable_schedule: {str: {(int, int): {str: (str, str)}}},
    input_path: str
    ) -> str:
    ''' Receives a dictionary with the days as keys and with values as dictionaries of intervals
        represented as tuples of integers, with values dictionaries of classrooms,
        with values tuples (teacher, subject).
        Returns a string formatted to look like an excel table with the days on the lines,
        the intervals on the columns and in the intersection of these, the 2 hour windows
        with the subjects allocated in each classroom to each teacher. '''

    max_len = 30

    teachers_to_initials, _ = get_teachers_initials(read_yaml_file(input_path)[TEACHERS].keys())

    table_str = '|           Interval           |             Luni             ' + \
                '|             Marti            |           Miercuri           ' + \
                '|              Joi             |            Vineri            |\n'

    no_classes = len(timetable_schedule['Luni'][(8, 10)])

    delim = '-' * 187 + '\n'
    table_str += delim

    for interval in timetable_schedule['Luni']:
        s_interval = '|'

        crt_str = allign_string_with_spaces(f'{interval[0]} - {interval[1]}', max_len, 'center')

        s_interval += crt_str

        for class_idx in range(no_classes):
            if class_idx != 0:
                s_interval += f'|{30 * " "}'

            for day in timetable_schedule:
                classes = timetable_schedule[day][interval]
                classroom = list(classes.keys())[class_idx]

                s_interval += '|'

                if not classes[classroom]:
                    s_interval += allign_string_with_spaces(f'{classroom} - goala', max_len, 'left')
                else:
                    teacher, subject = classes[classroom]
                    s_interval += allign_string_with_spaces(
                                    f'{subject} : ({classroom} - {teachers_to_initials[teacher]})',
                                    max_len, 'left')

            s_interval += '|\n'
        table_str += s_interval + delim

    return table_str

def pretty_print_timetable_aux_intervale(
    timetable_schedule: {(int, int): {str: {str: (str, str)}}},
    input_path: str
    ) -> str:
    ''' Receives a dictionary with intervals represented as tuples of integers,
        with values dictionaries of days, with values dictionaries of classrooms,
        with values tuples (teacher, subject).
        Returns a string formatted to look like an excel table with the days on the lines,
        the intervals on the columns and in the intersection of these, the 2 hour windows
        with the subjects allocated in each classroom to each teacher. '''

    max_len = 30

    teachers = read_yaml_file(input_path)[TEACHERS].keys()
    teachers_to_initials, _ = get_teachers_initials(teachers)

    table_str = '|           Interval           |             Luni             ' + \
                '|             Marti            |           Miercuri           ' + \
                '|              Joi             |            Vineri            |\n'

    no_classes = len(timetable_schedule[(8, 10)]['Luni'])

    delim = '-' * 187 + '\n'
    table_str = table_str + delim

    for interval in timetable_schedule:
        s_interval = '|' + allign_string_with_spaces(
            f'{interval[0]} - {interval[1]}', max_len, 'center')

        for class_idx in range(no_classes):
            if class_idx != 0:
                s_interval += '|'

            for day in timetable_schedule[interval]:
                classes = timetable_schedule[interval][day]
                classroom = list(classes.keys())[class_idx]

                s_interval += '|'

                if not classes[classroom]:
                    s_interval += allign_string_with_spaces(f'{classroom} - goala', max_len, 'left')
                else:
                    teacher, subject = classes[classroom]
                    s_interval += allign_string_with_spaces(f'{subject} : ({classroom} - \
                                    {teachers_to_initials[teacher]})', max_len, 'left')

            s_interval += '|\n'
        table_str += s_interval + delim

    return table_str

def pretty_print_timetable(timetable_schedule: dict, input_path: str) -> str:
    ''' Receives either a dictionary of days containing dictionaries of intervals containing
        dictionaries of classrooms with tuples (teacher, subject)   or a dictionary of intervals
        containing dictionaries of days containing dictionaries of classrooms with tuples:
        (teacher, subject).
        For the case in which a classroom is not occupied at a certain time, 'None' is expected
        as value, instead of a tuple. '''
    if 'Luni' in timetable_schedule:
        return pretty_print_timetable_aux_zile(timetable_schedule, input_path)
    return pretty_print_timetable_aux_intervale(timetable_schedule, input_path)


if __name__ == '__main__':
    FILENAME = 'inputs/orar_mic_exact.yaml'

    timetable_specs = read_yaml_file(FILENAME)

    acces_yaml_attributes(timetable_specs)

    timetable = {
        'Luni': {
            (8, 10): {
                'Sala 1': None,
                'Sala 2': ('Alexandru Popa', 'PL'),
                'Sala 3': ('Andrei Ionescu', 'IA')
            },
            (10, 12): {
                'Sala 1': ('Alexandru Popa', 'PL'),
                'Sala 2': ('Andrei Ionescu', 'IA'),
                'Sala 3': None
            }
        },
        'Marti': {
            (8, 10): {
                'Sala 1': ('Alexandru Popa', 'PL'),
                'Sala 2': ('Andrei Ionescu', 'IA'),
                'Sala 3': None
            },
            (10, 12): {
                'Sala 1': ('Alexandru Popa', 'PL'),
                'Sala 2': None,
                'Sala 3': ('Andrei Ionescu', 'IA')
            }
        }
    }
    print(pretty_print_timetable(timetable, FILENAME))
