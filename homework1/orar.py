import a_star
import hill_climbing
import sys

if __name__ == '__main__':
	# check if the number of parameters is correct
	if len(sys.argv) != 3:
		print('\nSintaxă greșită! Se rulează de exemplu:\n\npython3 orar.py astar orar_mic_exact\n')
		sys.exit(0)

	# get the 2 parameters from the command line
	algorithm = sys.argv[1]
	name = sys.argv[2]

	# check if the algorithm is A* or Hill Climbing
	if algorithm == 'astar':
		# run the A* algorithm
		a_star.run_test(f'inputs/{name}.yaml', f'statistics/astar/{name}.txt', f'outputs/astar/{name}.txt')
		print(f"\nOutput-ul a fost scris în fișierul outputs/astar/{name}.txt!\n")
	elif algorithm == 'hc':
		# run the Hill Climbing algorithm
		hill_climbing.run_test(f'inputs/{name}.yaml', f'statistics/hc/{name}.txt', f'outputs/hc/{name}.txt')
		print(f"\nOutput-ul a fost scris în fișierul outputs/hc/{name}.txt!\n")
	else:
		print('\nAlgoritmul trebuie să fie "astar" sau "hc"!\n')
		sys.exit(0)