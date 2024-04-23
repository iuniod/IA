""" Module that contains the implementation of a state. """
from __future__ import annotations
from typing import Callable
from copy import copy

import numpy as np
import random
import os

class State:
	def __init__(
		self, 
		size: int, 
		board: list[int] | None = None, 
		conflicts: int | None = None, 
		seed: int = 42
	) -> None:
		
		self.size = size
		self.board = board if board is not None else State.generate_board(size, seed)
		self.nconflicts = conflicts if conflicts is not None \
			else State.__compute_conflicts(self.size, self.board)
	
	def apply_move(self, queen: int, new_row: int) -> State:
		'''
		Construiește o stare vecină în care dama queen este mutată pe linia new_row.
		
		Numărul de conflicte este calculat prin diferența față de starea originală.
		'''
		old_row = self.board[queen]
		new_state = copy(self.board)
		new_state[queen] = new_row
		_conflicts = self.nconflicts
		for i in range(self.size):
			if i != queen:
				if self.board[i] == old_row: _conflicts -= 1
				if abs(self.board[i] - old_row) == abs(i - queen): _conflicts -= 1
				if self.board[i] == new_row: _conflicts += 1
				if abs(self.board[i] - new_row) == abs(i - queen): _conflicts += 1
		return State(self.size, new_state, _conflicts)
	
	@staticmethod
	def generate_board(size: int, seed: int) -> list[int]:
		'''
		Construiește o tablă de mărimea dată cu damele poziționate pe rânduri aleatoare.
		'''
		random.seed(seed)
		board = list(range(size))
		random.shuffle(board)

		return board
	
	@staticmethod
	def __compute_conflicts(size: int, board: list[int]) -> int:
		'''
		Calculează numărul de conflicte parcurgând toate perechile de dame
		'''
		_conflicts = 0
		for i in range(size):
			for j in range(i + 1, size):
				if board[i] == board[j]: _conflicts += 1
				if abs(board[i] - board[j]) == (j - i): _conflicts += 1

		return _conflicts
	
	def conflicts(self) -> int:
		'''
		Întoarce numărul de conflicte din această stare.
		'''
		return self.nconflicts
	
	def is_final(self) -> bool:
		'''
		Întoarce True dacă este stare finală.
		'''
		return self.nconflicts == 0
	
	def get_next_states(self) -> list[State]:
		'''
		Întoarce un generator cu toate posibilele stări următoare.
		'''
		return (self.apply_move(col, row) for col in range(self.size)
				for row in range(self.size) if row != self.board[col])
	
	def __str__(self) -> str:
		board = " " + "_ " * self.size + "\n"
		board += "|\n".join("|" + "|".join(("Q" if col == self.board[row] else"_") 
											for row in range(self.size)) 
											for col in range(self.size))
		board += "|\n"
		return board
	
	def display(self) -> None:
		'''
		Afișează tablei de joc
		'''
		print(self)
	
	def clone(self) -> State:
		'''
		Clonează tabla de joc
		'''
		return State(self.size, copy(self.board), self.nconflicts)