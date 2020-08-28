import random, sys
import numpy as np
from hands import Stage, Card
from models import PokerTransformerNetwork
import torch

class Player:

	def __init__(self, id_num=-1, pot=0, smallest_bid_increment=0, num_players=6, name=None):

		self.__id_num = id_num
		self.__pot = pot
		self.__current_cards = ["", ""]
		self.__smallest_bid_increment = smallest_bid_increment
		self.__num_players = num_players
		self.__name = name

		self.__original_pot = self.__pot

		self.model = PokerTransformerNetwork(previous_bet_vector_size=60 + (5 * self.__num_players) + 4, \
												current_state_vector_size=60 + (5 * self.__num_players), \
												number_of_blocks=3, number_of_heads=4, input_size=16)

	def set_values(self, id_num=None, pot=None, smallest_bid_increment=None, num_players=None):
		self.__id_num = id_num if id_num != None else self.__id_num
		self.__pot = pot if pot != None else self.__pot
		self.__smallest_bid_increment = smallest_bid_increment if smallest_bid_increment != None else self.__smallest_bid_increment
		self.__num_players = num_players if num_players != None else self.__num_players
		self.__original_pot = self.__pot

	def give_name(self, name):
		self.__name = name

	def get_name(self):
		return self.__name

	def reset(self):
		self.__pot = self.__original_pot
		self.__current_cards = ["", ""]

	def return_cards(self):
		self.__current_cards = ["", ""]

	def set_id_num(self, id_num):
		self.__id_num = id_num

	def get_id_num(self):
		return self.__id_num

	def set_pot(self, pot):
		self.__pot = pot

	def receive_dealt_cards(self, cards):

		self.__current_cards = list(cards)

	def reveal_cards(self):

		return self.__current_cards

	def get_pot(self):

		return self.__pot

	def win_money(self, winnings):

		self.__pot += winnings

	def bid(self, current_state, previous_bets, bid_to_evaluate=None):

		def calculate_distance(other_index):
			return (self.__id_num - other_index) % self.__num_players

		def get_card_index(card):
			if card.get_suit() in ["D", "d", "Diamonds"]:
				return card.get_number() - 2
			elif card.get_suit() in ["H", "h", "Hearts"]:
				return card.get_number() + 11
			elif card.get_suit() in ["S", "s", "Spades"]:
				return card.get_number() + 24
			return card.get_number() + 37

		def get_stage_number(stage):
			if stage == Stage.PRE_FLOP:
				return 0
			elif stage == Stage.POST_FLOP:
				return 1
			elif stage == Stage.POST_RIVER:
				return 2
			return 3

		def convert_state(state, as_tensor=False, include_cards_in_hand=True):
			current_state_rep = np.zeros([60 + (5 * self.__num_players)])
			current_state_rep[0] = state["Big Blind"]
			current_state_rep[1] = state["Small Blind"]
			current_state_rep[2] = state["Call Amount"]
			current_state_rep[3] = state["Pot"]
			current_state_rep[4 + get_stage_number(state["Stage"])] = 1

			for player_id in state["Folded Players"]:
				current_state_rep[8 + calculate_distance(player_id)] = 1

			current_state_rep[8 + self.__num_players + calculate_distance(state["Dealer Index"])] = 1
			current_state_rep[8 + (2 * self.__num_players) + calculate_distance(state["Last Raise Index"])] = 1
			current_state_rep[8 + (3 * self.__num_players) + calculate_distance(state["Current Bidder Index"])] = 1

			for i, budget in enumerate(state["Player Pots"]):
				current_state_rep[8 + (4 * self.__num_players) + calculate_distance(i)] = budget

			relevant_cards = state["Cards on Board"] + self.__current_cards if include_cards_in_hand else state["Cards on Board"]
			for card in relevant_cards:
				current_state_rep[8 + (5 * self.__num_players) + get_card_index(card)] = 1

			return torch.tensor(current_state_rep, dtype=torch.float) if as_tensor else current_state_rep

		def convert_previous_bets():

			previous_bet_reps = [np.zeros(60 + (5 * self.__num_players) + 4)]
			previous_bet_reps[0][-1] = 1
			for bet in previous_bets:
				bet_rep = np.concatenate([convert_state(bet), np.zeros(4)], axis=0)
				bet_rep[-4] = 1 if bet["Folded"] else 0
				bet_rep[-3] = 1 if bet["Called"] else 0
				bet_rep[-2] = bet["Bid Amount"]
				previous_bet_reps.append(bet_rep)

			previous_bet_matrix = np.stack(previous_bet_reps)
			return torch.tensor(previous_bet_matrix, dtype=torch.float)

		state_rep = convert_state(current_state, as_tensor=True)
		previous_state_rep = convert_previous_bets()
		if bid_to_evaluate == None:
			with torch.no_grad():
				bid_amount, fold = self.model.play(previous_state_rep, state_rep, self.__smallest_bid_increment, self.__pot, current_state["Call Amount"], noise=0, precision=5)
			if not fold:
				bid_amount = min(bid_amount, self.__pot)
				self.__pot -= bid_amount
			return fold, bid_amount, self.__pot <= 0
		else:
			return self.model.score_bid(bid_to_evaluate, previous_state_rep, state_rep)

	def pay_blind(self, amount):

		amount_paid = min(self.__pot, amount)
		self.__pot -= amount_paid
		return amount_paid

	def is_bankrupt(self):

		return self.__pot <= 0

	def __str__(self):
		return str(self.__id_num)

	def __repr__(self):
		return self.__id_num

