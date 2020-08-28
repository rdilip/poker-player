from enum import Enum
import functools, itertools, sys

@functools.total_ordering
class Card:

	def __init__(self, card_text):

		self.__card_text = card_text

		split_text = card_text.split()

		number = split_text[0]
		specials = ["J", "Q", "K", "A"]
		if number in specials:
			number = 11 + specials.index(number)
		self.__number = int(number)
		self.__suit = split_text[1]

	def get_suit(self):
		return self.__suit

	def switch_suit(self, new_suit):
		self.__suit = new_suit
		self.__card_text = self.__card_text[:-1] + new_suit

	def get_card_text(self):
		return self.__card_text

	def get_number(self):
		return self.__number

	def is_ace(self):
		return self.__number == 14

	def __lt__(self, other):
	    return self.__number < other.get_number()
	
	def __eq__(self, other):
	 	return self.__number == other.get_number()

	def __str__(self):
	 	return self.__card_text

	def __repr__(self):
		return self.__card_text

class Hand(Enum):
	HIGH_CARD = 0
	PAIR = 1
	TWO_PAIR = 2
	TRIPS = 3
	STRAIGHT = 4
	FLUSH = 5
	FULL_HOUSE = 6
	QUADS = 7
	STRAIGHT_FLUSH = 8

	@staticmethod
	def is_first_hand_better(hand_one, hand_two):

		def get_most_relevant(cards, marked):
			most_relevant_cards = []
			for i, mark in enumerate(marked):
				if mark:
					most_relevant_cards.append(cards[i])
			return most_relevant_cards

		hand_one_class, hand_one_cards, hand_one_marked = hand_one
		hand_two_class, hand_two_cards, hand_two_marked = hand_two

		if hand_one_class != hand_two_class:
			return hand_one_class._value_ > hand_two_class._value_, False

		for card_one, card_two in zip(get_most_relevant(hand_one_cards, hand_one_marked), get_most_relevant(hand_two_cards, hand_two_marked)):
			if card_one != card_two:
				return card_one > card_two, False

		for card_one, card_two in zip(hand_one_cards, hand_two_cards):
			if card_one != card_two:
				return card_one > card_two, False

		return False, True

	@staticmethod
	def classify_five_card_hand(five_cards):
		cards_in_order = sorted(five_cards, reverse=True)
		marked = [False for i in range(5)]

		# if flush
		flush = True
		main_suit = cards_in_order[0].get_suit()
		for card in cards_in_order[1:]:
			if card.get_suit() != main_suit:
				flush = False
				break

		# if straight
		straight = True
		current_gap = 1
		previous_card = cards_in_order[0]
		for card in cards_in_order[1:]:
			gap = previous_card.get_number() - card.get_number()
			if gap != 1 and not (previous_card.is_ace() and card.get_number() == 5):
				straight = False
				break

		if straight and flush:
			marked = [True for i in range(5)]
			return Hand.STRAIGHT_FLUSH, cards_in_order, marked

		multiples = [0 for _ in range(16)]
		for card in cards_in_order:
			multiples[card.get_number()] += 1

		if 4 in multiples:
			marked = [True, True, True, True, False] if cards_in_order[0] == cards_in_order[1] else [False, True, True, True, True]
			return Hand.QUADS, cards_in_order, marked

		if 3 in multiples and 2 in multiples:
			marked = [True for i in range(5)]
			return Hand.FULL_HOUSE, cards_in_order, marked

		if flush:
			marked = [True for i in range(5)]
			return Hand.FLUSH, cards_in_order, marked

		if straight:
			marked = [True for i in range(5)]
			return Hand.STRAIGHT, cards_in_order, marked

		if 3 in multiples:
			triple_value = multiples.index(3)
			marked = [cards_in_order[k].get_number() == triple_value for k in range(5)]

			return Hand.TRIPS, cards_in_order, marked

		if multiples.count(2) == 2:
			marked = [True for i in range(5)]
			for i in range(5):
				if multiples[cards_in_order[i].get_number()] != 2:
					marked[i] = False
					break
			return Hand.TWO_PAIR, cards_in_order, marked

		if 2 in multiples:
			marked = [True for i in range(5)]
			for i in range(5):
				if multiples[cards_in_order[i].get_number()] != 2:
					marked[i] = False
			return Hand.PAIR, cards_in_order, marked

		return Hand.HIGH_CARD, cards_in_order, marked

	@staticmethod
	def find_best_hand(cards_on_board, hand):

		all_cards = cards_on_board + list(hand)
		best_hand = None

		for five_cards in itertools.combinations(all_cards, 5):
			current_hand = Hand.classify_five_card_hand(five_cards)
			if best_hand == None:
				best_hand = current_hand
			else:
				is_better, _ = Hand.is_first_hand_better(current_hand, best_hand)
				if is_better:
					best_hand = current_hand

		return best_hand

class Stage(Enum):
	PRE_FLOP = 0
	POST_FLOP = 1
	POST_RIVER = 2
	POST_TURN = 3

