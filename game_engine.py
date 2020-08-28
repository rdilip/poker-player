from player import Player
from hands import Hand, Card, Stage

import random, copy


class GameEngine:

	def __init__(self, players=None, num_players=6, pot_size=6000, small_blind_amount=10, big_blind_amount=20, bid_increments=10):

		self.__cards = []
		with open("cards") as f:
			for line in f.readlines():
				if len(line) > 1:
					self.__cards.append(Card(line.strip()))

		if players == None:
			self.__players = [Player(i, int(pot_size / num_players), bid_increments, num_players) for i in range(num_players)]
		else:
			self.__players = players

		self.__num_players = num_players
		self.__pot_size = pot_size
		self.__small_blind_amount = small_blind_amount
		self.__big_blind_amount = big_blind_amount

		self.__current_state = {"Big Blind": self.__big_blind_amount, "Small Blind": self.__small_blind_amount}
		self.__previous_bets = []

		assert(self.__num_players == len(self.__players))

	def reset(self, shuffle_players=False):
		self.__current_state = {"Big Blind": self.__big_blind_amount, "Small Blind": self.__small_blind_amount}
		self.__previous_bets = []
		for player in self.__players:
			player.reset()

		if shuffle_players:
			random.shuffle(self.__players)
			for i, player in enumerate(self.__players):
				player.set_values(id_num=i)

	def get_player_by_id_num(self, id_num):
		for player in self.__players:
			if player.get_id_num() == id_num:
				return player
		return None

	def play_round_from_specific_moment(self, pots, dealer_idx, small_blind_amount, big_blind_amount, return_bets=True):
		self.reset()
		self.__small_blind_amount = small_blind_amount
		self.__big_blind_amount = big_blind_amount

		for i, player in enumerate(self.__players):
			player.set_pot(pots[i])

		bankrupt_players = []
		for player in self.__players:
			if player.is_bankrupt():
				bankrupt_players.append(player.get_id_num())

		winner = self.play_round(dealer_idx=dealer_idx, bankrupt_players=bankrupt_players)

		if return_bets:
			return self.__previous_bets
		return winner

	def get_settings(self):

		return {"Number of Players": self.__num_players, "Pot Size": self.__pot_size, "Small Blind Amount": self.__small_blind_amount, \
				"Big Blind Amount": self.__big_blind_amount}

	def get_current_pots(self):
		return [player.get_pot() for player in self.__players]


	def play_game(self, max_hand_count=None, display=False, return_results=False):

		results = []
		max_hand_count = max_hand_count if max_hand_count != None else float("inf")
		dealer_idx = 0
		game_number = 0
		bankrupt_players = []
		while len(bankrupt_players) < self.__num_players - 1 and game_number < max_hand_count:
			if display:
				print(f"\nGame {str(game_number + 1)}\n")
			winner = self.play_round(dealer_idx, bankrupt_players, display=display)
			for player in self.__players:
				if display:
					print(player.get_id_num(), player.get_pot())
				if player.is_bankrupt() and player.get_id_num() not in bankrupt_players:
					bankrupt_players.append(player.get_id_num())
			
			dealer_idx = self.find_next_playing_player((dealer_idx + 1) % self.__num_players).get_id_num()
			game_number += 1

			ps_check = 0
			temp_results = []
			for i, player in enumerate(self.__players):
				temp_pot = player.get_pot()
				ps_check += temp_pot
				temp_results.append(temp_pot)

			assert(int(ps_check) == self.__pot_size)
			results.append((temp_results, dealer_idx))

		winner_id = sorted(self.__players, key=lambda x: x.get_pot(), reverse=True)[0].get_id_num()
		if return_results:
			return winner_id, results

	def play_round(self, dealer_idx=0, bankrupt_players=None, display=False):

		self.__previous_bets = []
		self.__current_state = {"Big Blind": self.__big_blind_amount, "Small Blind": self.__small_blind_amount}
		self.__current_state["Dealer Index"] = dealer_idx
		self.__current_state["Cards on Board"] = []
		self.__current_state["Stage"] = Stage.PRE_FLOP
		self.__current_state["Last Raise Index"] = -1
		self.__current_state["Pot"] = 0
		self.__current_state["Folded Players"] = [player_id for player_id in bankrupt_players] if bankrupt_players != None else []
		self.__current_state["Cards on Board"] = []

		cards_on_board = []

		random.shuffle(self.__cards)
		for i, player in enumerate(self.__players):
			player.receive_dealt_cards((self.__cards[2 * i], self.__cards[(2 * i) + 1]))
			if display and not player.is_bankrupt():
				print("Player {} is dealt {}". format(player.get_id_num(), player.reveal_cards()))
		card_idx = 2 * self.__num_players

		folded_players = [player_id for player_id in bankrupt_players] if bankrupt_players != None else []
		bets_by_player, folded_players, last_raiser = self.__do_betting_round(stage=Stage.PRE_FLOP, dealer_idx=dealer_idx, already_folded_players=folded_players, display=display)
		self.__current_state["Pot"] += sum(bets_by_player)

		cards_on_board = [card for card in self.__cards[card_idx:card_idx + 3]]
		self.__current_state["Cards on Board"] = cards_on_board
		if display:
			print(f"{cards_on_board} are revealed")
		card_idx += 3

		for i in range(2):
			temp_bets_by_player, folded_players, last_raiser = self.__do_betting_round(already_folded_players=folded_players, stage=Stage.POST_FLOP if i == 0 else Stage.POST_RIVER, starter=last_raiser, display=display)
			for i, bet in enumerate(temp_bets_by_player):
				bets_by_player[i] += bet
			self.__current_state["Pot"] += sum(bets_by_player)

			cards_on_board.append(self.__cards[card_idx])
			self.__current_state["Cards on Board"] = cards_on_board
			if display:
				print(f"{cards_on_board[-1]} is revealed")
			card_idx += 1

		temp_bets_by_player, folded_players, last_raiser = self.__do_betting_round(already_folded_players=folded_players, stage=Stage.POST_TURN, starter=last_raiser, display=display)
		for i, bet in enumerate(temp_bets_by_player):
			bets_by_player[i] += bet
		self.__current_state["Pot"] += sum(bets_by_player)

		winners = self.__determine_winner(cards_on_board, [(player, player.reveal_cards()) for player in filter(lambda x: x.get_id_num() not in folded_players, self.__players)], display=display)

		#  give up if theres a tie and an all-in
		give_up = False
		if len(winners) > 1:
			for winner in winners:
				if bets_by_player[winner.get_id_num()] < max(bets_by_player):
					for i, bet in enumerate(bets_by_player):
						self.__players[i].win_money(bets_by_player[i])
					give_up = True
					break
		if not give_up:
			for winner in winners:
				if bets_by_player[winner.get_id_num()] == max(bets_by_player):
					winner.win_money(sum(bets_by_player) / len(winners))
				else:
					winnable_sum = 0
					for bet in bets_by_player:
						winnable_sum += min(bets_by_player[winner.get_id_num()], bet)

					amount_won = min(winnable_sum, sum(bets_by_player) / len(winners))
					winner.win_money(amount_won)
					for i, amount_bet in enumerate(bets_by_player):
						if amount_bet > bets_by_player[winner.get_id_num()]:
							self.__players[i].win_money(amount_bet - bets_by_player[winner.get_id_num()])

		return winners

	def __do_betting_round(self, already_folded_players=None, starter=0, stage=Stage.PRE_FLOP, dealer_idx=0, display=False):

		self.__current_state["Stage"] = stage
		self.__current_state["Dealer Index"] = dealer_idx

		last_raise_idx = starter
		current_bidder_idx = starter
		bids_this_round = [0 for _ in range(self.__num_players)]
		outstanding_bid_amount = 0
		folded_players = [] if already_folded_players == None else [idx for idx in already_folded_players]

		if stage == Stage.PRE_FLOP:
			small_blind_player = self.find_next_playing_player((dealer_idx + 1) % self.__num_players)
			small_blind = small_blind_player.pay_blind(self.__small_blind_amount)
			bids_this_round[small_blind_player.get_id_num()] = small_blind

			big_blind_player = self.find_next_playing_player((small_blind_player.get_id_num() + 1) % self.__num_players)
			big_blind = big_blind_player.pay_blind(self.__big_blind_amount)
			outstanding_bid_amount = self.__big_blind_amount
			bids_this_round[big_blind_player.get_id_num()] = big_blind

			current_bidder_idx = (big_blind_player.get_id_num() + 1) % self.__num_players
			last_raise_idx = big_blind_player.get_id_num()

			if display:
				print(f"{small_blind_player} blinds {str(small_blind)}")
				print(f"{big_blind_player} blinds {str(big_blind)}")

			if big_blind_player == small_blind_player:
				for i, player in enumerate(self.__players):
					print(i, player.get_pot() + bids_this_round[i])
				assert(big_blind_player != small_blind_player)

		bid_opportunities_this_round = [(i in folded_players) for i in range(self.__num_players)]
		while (current_bidder_idx != last_raise_idx or not bid_opportunities_this_round[current_bidder_idx]) and len(folded_players) < self.__num_players - 1 and len(self.__previous_bets) < 50:

			bid_opportunities_this_round[current_bidder_idx] = True

			if current_bidder_idx not in folded_players:

				state_to_pass = copy.deepcopy(self.__current_state)
				state_to_pass["Call Amount"] = outstanding_bid_amount - bids_this_round[current_bidder_idx]
				state_to_pass["Last Raise Index"] = last_raise_idx
				state_to_pass["Folded Players"] = [player_id for player_id in folded_players]
				state_to_pass["Player Pots"] = [player.get_pot() for player in self.__players]
				state_to_pass["Pot"] += sum(bids_this_round)
				state_to_pass["Current Bidder Index"] = current_bidder_idx

				folded, bid_amount, all_in = self.__players[current_bidder_idx].bid(current_state=state_to_pass, previous_bets=self.__previous_bets)

				called = False
				if folded:
					if display:
						print(f"{current_bidder_idx} folds")
					folded_players.append(current_bidder_idx)
				else:
					bids_this_round[current_bidder_idx] += bid_amount
					if bids_this_round[current_bidder_idx] > outstanding_bid_amount:
						last_raise_idx = current_bidder_idx
						outstanding_bid_amount = bids_this_round[current_bidder_idx]
						if display:
							print(f"{current_bidder_idx} raises with a bet of {bid_amount}")
					else:
						called = True
						if display:
							print(f"{current_bidder_idx} calls with a bet of {bid_amount}")
					assert all_in or bids_this_round[current_bidder_idx] == outstanding_bid_amount


				last_bet_state = state_to_pass
				last_bet_state["Folded"] = folded
				last_bet_state["Called"] = called
				last_bet_state["Bid Amount"] = bid_amount
				self.__previous_bets.append(last_bet_state)


			if len(folded_players) < self.__num_players - 1:
				current_bidder_idx = (current_bidder_idx + 1) % self.__num_players
				while current_bidder_idx in folded_players:
					current_bidder_idx = (current_bidder_idx + 1) % self.__num_players
		
		return bids_this_round, folded_players, last_raise_idx

	def __determine_winner(self, cards_on_board, hands, display=False):

		if display:
			print("CARDS ON BOARD:", cards_on_board)
		best_hand_overall = None
		winner = None
		for player, hand in hands:
			best_hand_player = Hand.find_best_hand(cards_on_board, hand)
			if display:
				print(f"{player}'s cards are {player.reveal_cards()}. Their best hand is {best_hand_player}")
			if winner == None:
				best_hand_overall = best_hand_player
				winner = [player]
			else:
				is_better, is_tie = Hand.is_first_hand_better(best_hand_player, best_hand_overall)
				if is_better:
					best_hand_overall = best_hand_player
					winner = [player]
				elif is_tie:
					winner = winner + [player]

		if display:
			print(f"The winner is {winner[0]}")

		return winner

	def find_next_playing_player(self, start_idx):

		current_idx = start_idx
		while self.__players[current_idx].is_bankrupt():
			current_idx = (current_idx + 1) % self.__num_players

		return self.__players[current_idx]

