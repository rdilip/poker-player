import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import sys, math

class TargetModel(nn.Module):

	def __init__(self, num_players):

		super().__init__()

		self.__num_players = num_players
		self.lb = LinearBlock((num_players * 2) + 2, 16, num_players)

	def forward(self, pots, dealer_tensor, big_blind_size, small_blind_size):

		blind_addendum = torch.stack([big_blind_size, small_blind_size]).transpose(1, 0)
		full_input = torch.cat([pots, dealer_tensor, blind_addendum], dim=1)
		scores = self.lb(full_input)
		return scores

class PokerTransformerNetwork(nn.Module):

	def __init__(self, previous_bet_vector_size, current_state_vector_size, number_of_blocks=3, number_of_heads=4, input_size=16):

		super().__init__()

		self.__input_size = input_size
		self.__softmax = nn.Softmax(dim=0)

		self.__previous_bet_lb = LinearBlock(previous_bet_vector_size, input_size * 2, input_size)
		self.__previous_bet_transformer = TransformerNetwork(number_of_blocks=number_of_blocks, number_of_heads=number_of_heads, input_size=input_size)

		self.__current_state_lb_one = LinearBlock(current_state_vector_size, input_size * 2, input_size)
		self.__current_state_lb_two = LinearBlock(current_state_vector_size, input_size * 2, input_size)
		self.__final_lb = LinearBlock(input_size * 2, input_size, input_size)

		self.__scorer = LinearBlock(input_size + 3, 8, 1)

	def forward(self, previous_bets, current_state):

		projected_previous_bets = self.__previous_bet_lb(previous_bets)
		transformed_pb = self.__previous_bet_transformer(projected_previous_bets) # [number of events x input size]

		projected_current_state = self.__current_state_lb_one(current_state) # [input size]

		score_values = torch.matmul(transformed_pb, projected_current_state.unsqueeze(1)) / math.sqrt(self.__input_size)
		scores = self.__softmax(score_values)
		scored_tensors = torch.mul(scores.repeat(1, self.__input_size), transformed_pb)
		final_previous_bet_representation = torch.sum(scored_tensors, dim=0) 

		final_input = torch.cat([final_previous_bet_representation, projected_current_state])
		final_state_representation = self.__final_lb(final_input)

		return final_state_representation

	def play(self, previous_bets, current_state, bet_increment_size, maximum_bet_size, call_size, noise=0, precision=5):

		final_state_representation = self(previous_bets, current_state)

		bet_values = {}

		bet_values[-1] = self.score(final_state_representation, 0, call_size>0, call_size==0)
		bet_values[call_size] = self.score(final_state_representation, call_size, False, True)
		bet_values[maximum_bet_size] = self.score(final_state_representation, maximum_bet_size, False, call_size >= maximum_bet_size)

		best_score = float("-inf")
		best_score_bet = -1

		lower_bound = call_size
		upper_bound = maximum_bet_size
		depth = 0
		max_depth = 10
		while upper_bound - lower_bound > bet_increment_size and depth < max_depth:
			precision_actual = min(precision, int((upper_bound - lower_bound) / bet_increment_size))
			jump_size = max(int(((upper_bound - lower_bound) / precision_actual)), bet_increment_size)
			if jump_size > bet_increment_size and precision_actual == 1:
				jump_size = bet_increment_size
			
			bet_size = lower_bound
			while bet_size <= upper_bound:
				if bet_size not in bet_values:
					bet_values[bet_size] = self.score(final_state_representation, bet_size, False, False)
				bet_size += jump_size

			best_score = float("-inf")
			best_score_bet = -1
			for bet in filter(lambda x: x >= lower_bound and x <= upper_bound, bet_values):
				if bet_values[bet] > best_score:
					best_score_bet = bet
					best_score = bet_values[bet]

			upper_score = bet_values[best_score_bet + jump_size] if best_score_bet + jump_size in bet_values else float("-inf")
			lower_score = bet_values[best_score_bet - jump_size] if best_score_bet - jump_size in bet_values else float("-inf")

			if best_score_bet == maximum_bet_size:
				upper_bound = maximum_bet_size
				lower_bound = maximum_bet_size - jump_size
			elif best_score_bet <= call_size:
				lower_bound = call_size
				upper_bound = call_size + jump_size
			elif upper_score > lower_score:
				upper_bound = best_score_bet + jump_size
				lower_bound = best_score_bet
			else:
				lower_bound = best_score_bet - jump_size
				upper_bound = best_score_bet

			depth += 1

		print(bet_values[-1], bet_values[call_size], bet_values[maximum_bet_size], best_score)
		if bet_values[-1] >= best_score:
			return 0, maximum_bet_size > 0
		else:
			return best_score_bet, False

	def score(self, final_state_representation, bet_size, is_fold, is_call):

		extra_input = torch.tensor(np.asarray([bet_size, 1 if is_fold else 0, 1 if is_call else 0]), dtype=torch.float)
		full_input = torch.cat([final_state_representation, extra_input])
		score = self.__scorer(full_input)
		return score

	def score_bid(self, bet, previous_bets, current_state):

		final_state_representation = self(previous_bets, current_state)
		return self.score(final_state_representation, bet["Bid Amount"], bet["Folded"], bet["Called"])


class TransformerNetwork(nn.Module):

	def __init__(self, number_of_blocks=3, number_of_heads=4, input_size=16):

		super().__init__()

		self.__blocks = [TransformerBlock(number_of_heads=number_of_heads, input_size=input_size) for block in range(number_of_blocks)]

	def forward(self, input_tensor):

		current_input = input_tensor
		for block in self.__blocks:
			current_input = block(current_input)
		return current_input

class TransformerBlock(nn.Module):

	def __init__(self, number_of_heads=4, input_size=16):

		super().__init__()

		self.__number_of_heads = number_of_heads
		self.__input_size = input_size
		self.__heads = nn.Linear(input_size * number_of_heads, input_size * number_of_heads)
		self.__linear_reducer = nn.Linear(input_size * number_of_heads, input_size)
		self.__ff_block = LinearBlock(input_size, input_size, input_size)

		self.__softmax = nn.Softmax(dim=0)
		self.__layer_norm = nn.LayerNorm(input_size)

	def forward(self, input_tensor):

		# input_tensor is going to be number_of_events x input_size
		number_of_events, _ = input_tensor.size()

		repeated_tensor = input_tensor.repeat(1, self.__number_of_heads) # number_of_events x (input_size * number_of_heads)
		projected_tensor = self.__heads(repeated_tensor)
		split_tensor = torch.split(projected_tensor, self.__input_size, dim=1) # tuple w/ number_of_heads entries where each entry is number_of_events * input_size

		# self attention sub-block
		reconstructions = []
		for head_section in split_tensor:
			new_event_representations = []
			inner_product = torch.matmul(head_section, head_section.transpose(1, 0)) / math.sqrt(self.__input_size)
			for i in range(number_of_events):
				scores = self.__softmax(inner_product[i, 0:i+1])
				scored_tensors = torch.mul(head_section[0:i+1, :], scores.unsqueeze(1).repeat(1, self.__input_size))
				reconstructed_tensor = torch.sum(scored_tensors, dim=0)
				new_event_representations.append(reconstructed_tensor)
			head_reconstructed_tensor = torch.stack(new_event_representations)
			normalized_head_reconstructed_tensor = self.__layer_norm(head_reconstructed_tensor + input_tensor)
			reconstructions.append(normalized_head_reconstructed_tensor)
		reconstructed_tensor = torch.cat(reconstructions, dim=1)
		attention_sublayer_output = self.__linear_reducer(reconstructed_tensor)

		# feed-forward sub-block
		penultimate_output = self.__ff_block(attention_sublayer_output)
		final_output = self.__layer_norm(penultimate_output + attention_sublayer_output)

		return final_output

class LinearBlock(nn.Module):

	def __init__(self, input_size, middle_size, output_size):

		super().__init__()

		self.first_layer = nn.Linear(input_size, middle_size)
		self.second_layer = nn.Linear(middle_size, output_size)

	def forward(self, input_tensor):

		return self.second_layer(F.relu(self.first_layer(input_tensor)))

if __name__ == "__main__":

	network = PokerTransformerNetwork(86, 84)
	previous_bets = torch.zeros([15, 86])
	current_state = torch.zeros([84])
	output = network(previous_bets, current_state)
	print(output.size())

