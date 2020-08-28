from player import Player
from game_engine import GameEngine
from models import TargetModel
from hands import Stage

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import itertools, argparse, random, pickle, copy, math, time, sys, os, re


def find_presaved_agents():
	eligible = []
	for filename in os.listdir(path=args.agent_path):
		if re.match("poker_agent\d+.pt", filename) != None:
			version_number = int(re.match("poker_agent(\d+).pt", filename).group(1))
			eligible.append((int(version_number), filename))
	eligible = sorted(eligible, key=lambda x: int(x[0]), reverse=True)
	return eligible

def rotate_bets(bet_dict):
	bet_dict_versions = []
	suit_ordering = ["D", "H", "S", "C"]
	for perm in itertools.permutations(suit_ordering):
		new_version = copy.deepcopy(bet_dict)
		old_cards = new_version["Cards on Board"]
		new_cards = []
		for i, card in enumerate(old_cards):
			new_suit = perm[suit_ordering.index(card.get_suit())]
			card.switch_suit(new_suit)
			new_cards.append(card)
		new_version["Cards on Board"] = new_cards
		bet_dict_versions.append(new_version)
	return bet_dict_versions

def construct_onehot_dealer_tensor(dealer_idxs):
	dealer_indices = torch.tensor(dealer_idxs, dtype=torch.long).unsqueeze(1)
	dealer_tensor = torch.zeros([len(dealer_idxs), args.num_players], dtype=torch.float)
	dealer_tensor.scatter_(1, dealer_indices, 1)
	return dealer_tensor

def rotate_results(winner_id, result_to_add, dealer_idx):
	all_data = []
	for i in range(args.num_players):
		new_result_to_add = [result_to_add[(j + i) % args.num_players] for j in range(args.num_players)]
		new_winner_id = (winner_id - i) % args.num_players
		new_dealer_idx = (dealer_idx - i) % args.num_players
		all_data.append((new_winner_id, new_result_to_add, new_dealer_idx))
	return all_data

# parses args
parser = argparse.ArgumentParser()
parser.add_argument("--player_pot_sizes", default=100, type=int, help="how much money each player starts with")
parser.add_argument("--num_players", default=6, type=int, help="number of players in the game")
parser.add_argument("--small_blind_size", default=1, type=int, help="size of small blind")
parser.add_argument("--big_blind_size", default=2, type=int, help="size of big blind")
parser.add_argument("--bid_increments", default=1, type=int, help="increments that bets must be in")
parser.add_argument("--constant_blinds", default=False, action="store_true", help="whether the blind sizes should vary")
parser.add_argument("--target_model_path", default="pretrained/target_model.pt", type=str, help="path to location of target model (if it exists)")
parser.add_argument("--target_model_data_path", default="data/target_model_data.p", type=str, help="path to location of target model data (if it exists)")
parser.add_argument("--agent_path", default="pretrained", type=str, help="folder with the agents saved in them")
parser.add_argument("--agent_data_path", default="data/agent_model_data.p", type=str, help="path to location of agent model data (if it exists")
parser.add_argument("--no_train", default=False, action="store_true", help="simulates a run with no training")
args = parser.parse_args()




if not args.no_train:

	# load target model data
	target_model_data = []
	if os.path.exists(args.target_model_data_path):
		with open(args.target_model_data_path, "rb") as f:
			target_model_data = pickle.load(f)

	# loads target model
	target_model = TargetModel(num_players=args.num_players)
	if os.path.exists(args.target_model_path):
		target_model.load_state_dict(torch.load(args.target_model_path))
	target_model_optimizer = optim.AdamW(target_model.parameters(), lr=10e-3)
	target_model_criterion = nn.CrossEntropyLoss()

	# load agent data
	agent_data = []
	if os.path.exists(args.agent_data_path):
		with open(args.agent_data_path, "rb") as f:
			agent_data = pickle.load(f)

# selects your agent
main = Player()
presaved_agents = find_presaved_agents()
if len(presaved_agents) > 0:
	main_name = args.agent_path + "/" + presaved_agents[0][1]
	main.model.load_state_dict(torch.load(main_name))
	main.give_name(presaved_agents[0][1])
agent_optimizer = optim.AdamW(main.model.parameters(), lr=10e-5)
agent_criterion = nn.MSELoss()

start_time = time.time()

iteration_number = 0

while True:

	# selects the other agents to play
	presaved_agents = find_presaved_agents()
	if len(presaved_agents) > 0:
		if len(presaved_agents) > 1:
			opponent_choices = random.choices(presaved_agents, weights=[int(pair[0]) for pair in presaved_agents], k=args.num_players - 1)
		else:
			opponent_choices = [presaved_agents[0] for i in range(args.num_players - 1)]
		opponents = [Player() for i in range(args.num_players - 1)]
		for i, opponent in enumerate(opponents):
			opponent.model.load_state_dict(torch.load(args.agent_path + "/" + opponent_choices[i][1]))
			print("Playing against {}".format(opponent_choices[i][1]))
			opponent.give_name(opponent_choices[i][1])
	else:
		opponents = [Player() for i in range(args.num_players - 1)]

	# gets players ready to play
	all_players = [opponent for opponent in opponents] + [main]
	random.shuffle(all_players)
	for i, player in enumerate(all_players):
		player.set_values(id_num=i, pot=args.player_pot_sizes, smallest_bid_increment=args.bid_increments, num_players=args.num_players)

	# create game engine
	engine = GameEngine(players=all_players, num_players=args.num_players, pot_size=args.player_pot_sizes * args.num_players, \
							small_blind_amount=args.small_blind_size, big_blind_amount=args.big_blind_size, bid_increments=args.bid_increments)
	engine.reset(shuffle_players=True)

	# get target model data
	main_wins = 0
	opponent_wins = [0 for i in range(5)]
	max_iters = 1
	for i in tqdm(range(max_iters)):
		winner_id, results = engine.play_game(max_hand_count=500, display=True, return_results=True)

		if winner_id == main.get_id_num():
			main_wins += 1
		else:
			for i, opponent in enumerate(opponents):
				if winner_id == opponent.get_id_num():
					opponent_wins[i] += 1

		if not args.no_train and False:
			result_to_add, dealer_idx = random.choice(results)
			while result_to_add.count(0) >= 5:
				result_to_add, dealer_idx = random.choice(results)
			settings = engine.get_settings()
			rotated_data_to_add = rotate_results(winner_id, result_to_add, dealer_idx)
			for winner_id, result_to_add, dealer_idx in rotated_data_to_add:
				target_model_data.append((winner_id, result_to_add, dealer_idx, settings["Small Blind Amount"], settings["Big Blind Amount"]))

		engine.reset(shuffle_players=True)

	print("{} won {}% of the time".format(main.get_name(), round(main_wins * 100 / max_iters, 2)))
	for i, opponent_win in enumerate(opponent_wins):
		print("{} won {}% of the time".format(opponents[i].get_name(), round(opponent_win * 100 / max_iters, 2)))

	if args.no_train:
		sys.exit()

	with open(args.target_model_data_path, "wb") as f:
		pickle.dump(target_model_data, f)

	# train target model
	total_loss = 0
	max_iters = 256
	for i in tqdm(range(max_iters)):
		raw_batch = random.choices(target_model_data, k=32)

		labels = []
		pots = []
		dealer_idxs = []
		small_blind_amounts = []
		big_blind_amounts = []
		for label, player_pots, dealer_idx, small_blind_amount, big_blind_amount in raw_batch:
			labels.append(label)
			pots.append(player_pots)
			dealer_idxs.append(dealer_idx)
			small_blind_amounts.append(small_blind_amount)
			big_blind_amounts.append(big_blind_amount)

		probs = target_model(torch.tensor(pots, dtype=torch.float), construct_onehot_dealer_tensor(dealer_idxs), \
			torch.tensor(small_blind_amounts, dtype=torch.float), torch.tensor(big_blind_amounts, dtype=torch.float))

		labels_tensor = torch.tensor(labels, dtype=torch.long)
		loss = target_model_criterion(probs, labels_tensor)
		loss.backward()
		target_model_optimizer.step()
		target_model_optimizer.zero_grad()
		total_loss += loss.item()

	torch.save(target_model.state_dict(), args.target_model_path)
	print("Average target model loss was {}".format(round(total_loss / max_iters, 2)))

	# get agent model data
	max_iters = 256
	for i in tqdm(range(max_iters)):
		engine.reset(shuffle_players=True)
		_, pots, dealer_idx, small_blind_amount, big_blind_amount = random.choice(target_model_data)
		previous_bets = engine.play_round_from_specific_moment(pots, dealer_idx, small_blind_amount, big_blind_amount)
		selection_index = random.choices(list(range(len(previous_bets))), weights=[math.sqrt(j) for j in range(len(previous_bets))], k=1)[0] if len(previous_bets) > 1 else 0
		bet_to_consider = previous_bets[selection_index]
		bidder_index = bet_to_consider["Current Bidder Index"]
		bidder_cards = engine.get_player_by_id_num(bidder_index).reveal_cards()
		final_pots = engine.get_current_pots()
		next_dealer_idx = engine.find_next_playing_player(dealer_idx).get_id_num()

		rotated_bets = rotate_bets(bet_to_consider)
		rotated_previous_bets = [rotate_bets(previous_bet) for previous_bet in previous_bets[0:selection_index]]
		for i in range(len(rotated_bets)):
			relevant_rotated_previous_bets = [bet[i] for bet in rotated_previous_bets]
			agent_data.append((bidder_index, bidder_cards, rotated_bets[i], relevant_rotated_previous_bets, next_dealer_idx, final_pots))

	if len(agent_data) > 100000:
		agent_data = agent_data[-100000:]

	with open(args.agent_data_path, "wb") as f:
		pickle.dump(agent_data, f)

	# train agent model
	total_loss = 0
	max_iters = 12288
	for i in tqdm(range(max_iters)):
		engine.reset()
		bidder_index, bidder_cards, bet_to_consider, previous_bets, next_dealer_idx, final_pots = random.choice(agent_data)

		main.set_pot(bet_to_consider["Player Pots"][bidder_index])
		main.receive_dealt_cards(bidder_cards)
		main.set_id_num(bidder_index)
		score = main.bid(bet_to_consider, previous_bets, bet_to_consider)
		with torch.no_grad():
			victory_scores = target_model(torch.tensor(final_pots, dtype=torch.float).unsqueeze(0), construct_onehot_dealer_tensor([next_dealer_idx]), \
								torch.tensor(bet_to_consider["Big Blind"], dtype=torch.float).unsqueeze(0), torch.tensor(bet_to_consider["Small Blind"], dtype=torch.float).unsqueeze(0))
			label = victory_scores[0, bidder_index]

		loss = agent_criterion(score.squeeze(), label)
		loss.backward()
		agent_optimizer.step()
		agent_optimizer.zero_grad()
		total_loss += loss.item()
	print("Average agent model loss was {}".format(round(total_loss / max_iters, 2)))

	new_number_string = str(presaved_agents[0][0] + 1) if len(presaved_agents) > 0 else "0"
	torch.save(main.model.state_dict(), f"{args.agent_path}/poker_agent{new_number_string}.pt")
	print(f"Saving {args.agent_path}/poker_agent{new_number_string}.pt")
	main.give_name(f"poker_agent{new_number_string}.pt")
	print("Total training time is {} seconds\n".format(round(time.time() - start_time, 0)))





