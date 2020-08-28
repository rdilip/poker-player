from game_engine import GameEngine

import pickle

with open("data/target_model_data.p", "rb") as f:
	data = pickle.load(f)

for entry in data:
	bankrupts = entry[1].count(0)
	if bankrupts > 4:
		print("uh oh")
print(len(data))

#engine = GameEngine()
#engine.play_game(display=True)
