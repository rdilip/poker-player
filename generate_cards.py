
order = [str(i) for i in range(2, 11)] + ["J", "Q", "K", "A"]
suits = ["S", "H", "C", "D"]

for num in order:
	for suit in suits:
		print(num, suit)
