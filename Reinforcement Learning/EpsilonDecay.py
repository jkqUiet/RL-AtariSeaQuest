
epsilon_min = 0.1
episodes_target = 38
epsilon_start = 0.9

epsilon_decay_target = (epsilon_min / epsilon_start) ** (1 / episodes_target)
print(epsilon_decay_target)