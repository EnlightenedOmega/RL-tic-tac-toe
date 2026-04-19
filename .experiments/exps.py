import numpy as np
state = np.array([1, 0, -1, 0, 1, 0, -1, 0, 0])

valid_moves = np.where(state == 0)[0]
print(valid_moves)