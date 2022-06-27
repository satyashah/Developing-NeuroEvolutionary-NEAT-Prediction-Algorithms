import neat
import pickle
import basics
import matplotlib.pyplot as plt

# xor_inputs = [[0.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]]
# xor_outputs = [[0.0], [1.0], [1.0], [0.0]]
train_features, train_labelsTP, train_labelsSL = basics.getLimitsTrainingData()

with open("winner.pickle", "rb") as f:
    winner = pickle.load(f)


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'limit-config')


winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

errArr = []

for xi, xo in zip(train_features, train_labelsTP):
    output = winner_net.activate(xi)
    # print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    errArr.append(abs(output[0]-xo/xo))

print("Average Error:", sum(errArr)/len(errArr))


# plt.scatter(weightArr, profArr)
# plt.show()