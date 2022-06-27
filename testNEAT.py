import neat
import pickle
import basics
import matplotlib.pyplot as plt

# xor_inputs = [[0.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]]
# xor_outputs = [[0.0], [1.0], [1.0], [0.0]]

xor_inputs, xor_outputs, profits, days = basics.getTrainingData()

with open("winner.pickle", "rb") as f:
    winner = pickle.load(f)


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')


winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

winArr = []
profArr = []
daysArr = []
weightArr = []

counter = 0
for xi, xo in zip(xor_inputs, xor_outputs):
    output = winner_net.activate(xi)
    # print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    if output[0]>.90:
        winArr.append(xo)#Im genius dont come at me this is so big brain right here
        profArr.append(profits[counter])
        daysArr.append(days[counter])
        weightArr.append(output[0])

    counter+=1

print("Number of Trades:", len(profArr))
print("Success Rate:", sum(winArr)/len(winArr))
print("Profit Per Trade:", sum(profArr)/len(profArr))

ppdArr = []
for i in range(len(profArr)):
    ppdArr.append(profArr[i]/daysArr[i])

print("Profit Per Day:", sum(ppdArr)/len(ppdArr))

plt.scatter(weightArr, profArr)
plt.show()