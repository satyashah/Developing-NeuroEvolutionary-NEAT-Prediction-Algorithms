from distutils.log import error
import os

import neat
import pickle
import basics
# 2-input XOR inputs and expected outputs.
# xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
# xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]




xor_inputs, xor_outputs, profits, days = basics.getTrainingData()


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        errorArr = []

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)

            if xo == 1 and output[0]>.75:
                errorArr.append(1)
            elif xo == 0 and output[0]<.25:
                errorArr.append(1)
            else:
                errorArr.append(0)
        
        genome.fitness = sum(errorArr)/len(errorArr)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    with open("winner.pickle", "wb") as f:
        pickle.dump(winner, f)


    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-34')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'trigger-config')
    run(config_path)