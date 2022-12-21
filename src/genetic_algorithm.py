# genetic algorithm search of the one max optimization problem
from numpy.random import randint
from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt
from deeplearning import DeepLearningModel, DLModelHelper

dl_helper = None

def accuracy_score(spec, path):
    print("[Info][accuracy_score] > Evaluating model accuracy of spec: {}".format(spec))
    global dl_helper
    dlm = DeepLearningModel(path, dl_helper)
    dlm.build(spec)
    acc = dlm.evaluate()
    return acc * 100  # return the accuracy in percent


def dummy_fitness(x):  # dummy fitness score. The higher the better.
    return sum(x)


# tournament selection
def tournament_selection(pop, scores, k=2):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k):
        # check if better (e.g. perform a tournament)
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1: list, p2: list, r_cross, n_point: int=1) -> list:
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()

    # check for recombination
    if rand() < r_cross:
        for _ in range(n_point):  # n_point crossover is simply single point crossover done n times.
            # select crossover point that is not on the end of the string
            pt = randint(1, len(p1)-2)

            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
    
    return [c1, c2]


# mutation operator
def do_mutation(bitstring: list, r_mut_nlayers: float, r_mut_layers: float) -> None:  # this function operates inplace
    if rand() < r_mut_nlayers:  # this part mutates only the number of layers
        bitstring[0] += np.cos(np.pi*randint(0,high=2))  # this adds a +1 or -1
        if bitstring[0] == 9 or bitstring[0] == 0:
            bitstring[0] = randint(1, high=9)  # if we pass the max or min value, just get a new random value
    
    if rand() < r_mut_layers:  # this part mutates the layers themselves
        mutated_layer_idx = randint(1, high=9)  # select a layer to be mutated (1-8)
        n_neurons_idx = 2*mutated_layer_idx - 1
        activationf_idx = n_neurons_idx + 1

        if rand() < 0.5:
            m = int(np.log2(bitstring[n_neurons_idx])) + np.cos(np.pi*randint(0, high=2))  # this adds a +1 or -1 to the exponent (base 2)
            if m == 11 or m == -1:  # catch if we go out of bounds
                m = randint(0, high=11)
            bitstring[n_neurons_idx] = 2**m

        if rand() < 0.5:
            bitstring[activationf_idx] = randint(0, high=10)  # for the activation function, get a new value. 


def create_initial_population(n_pop: int) -> list:
    init_population = list()

    for _ in range(n_pop):
        individuum = list()
        n_active_layers = randint(1,high=9)  # generate values between 1-8
        individuum.append(n_active_layers)

        for _ in range(8):
            n_neurons = 2**randint(0, high=11)  # generate values between 1-1024
            activationf_type = randint(1, high=10)  # generate values between 1-9
            individuum += [n_neurons, activationf_type]

        init_population.append(individuum)
    
    print("[Info][create_initial_population] > Created initial population: {}".format(init_population))
    return init_population


# genetic algorithm
def genetic_algorithm(objective, n_iter: int, n_pop: int, r_cross, r_mut_nlayers, r_mut_layers, mutation: bool=True, n_point: int=1, dataset_path: str=None):
    assert n_pop%2==0, "Population number should be even number!"
    fitness_tracker = list()
    
    # initial population of random bitstring
    pop = create_initial_population(n_pop)

    # keep track of best solution
    if dataset_path is not None:
        best, best_eval = 0, objective(pop[0], dataset_path)
    else:
        best, best_eval = 0, objective(pop[0])

    # enumerate generations
    for gen in range(n_iter):
        print("[Info][genetic_algorithm] > Starting generation: {}".format(gen))

        # evaluate all candidates in the population
        if dataset_path is not None:
            scores = [objective(c, dataset_path) for c in pop]
        else:
            scores = [objective(c) for c in pop]  # in the case we are running the demo

        # check for new best solution
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print("[Info][genetic_algorithm] > Generation: %d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
        fitness_tracker.append(best_eval)

        # select parents
        selected = [tournament_selection(pop, scores) for _ in range(n_pop)]

        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]

            # crossover and mutation
            for c in crossover(p1, p2, r_cross, n_point):
                # mutation
                if mutation:
                    do_mutation(c, r_mut_nlayers, r_mut_layers)
                
                # store for next generation
                children.append(c)

        # replace population
        pop = children
    return best, best_eval, fitness_tracker


def demo():
    # define the total iterations
    n_iter = 100
    # bits
    n_bits = 50
    # define the population size
    n_pop = 200
    # crossover rate
    r_cross = 1
    # mutation rate
    r_mut_nlayers = 0.2  # this controls the probability of mutating only the number of layers
    r_mut_layers = 0.2  # this controls the probability of mutating only the layers individually 
    # perform the genetic algorithm search
    fitness_global = list()
    n_runs = 3
    for r in range(n_runs):
        best, score, fitness_story = genetic_algorithm(dummy_fitness, n_iter, n_pop, r_cross, r_mut_nlayers, r_mut_layers, mutation=False, n_point=3)
        #print('Done!')
        #print('f(%s) = %f' % (best, score))
        fitness_global.append(np.array(fitness_story))

    y = np.abs(np.average(fitness_global, axis=0))
    plt.plot(y)
    plt.xlabel("Generations")
    plt.ylabel("Averaged Best Fitness Scores for 10 Runs")
    plt.title("Average Best Fitness over 100 Generations")
    plt.show()


def main():
    # define the total iterations
    n_iter = 14

    # define the population size
    n_pop = 10

    # crossover rate
    r_cross = 1

    # mutation rate
    r_mut_nlayers = 0.2  # this controls the probability of mutating only the number of layers
    r_mut_layers = 0.2  # this controls the probability of mutating only the layers individually 

    # configure deep learning model stuff
    dataset_path = "covid.csv"

    # perform the genetic algorithm search
    global dl_helper
    dl_helper = DLModelHelper(dataset_path)
    best, score, fitness_story = genetic_algorithm(accuracy_score, n_iter, n_pop, r_cross, r_mut_nlayers, r_mut_layers, mutation=False, n_point=3, dataset_path=dataset_path)
    print('Done!')
    print("Best: ", best)
    print("Score: ", score)
    print(fitness_story)


if __name__ == "__main__":
    #demo()  # NOTE: use this for trying out the GA algorithm
    main()

    