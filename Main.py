import copy 
# Number of the weights we are looking to optimize.
num_weights = 4 # numer of detector
import numpy as np
sol_per_pop = 8
# Defining the population size.

pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#in sort arrau of 1X8


#Creating the initial population.

new_population = np.random.uniform(low=3, high=9, size=pop_size)
new_population = [[ 2**int(j) for j in i]for i in new_population]
import random
for i in range(len(new_population)):
    new_population[i][0] = random.randint(1,3)
    new_population[i][3] = 2**random.randint(3,10)
new_population = np.array(new_population) 
print(new_population)


#import ga
num_generations = 8

num_parents_mating = 4
for generation in range(num_generations):
    print("Generation = "+ str(generation+1))
     # Measuring the fitness of each chromosome in the population.
     #ga.load_images()
    if generation == 0:
         fitness = cal_pop_fitness(new_population)
         fitness_copy = sorted(copy.deepcopy(fitness))
         top = fitness[0].copy()
    else:
         fitness[:4] =  fitness_copy[:4]
         fitness[4:] = cal_pop_fitness(new_population[4:,:])
         top = fitness[0].copy()
    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness, 
                                       num_parents_mating)
    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents,
                                        offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    # Adding some variations to the offsrping using mutation.
    offspring_mutation = ga.mutation(offspring_crossover)
    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

print(new_population[0]+top)
