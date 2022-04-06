"""
Name: Ibrahim Hashmi
  ID: 6352926
 IDE: VSCode
"""
from statistics import mean
import math
import random
import csv






file = open('ulysses22.txt')# SWITCH BETWEEN TEXT FILES HERE
n = file.readline()
num_cities = int(n)

#creates list of lists containing all city data
city_data = []
cities = file.readlines()[0:-1]
for city in cities:
  city_data.append(city.split())
# print(city_data)

# creates list of cities according to how they appear in the text file
list_of_cities = []
for i in range(0,len(city_data)):
  list_of_cities.append(int(city_data[i][0]))

#initialize population
def initialize_pop(popSize):
  initial_pop = []
  c = 0
  while c < popSize:
    chromosome = random.sample(list_of_cities,k=num_cities)
    initial_pop.append(chromosome)
    c+=1
  return initial_pop

# Calculate Euclidean Distance
def euclidean_dist(x1, x2, y1, y2):

  e_distance = math.sqrt((x1-x2)**2+(y1-y2)**2)
  return e_distance

# Takes in a chromosome and returns it's fitness score
def evaluate(chromosome):
  fitness = 0.0
  i = 0
  # print(chromosome)
  for i in range(len(city_data)-1):
    # print(i)
    # print(city_data[chromosome[i]])

    fitness += euclidean_dist(float(city_data[chromosome[i]-1][1]), float(city_data[chromosome[i+1]-1][1]), float(city_data[chromosome[i]-1][2]), float(city_data[chromosome[i+1]-1][2]))


  # first_city = chromosome[0]
  # last_city = chromosome[-1]
  fitness += euclidean_dist(float(city_data[chromosome[0]-1][1]), float(city_data[chromosome[-1]-1][1]), float(city_data[chromosome[0]-1][2]), float(city_data[chromosome[-1]-1][2]))

  return fitness


# Takes a population and selects k random chromosomes from within it, then returns the one with the highest fitness
def tournament(population,k_val):
  selections = random.choices(population, k=k_val)

  i=0
  distances = []
  while i<k_val:
    
    distances.append(evaluate(selections[i]))
    i+=1

  shortest_dist = distances[0]
  best_fit = selections[0]
  for j in (range(k_val)):
    if distances[j] < shortest_dist:
        shortest_dist = distances[j]
        best_fit = selections[j]

  return best_fit


# Takes two parent chromosomes and sets a random crossover point in them, then creates and returns two children by combining all genetic information on either side of crossover point
def crossover_1pt(parentA,parentB):
  point = random.randint(1, len(parentA)-1) #pick a random point in the chromosome

  childA = parentA[0:point] + parentB[point:] #take everything left of the crossover point in parent1 and everything right of it in parent 2 to make child 1
  childA = fixer(list_of_cities,childA)
  childB = parentB[0:point] + parentA[point:]#take everything right of the crossover point in parent1 and everything left of it in parent 2 to make child 2
  childB = fixer(list_of_cities,childB)

  return childA, childB

# Fixes a chromosome by replacing all duplicates it may have with it's missing cities, ensuring every city appears in it exactly once
def fixer(cities,chromosome):
  i=0
  missing_cities = []
  for city in cities:

    # replace all duplicates with "-"
    while chromosome.count(city) > 1: 
      chromosome[chromosome.index(city)] = "-"
    
    # makes a list of all cities missing from the chromosome
    if chromosome.count(city) == 0: 
      missing_cities.append(city)

  #Replace all instances of "-" in the chromosome with the missing cities 
  for city in missing_cities:
    chromosome[chromosome.index("-")] = city

  return chromosome


# Takes two parents and generates two children using a bitmask to replace indexes in each child with values from both parents
def uox(p1,p2):
  c1=[]
  c2=[]
  mask = []
  for city in p1:
    c1.append("-")
    c2.append("-")
    mask.append("-")

  #generate a randomized mask
  # i=0
  # while i<(len(p1)-1):
  #   mask[i] = random.randint(0,1)
  #   i+=1

  mask = [1,0,0,1,1,0,1]
  
  for j in range(len(mask)):
    if mask[j]==1:
      c1[j] = p1[j]
      c2[j] = p2[j]
      
  #if city in p2 is not in c1, replace first appearance of "-" in c1 with the city
  for city in p2:
    if city not in c1:
      c1[c1.index("-")] = city 

  #if city in p1 is not in c2, replace first appearance of "-" in c2 with the city
  for city in p1:
    if city not in c2:
      c2[c2.index("-")] = city
  
  return c1,c2

#select a random city in the chromosome and insert it in a random place within the chromosome, removing it from it's original place
def mutate(chromosome):
  current_spot = random.randint(0,(len(chromosome)-1))
  new_spot = random.randint(0,(len(chromosome)-1))

  while current_spot == new_spot: #Prevents from removing and inserting a city into the same space it was already in originally
      new_spot = random.randint(0,(len(chromosome)-1))

  random_city = chromosome[current_spot]
  chromosome.remove(random_city) #remove from original place in chromosome
  chromosome.insert(new_spot,random_city) #insert to new random place in chromosome
  return chromosome






# return chromosome with highest score
def elitism1(population):
  fitnesses = []
  highest = []
  chosen = []
  highest_fitness = math.inf
  for chromosome in population:
    fitnesses.append(evaluate(chromosome))
    if evaluate(chromosome)<highest_fitness:
      highest_fitness = evaluate(chromosome)

  # print(population[fitnesses.index(highest_fitness)])
  elite = population[fitnesses.index(highest_fitness)]
  return elite

# returns list of highest fitness % chromosomes
def elitism2(population,rate):
  i=0
  j=0
  # amount = math.floor(len(population)/2)
  amount = math.floor(len(population)/rate) # 10% of population size
  fitnesses = []
  highscores = []
  chosen = []
  
  for chromosome in population:
    fitnesses.append(evaluate(chromosome))

  fitnesses.sort(reverse=True)
  # print(fitnesses)
  while i<amount:
    highscores.append(fitnesses[i])
    i+=1
  # print(highscores)
  while j<amount:
    chosen.append(population[fitnesses.index(highscores[j])])
    j+=1

  return chosen


#function to run Genetic algorithm
def run_ga(generations,kValue,seed,elitismPercent,crossoverRate,mutationRate,popSize):
  
  random.seed(seed)
  crossover_rate = float(crossoverRate)/100 # probability of chromosome undergoing crossover
  mutation_rate = float(mutationRate)/100 # probability of chromosome being mutated  
  total_gens = generations

  num_of_gens = 1

  best_fits = []
  best_chromosomes = []
  i_pop = initialize_pop(popSize) #set initial random population
  best_solution_fitness = math.inf
  best_solution_chromosome  = []
  

  print("GA Parameters: Generation No., K-Value, Random Number Seed, Elitism %, Crossover Rate, Mutation Rate, Population Size")
  print("Random Seed: "+str(seed) + ", Population Size: "+str(popSize) + ", Crossover Rate: "+str(crossoverRate)+"%, "+"Mutation Rate: "+str(mutationRate)+"% " + "Elitism: "+str(elitismPercent))

  
  while num_of_gens<=(total_gens):
    
    new_pop = []



    gen_best_chromosome = [] # best chromosome in current gen
    best_fitness = math.inf # best fitness in current gen (initialized to infinite)
    gen_fits = []
    gen_avg_fit = math.inf
    
    
    new_pop.append(elitism1(i_pop)) # automatically add the chromosome with the highest fitness score

    elitest = elitism2(i_pop,elitismPercent) # automatically add chromosomes within the top 5% highest scores
    for chromosome in elitest:
      new_pop.append(chromosome)


    while len(new_pop)<len(i_pop) :  # FILL UP EACH POPULATION

      parent1 = tournament(i_pop,kValue)
      parent2 = tournament(i_pop,kValue)

      crossover_chance = random.uniform(0.0,1.0)
      mutation_chance_1 = random.uniform(0.0,1.0) # chance of child 1 being mutated
      mutation_chance_2 = random.uniform(0.0,1.0) # chance of child 2 being mutated

      if crossover_chance < crossover_rate:
        child1,child2 = crossover_1pt(parent1,parent2)  #SWITCH BETWEEN CROSSOVER METHODS HERE
        # child1,child2 = uox(parent1,parent2)
        if mutation_chance_1 < mutation_rate:
          child1 = mutate(child1)
        if mutation_chance_2 < mutation_rate:
          child2 = mutate(child2)

        new_pop.append(child1)
        new_pop.append(child2)

      else:

        if mutation_chance_1 < mutation_rate:
          parent1 = mutate(parent1)
        if mutation_chance_2 < mutation_rate:
          parent2 = mutate(parent2)

        new_pop.append(parent1)
        new_pop.append(parent2)

    #if new population exceeds required size, delete chromosomes from the end of list until required size is met
    while len(new_pop) > len(i_pop):  
      del new_pop[-1]

    # make list of all fitness scores in the new population
    for chromosome in new_pop:
      # print(chromosome)
      gen_fits.append(evaluate(chromosome))

    gen_avg_fit = mean(gen_fits)

    # find the current gen's best fitness (lowest euclidean distance) and it's corresponding chromosome (path of cities with shortest distance) 
    for chromosome in new_pop:
      if evaluate(chromosome) < best_fitness:
        best_fitness = evaluate(chromosome)
        gen_best_chromosome = chromosome

    best_fits.append(best_fitness)
    best_chromosomes.append(gen_best_chromosome)


    # print("Generation #: "+ str(num_of_gens)+"  Average Fitness: "+str(gen_avg_fit)+"  Best Solution Fitness: "+str(best_fitness)+"  Best Solution Chromosome: "+str(gen_best_chromosome))
    print("Generation #: "+ str(num_of_gens)+"  Average Population Fitness: "+str(gen_avg_fit)+"  Best Population Fitness: "+str(best_fitness))

    i_pop = new_pop
    # print("updated init_pop: "+ str(i_pop))
    num_of_gens+=1

    


    # END OF OVER-ARCHING WHILE LOOP

    # PART C)
    
  # find the best overall fitness (shortest distance) and it's corresponding chromosome ()
  for a in range(len(best_fits)):
    if best_fits[a] < best_solution_fitness:
      best_solution_fitness = best_fits[a]
      best_solution_chromosome = best_chromosomes[a]
      # best_gen = a+1


p1=[1,2,3,4,5,6,7]
p2 = [6,3,4,2,7,1,5]
print(uox(p1,p2))

  
  # print("Best Solution Fitness: " + str(best_solution_fitness))
  # print("Best Solution Chromosome: " + str(best_solution_chromosome))





print("GA Parameters: Generation No., K-Value, Random Number Seed, Elitism %, Crossover Rate, Mutation Rate, Population Size")

# generations,kValue,seed,elitismRate,crossoverRate,mutationRate,popSize

# 1. Crossover Rate: 100%,  Mutation Rate: 0%
# print("RUN 1:")
# run_ga(50,2,0,7,100,0,1000) #Run 1
# print("\nRUN 2")
# run_ga(50,2,1,7,100,0,1000) #Run 2
# print("\nRUN 3")
# run_ga(50,2,2,7,100,0,1000) #Run 3
# print("\nRUN 4")
# run_ga(50,2,3,7,100,0,1000) #Run 4
# print("\nRUN 5")
# run_ga(50,2,4,7,100,0,1000) #Run 5
# print("\n\n")

# 2. Crossover Rate: 100%, Mutation Rate: 10%
# print("\nRUN 1")
# run_ga(50,2,0,7,100,10,1000)
# print("\nRUN 2")
# run_ga(50,2,1,7,100,10,1000)
# print("\nRUN 3")
# run_ga(50,2,2,7,100,10,1000)
# print("\nRUN 4")
# run_ga(50,2,3,7,100,10,1000)
# print("\nRUN 5")
# run_ga(50,2,4,7,100,10,1000)
# print("\n\n")

# # 3. Crossover Rate: 90%, Mutation Rate: 0%
# print("\nRUN 1")
# run_ga(50,2,0,7,90,0,1000)
# print("\nRUN 2")
# run_ga(50,2,1,7,90,0,1000)
# print("\nRUN 3")
# run_ga(50,2,2,7,90,0,1000)
# print("\nRUN 4")
# run_ga(50,2,3,7,90,0,1000)
# print("\nRUN 5")
# run_ga(50,2,4,7,90,0,1000)
# print("\n\n")

# # # 4. Crossover Rate: 90%, Mutation Rate: 10%
# print("\nRUN 1")
# run_ga(50,2,0,7,90,10,1000)
# print("\nRUN 2")
# run_ga(50,2,1,7,90,10,1000)
# print("\nRUN 3")
# run_ga(50,2,2,7,90,10,1000)
# print("\nRUN 4")
# run_ga(50,2,3,7,90,10,1000)
# print("\nRUN 5")
# run_ga(50,2,4,7,90,10,1000)
# print("\n\n")

# # 5. Crossover Rate: 92%, Mutation Rate: 8% (Custom parameter settings)
# print("\nRUN 1")
# run_ga(50,2,0,7,92,8,1000)
# print("\nRUN 2")
# run_ga(60,2,1,7,92,8,300)
# print("\nRUN 3")
# run_ga(50,2,2,7,92,8,1000)
# print("\nRUN 4")
# run_ga(50,2,3,7,92,8,1000)
# print("\nRUN 5")
# run_ga(50,2,4,7,92,8,1000)
# print("\n\n")







  