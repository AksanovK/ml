import random

GEN_SIZE = 20
MAX_FITNESS = 10000


def equilibrium(chromosome):
    return (chromosome[0] + 2 * chromosome[1] - 3 * chromosome[2] + chromosome[3] +
            4 * chromosome[4] + chromosome[5]) - 5


def fitness(chromosome):
    ans = equilibrium(chromosome)
    if ans == 0:
        return MAX_FITNESS
    return abs(1 / ans)


def selection(population, fit_chromosomes):
    fitness_score = []
    for chromosome in population:
        individual_fitness = fitness(chromosome)
        fitness_score.append(individual_fitness)
    score_card = list(zip(fitness_score, population))

    for individual in score_card:
        if individual[0] == MAX_FITNESS:
            if individual[1] not in fit_chromosomes:
                fit_chromosomes.append(individual[1])

    score_card = score_card[:GEN_SIZE]
    score, population = zip(*score_card)

    return list(population)


def mutation(population):
    mutated_chromosomes = []
    for chromosome in population:
        mutation_site = random.randint(0, 5)
        chromosome[mutation_site] = random.randint(1, 9)
        mutated_chromosomes.append(chromosome)
    return mutated_chromosomes


def crossover(population):
    random.shuffle(population)
    population_half = GEN_SIZE // 2
    fatherChromosome = population[:population_half]
    motherChromosome = population[population_half:]
    children = []

    for i in range(len(fatherChromosome)):
        chromo_length = random.randint(1, 5)
        fatherFragments = [fatherChromosome[i][:chromo_length], fatherChromosome[i][chromo_length:]]
        motherFragments = [motherChromosome[i][:chromo_length], motherChromosome[i][chromo_length:]]
        firstChild = fatherFragments[0] + motherFragments[1]
        secondChild = motherFragments[0] + fatherFragments[1]
        children.append(firstChild)
        children.append(secondChild)
    return children


def solve(generations):
    population = [[random.randint(1, 9) for i in range(6)] for j in range(GEN_SIZE)]
    fit_chromosomes = []
    for generation in range(generations):
        population = selection(population, fit_chromosomes)
        crossover_children = crossover(population)
        population = crossover_children
        mutated_population = mutation(population)
        population = mutated_population
    return fit_chromosomes


solutions = solve(200)

for solution in solutions:
    print(equilibrium(solution))
print(solutions)