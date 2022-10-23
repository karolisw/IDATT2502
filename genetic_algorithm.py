import random

# Initialize target
target = 11001000 # 200

# Should be the same as the target
result = 0

# Initialize random population and put them into a list
population = [1110011, 1100, 11, 11011, 1100101,               # 115, 12, 3, 27, 101,
              11000111, 11111010, 11010011, 10111000, 1001000] # 199, 250, 211, 184, 72

mut_prob_count = 0

# TODO Create a class with a main method that runs this function only 
# TODO the goal is to be able to use the function in a loop and see the results on average

# To be called inside of the main function
def genetic_algorithm():
    # While target not reached (result is set in selection algorithm when an individual has fitness == 0)
    while (result == 0):
        # Fitness is measured throgh selection and the two weakest individuals are removed
        select()  # TODO could randomize how many are removed... (or use a specific percentage)
        # Combining the selected individuals "genes"
        new_gen = crossover()
        # Mutate their genes (possibly)
        new_gen = mutation(new_gen)
        # Apply the new generation to the population 
        population.clear() #TODO could be unnecessary and could be that the parentheses is not necessary hmmm.....
        population = new_gen[:] 

    # Loop broken when target found
    return result


# Fitness algorithm
def fitness(individual):
    individual_dec = to_decimal(individual)
    target_dec = to_decimal(target)

    return -abs(individual_dec - target_dec)

# Selection algorithm that removes the two weakest individuals and sorts population by fitness
def select():
    global population
    strongest = []
    # Add the first individual to the list of strongest individuals
    strongest.append(population[0])
    lowest_fitness = fitness(strongest[0])
    for i in range(len(population)):
        if (i == 0):
            continue
        # If decimal version of i has a lower fitness than the current strongest
        current_fitness = fitness(population[i]) 
        # Algorithm termination base line
        if (current_fitness == 0):
            # This bit-value is the final result
            result = population[i]
            break # No need to continue the loop
        if (current_fitness < lowest_fitness):
            # Set new lowest fitness
            lowest_fitness = current_fitness
            # Append first
            strongest.insert(0, population[i])
        else:
            # Append at end of list
            strongest.append(population[i])
    
    # Cut of the two weakest (the two first) elements and assign to the population
    population.clear()
    population = strongest[2:]

# Crossover algorithm
def crossover():
    new_generation = []

    # Randomize the order of the population
    random.shuffle(population)


    for i in range (len(population)):

        # Select two individuals from the population
        mom = population[i]
        dad = population[i + 1]
        
        # The mom and dad create a random number of children
        amount_children = child_count()
        moms_genes = []
        dads_genes = []
        
        if (fitness(mom) > fitness(dad)):
            # If mom has the best fitness, then dad cannot be longer than mom
            if (length(dad) > length(mom)):
                dad = dad[:length(mom)] 
        elif (fitness(dad) > fitness(mom)):
            # If dad has the best fitness, then mom cannot be longer than dad
            if (length(mom) > length(dad)):
                mom = mom[:length(dad)] # Slice off yet again...
            
        # Two elements in each list (split method defined in script; not python method)
        moms_genes = split(mom) 
        dads_genes = split(dad) 


        if (amount_children == 1):
            child = str(min(dads_genes)) + str(max(moms_genes)) 
            new_generation.append(int(child))
            mut_prob_count = mut_prob_count + 1
        elif (amount_children == 2):
            child1 = str(min(moms_genes)) + str(max(dads_genes)) 
            child2 = str(min(dads_genes)) + str(max(moms_genes))
            new_generation.append(int(child1))
            new_generation.append(int(child2))
            mut_prob_count = mut_prob_count + 2
        elif (amount_children == 3):
            child1 = str(min(moms_genes)) + str(max(dads_genes))
            child2 = str(min(dads_genes)) + str(max(moms_genes))
            child3 = str(min(dads_genes)) + str(min(moms_genes))
            new_generation.append(int(child1))
            new_generation.append(int(child2))
            new_generation.append(int(child3))
            mut_prob_count = mut_prob_count + 3
        else:
            child1 = str(min(moms_genes)) + str(max(dads_genes))
            child2 = str(min(dads_genes)) + str(max(moms_genes))
            child3 = str(min(dads_genes)) + str(min(moms_genes))
            child4 = str(max(dads_genes)) + str(max(moms_genes))
            new_generation.append(int(child1))
            new_generation.append(int(child2))
            new_generation.append(int(child3))
            new_generation.append(int(child4))
            mut_prob_count = mut_prob_count + 4

    # Return the new generation
    return new_generation

# Mutation algorithm (bit flip)
def mutation(new_generation):

    # 1% Chance of mutation
    if (mut_prob_count >= 100):
        # The element from 'new_generation' that we will mutate
        element = 0
        children_since = mut_prob_count - 100
        if (children_since == 0): # mut_prob_count = 100
            element = new_generation[-1]
            element = bit_flip(element) 
            new_generation.insert(len(new_generation), element) # Switch in the new, mutated individual at the correct pos 
        else: # mut_prob_count > 100
            element = new_generation[-children_since] # Mutate the correct individual
            element = bit_flip(element) 
            new_generation.insert(len(new_generation) - children_since, element) 

        # Reset probability count
        mut_prob_count = 0

    # Return the list, mutated individuals or not
    return new_generation

# Support method for mutation alg
def bit_flip(individual):
    length = len(individual) 
    # Access a random bit from the individual
    random_bit = random.randint(1, length) 
    str_individual = str(individual)
    bit = str_individual[random_bit - 1] # -1 for correct indexation

    if (bit == 1):
        str_individual[random_bit - 1] = 0
    else:
        str_individual[random_bit - 1] = 1
    
    return int(str_individual)

# Supporting method for crossover algorithm that returns a bit sequence in the given length
def split(parent):
    split_parent = [] # Room for two elements
    half = 0

    # Specialcase of bit-representation of decimal numbers 0 and 1
    if (length(parent) == 1):
        # Both children will inherit the single bit from this parent
        split_parent == [parent, parent] 
        return split_parent

    # If even number > 1
    elif (length(parent) % 2 == 0):
        # If parent and child is the same length
        half = (length(parent))/2

    # If odd number >= 3
    else:
        half = (length(parent) - 1)/2 

    # Both children get half of the binary number
    split_parent.append(str(parent)[:int(half)]) 
    split_parent.append(str(parent)[int(half):])
    return split_parent
 
#TODO TypeError: slice indices must be integers or None or have an __index__ method
# Supporting method for crossover algorithm
def child_count():
    # Two individuals can have 1-4 children 
    return random.randint(1,4)

# Supporting method for fitness algorithm
def to_decimal(bits):
    # From bits to decimal 
    return int(str(bits), 2)

# Support method that finds the length of a number
def length(number):
    return len(str(number)) 



#################################################################################################################

bit_res = genetic_algorithm()
print("The result was: \n", bit_res)
if(bit_res == target):
    print("Which is equal to the target!")
else:
    print("Which is not equal to the target, because the target was: \n", target)