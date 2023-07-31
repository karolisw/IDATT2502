import random
import time
from sklearn.utils import shuffle
import matplotlib as plt

# Initialize target
target = 11001000 # 200

# Should be the same as the target
result = 0

# Initialized at runtime
population = []

mut_prob_count = 0
mut_count = 90


# To be called inside of the main function
def genetic_algorithm():
    global population
    global result
    counter = 0
    # While target not reached (result is set in selection algorithm when an individual has fitness == 0)
    while (result == 0):
        #print ("population before selection: ", population)
        # Fitness is measured through selection and the weakest are removed
        select() 
        if (result != 0):
            break
        if (counter == 400):
            print("Randomizing")
            randomize()
            counter = 0
        #print ("population after selection: ", population)
        # Combining the selected individuals "genes"
        new_gen = crossover()

        # Add two randomly generated bit strings
        new_gen.append(random.getrandbits((len(target))))
        new_gen.append(random.getrandbits((len(target))))

        # Apply the new generation to the population 
        population = new_gen[:] 
        counter += 1
       # if(counter == 10):
          #  result = 1

    # Loop broken when target found
    return result


# Fitness algorithm
def fitness(individual):
    individual_dec = to_decimal(individual)
    target_dec = to_decimal(target)
    #print("fitness: ", abs(individual_dec - target_dec))

    return -abs(individual_dec - target_dec) #Should be -abs

def compare(bit1, bit2):
    if (fitness(bit1) > fitness(bit2)):
        return 1
    elif (fitness(bit1) == fitness(bit2)):
        return 0
    else:
        return -1

def select():
    global population
    global result
    #strongest = []
    best_fitness = 255
    worst_fitness = 0
    total_fitness = 0

    #print("population before selection: ", len(population))

    # Sort population
    sorted(population, key=lambda x: -fitness(x))
    # Cut off the two worst
    population = population[:-2] 

    '''
    for i in range(len(population)):
        
        current_fitness = abs(fitness(population[i]))
        total_fitness += current_fitness

        # Algorithm termination base line
        if (current_fitness == 0):
            result = population[i]
            break 

        if (current_fitness < best_fitness):
            best_fitness = current_fitness
            strongest.insert(0, population[i])

        else:
            strongest.append(population[i])
    '''

    #TODO print("population after selection: ", len(population))

    # Print the best fitness for this generation
    print("Best fitness of the generation: ", best_fitness)
    print("Average fitness of the generation: ", total_fitness/(len(population)))

# Crossover algorithm
def crossover():
    global mut_prob_count
    global population
    new_generation = []

    shuffle(population)

    # Randomize the order of the population
    # random.shuffle(population)

    for i in range (0,len(population), 2):

        if (mut_prob_count > mut_count):
            new_generation = mutation(new_generation)
            mut_prob_count = 0

        # Because each loop used i and i + 1, we skip every odd-i loop
        #if (i % 2 != 0): 
         #   continue

        # Select two individuals from the population
        mom = population[i]
        dad = population[i + 1]

        # The mom and dad create a random number of children
       # amount_children = child_count()

        # To not empty out of individuals in population
        #if (len(population) < 10):
            #amount_children = 4

        #moms_genes = []
        #dads_genes = []
        
        '''
        if (fitness(mom) > fitness(dad)):
            # If mom has the best fitness, then dad cannot be longer than mom
            if (length(dad) > length(mom)):
                dad = str(dad)[:length(mom)] 
        elif (fitness(dad) > fitness(mom)):
            # If dad has the best fitness, then mom cannot be longer than dad
            if (length(mom) > length(dad)):
                mom = str(mom)[:length(dad)] # Slice off yet again...
        '''

        # Two elements in each list (split method defined in script; not python method)
        moms_genes = split(mom) 
        dads_genes = split(dad) 

        child1 = str(moms_genes[0]) + str(dads_genes[1])
        child2 = str(dads_genes[0]) + str(moms_genes[1])
      
        new_generation.append(child1)
        new_generation.append(child2)

        mut_prob_count += mut_prob_count + 2

        if (mut_prob_count > mut_count):
            new_generation = mutation(new_generation)
            mut_prob_count = 0

    # Return the new generation
    return new_generation


# Mutation algorithm (bit flip)
def mutation(new_generation):
    global mut_prob_count

    # The element from 'new_generation' that we will mutate
    element = new_generation[-1]
    element = bit_flip(element) 
    new_generation.insert(len(new_generation), element) # Switch in the new, mutated individual at the correct pos 
    
    # Reset probability count
    mut_prob_count = 0

    # Return the list, mutated individuals or not
    return new_generation

# Randomizes half of the population with different bit lengths
def randomize():
    global population
    new_bits = []
    iter_size = 0
    half_target_len = length(target)/2

    # Modifications to avoid floating point numbers
    if (len(population) % 2 != 0):
        iter_size = (len(population)-1)/4
    else:
        iter_size = len(population)/4
    if (length(target) % 2 != 0):
        half_target_len = (length(target)+1)/2  

    # Generates decimal number with a specified bit length (if converted to bits)
    for i in range (int(iter_size)):
        rand_nr = random.getrandbits(int(half_target_len))
        #while (rand_nr == 0 or rand_nr == 1):
        #    rand_nr = random.getrandbits(int(half_target_len))

        new_bits.append(get_bin(rand_nr))

    for i in range (int(iter_size)):
        rand_nr = random.getrandbits(length(target))
        #while (rand_nr == 0 or rand_nr == 1):
        #    rand_nr = random.getrandbits(int(half_target_len))

        new_bits.append(get_bin(rand_nr))
    
    shuffle(new_bits)   

    # Add to the population
    population[:len(new_bits)] = new_bits


# Support method for mutation alg
def bit_flip(individual):
    str_individual = list(str(individual)) # Strings are immutable but char-list is not

    # Access a random bit from the individual
    digits = length(individual) 
    random_bit = random.randint(1, digits) 
    bit = str_individual[random_bit - 1] # -1 for correct indexation
    global target

    if (bit == 1):
        str_individual[random_bit - 1] = '0' 
    else:
        str_individual[random_bit - 1] = '1'
    
    # Back to string before converting to int
    str_individual = ''.join(str_individual)

    return str_individual

# Supporting method for crossover algorithm that returns a bit sequence in the given length
def split(parent):
    split_parent = [] # Room for two elements
    half = 0

    # Specialcase of bit-representation of decimal numbers 0 and 1
    if (length(parent) == 1):
        # Both children will inherit the single bit from this parent
        split_parent.append(parent)
        split_parent.append(parent)
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
    # Two individuals can have 1-2 children 
    return random.randint(1,2)

# Supporting method for fitness algorithm
def to_decimal(bits):
    # From bits to decimal 
    return int(str(bits), 2)

# Decimal to binary number lambda function
get_bin = lambda x: format(x, 'b')

# Support method that finds the length of a number
def length(number):
    return len(str(number)) 

def init_population():
    global population
    new_population = []

    # Randomize the population to a count of 10
    for i in range (10):   
        individual = random.getrandbits(len(target)) 
        new_population.append(get_bin(individual))
    population = new_population[:]


#################################################################################################################

'''
# Randomize the population
for i in range (50):   
    rand_length = random.randint(1,8)
    individual = random.getrandbits(rand_length) # Might have to insert while loop to remove 0's
    #while(individual == 0):
    #    individual = random.getrandbits(rand_length)
    population.append(get_bin(individual))

bit_res = genetic_algorithm()
print("The result was: \n", bit_res)
if(bit_res == target):
    print("Which is equal to the target!")
else:
    print("Which is not equal to the target, because the target was: \n", target)
'''

# For 8 iterations, create a random bit-sequence (with iteratively larger bit length -> +1) 
y_axis = []
for i in range (10): 
    bit_sequence = random.getrandbits(i + 1)
    binary = get_bin(bit_sequence)

    # To be able to predict the length based on the loop
    while (length(binary) != (i + 1)):
        bit_sequence = random.getrandbits(i + 1)
        binary = get_bin(bit_sequence)

    # Assign the target
    target = binary

    # Randomize the population
    init_population()

    runs = 3
    # Total time for all n runs
    tot_time = 0
    # For 3 iterations 
    for i in range (runs):
        # Record the time the algorithm runs each time
        print("Target is: ", target)
        start_time = time.time()
        res = genetic_algorithm()
        end_time = time.time()
        tot_time += end_time-start_time
        
        print("Elapsed time: ", end_time - start_time)
        print("Result was: ", res)
        result = 0 # Reset the result variable
        init_population() # New population
        break
    
    avg_time = tot_time/runs
    y_axis.append(avg_time)

# Line diagram    
plt.plot(range(10), y_axis)
plt.show()
# Plot bit length (bit sequence) to the average time using a line diagram
# x-axis = bit length
# y-axis = avg_time