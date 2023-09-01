import sys
import random
import os
from collections import Counter
import threading
import time
import multiprocessing


ALG_NORMAL = "normal"
ALG_DARWINIAN = "darwinian"
ALG_LAMIRICK = "lamirick"

INCORRECT_ARGS_MESSAGE = "Usage: python decode.py FILE 1/2/3"

CSV_FILE_FIRST_ROW = "algorithm, mutation chance, population size, survivor fraction, search count, iterations_number, best fitness, total calls to fitness\n"

FILE_NAME_RESULTS ="results.csv"
FILE_NAME_PLAIN = "plain.txt"
FILE_NAME_PERM = "perm.txt"
FILE_NAME_DICT = "dict.txt"
FILE_NAME_LETTER_FREQ = "Letter_Freq.txt"
FILE_NAME_TWO_LETTER_FREQ = "Letter2_Freq.txt"
ALL_LETTERS = set(chr(ord('a') + i) for i in range(26))

WORD_SCORE_LAMBDA = 10
LETTER_SCORE_LAMBDA = 0.5
TWO_LETTER_SCORE_LAMBDA = 1

MUTATIONS_CHANCE = 0.5
MAX_ITERATIONS = 10000
POPULATION_SIZE = 2000
SURVIVOR_FRACTION = 0.1
SEARCH_COUNT = 2
MAX_NO_IMPROVEMENTS_COUNT = 20
DEFAULT_STEPS_PER_PRINT = 10000 # usually 10







def normal_genetic(text, letter_freq, two_letter_freq, words_set, try_export = True, population_size=POPULATION_SIZE, survivor_fraction=SURVIVOR_FRACTION, mutation_chance=MUTATIONS_CHANCE):
    last_iteration, fitness_counter, best_score, no_improvement_counter = 0, 0, 0, 0
    fitnesses = [0 for _ in range(population_size)]
    population = initialize_population(text,population_size)
    for i in range(MAX_ITERATIONS):
        fitnesses, times_called_fitness = calc_fitnesses(text, population, letter_freq, two_letter_freq, words_set)
        fitness_counter += times_called_fitness
        #print_best_fitness(i, fitnesses)
        population = selection(population, fitnesses, population_size, survivor_fraction)
        population = repopulate(population, population_size, mutation_chance)
        #print_decoded(text, population[0], i)
        is_stuck, no_improvement_counter, best_score = calc_stuck(fitnesses, best_score, no_improvement_counter, i)
        if (is_stuck):
            last_iteration = i
            break
    if try_export:
        export(text, population, letter_freq, two_letter_freq, words_set, fitness_counter)
    return last_iteration, best_score, fitness_counter

def darwinian_genetic(text, letter_freq, two_letter_freq, words_set, try_export = True, population_size=POPULATION_SIZE, survivor_fraction=SURVIVOR_FRACTION, mutation_chance=MUTATIONS_CHANCE,search_count=SEARCH_COUNT):
    last_iteration, fitness_counter, best_score, no_improvement_counter = 0, 0, 0, 0
    fitnesses = [0 for _ in range(population_size)]
    population = initialize_population(text,population_size)
    for i in range(MAX_ITERATIONS):
        after_improvement, times_called_fitness  = improve_all_gens(text, population, letter_freq, two_letter_freq, words_set,fitnesses,search_count)
        fitness_counter += times_called_fitness
        fitnesses, times_called_fitness = calc_fitnesses(text, after_improvement, letter_freq, two_letter_freq, words_set)
        fitness_counter += times_called_fitness
        #print_best_fitness(i, fitnesses)
        population = selection(population, fitnesses, population_size, survivor_fraction)
        population = repopulate(population, population_size, mutation_chance)
        #print_decoded(text, population[0], i)
        is_stuck, no_improvement_counter, best_score = calc_stuck(fitnesses, best_score, no_improvement_counter, i)
        if (is_stuck):
            last_iteration = i
            break
    if try_export:
        export(text, population, letter_freq, two_letter_freq, words_set, fitness_counter)
    return last_iteration, best_score, fitness_counter


def lamirick_genetic(text, letter_freq, two_letter_freq, words_set, try_export = True, population_size=POPULATION_SIZE, survivor_fraction=SURVIVOR_FRACTION, mutation_chance=MUTATIONS_CHANCE, search_count=SEARCH_COUNT):
    last_iteration, fitness_counter, best_score, no_improvement_counter = 0, 0, 0, 0
    fitnesses = [0 for i in range(population_size)]
    population = initialize_population(text,population_size)
    for i in range(MAX_ITERATIONS):
        population, times_called_fitness = improve_all_gens(text, population, letter_freq, two_letter_freq, words_set,fitnesses, search_count)
        fitness_counter += times_called_fitness
        fitnesses, times_called_fitness = calc_fitnesses(text, population, letter_freq, two_letter_freq, words_set)
        fitness_counter += times_called_fitness
        #print_best_fitness(i, fitnesses)
        population = selection(population, fitnesses, population_size, survivor_fraction)
        population = repopulate(population, population_size, mutation_chance)
        #print_decoded(text, population[0], i)
        is_stuck, no_improvement_counter, best_score = calc_stuck(fitnesses, best_score, no_improvement_counter, i)
        if (is_stuck):
            last_iteration = i
            break
    if try_export:
        export(text, population, letter_freq, two_letter_freq, words_set, fitness_counter)
    return last_iteration, best_score, fitness_counter

def calc_stuck(fitnesses, best_score, no_improvement_counter, i):
    if i == MAX_ITERATIONS - 1:
        return False, no_improvement_counter, best_score
    if max(fitnesses) > best_score:
        return False, 0, max(fitnesses)
    else:
        if no_improvement_counter > MAX_NO_IMPROVEMENTS_COUNT:
            return True, no_improvement_counter+1, best_score
        else:
            return False, no_improvement_counter+1, best_score

def improve_all_gens(text, population, letter_freq, two_letter_freq, words_set, fitnesses, search_count=SEARCH_COUNT):
    return [improve_gen(text, gen, letter_freq, two_letter_freq, words_set, search_count, fitness) for gen, fitness in zip(population, fitnesses)], len(population)*search_count


def improve_gen(text, gen, letter_freq, two_letter_freq, words_set, search_count, current_fitness):
    best_gen = gen.copy()
    for _ in range(search_count):
        new_gen = mutate(best_gen)
        new_fitness = fitness(text, new_gen, letter_freq, two_letter_freq, words_set)
        if new_fitness > current_fitness:
            best_gen = new_gen
            current_fitness = new_fitness
    return best_gen
        

def print_decoded(text, gen, i, steps=DEFAULT_STEPS_PER_PRINT):
    if (i+1) % steps == 0:
        print(decode(text, gen))

def print_best_fitness(iteration_number, fitnesses):
    print("Iteration: " + str(iteration_number) + " Best Fitness: " + str(max(fitnesses)))
    return max(fitnesses)

def calc_fitnesses(text, population, letter_freq, two_letter_freq, words_set):
    fitnesses = []
    for gen in population:
        fitnesses.append(fitness(text, gen, letter_freq, two_letter_freq, words_set))
    return fitnesses, len(population)

def repopulate(population, population_size=POPULATION_SIZE, mutation_chance=MUTATIONS_CHANCE):
    original_population_size = len(population)
    new_individuals_needed = population_size - len(population)

    for _ in range(new_individuals_needed):
        parent1 = random.choice(population[:original_population_size])
        parent2 = random.choice(population[:original_population_size])
        
        if random.random() < mutation_chance:
            child = mutate(crossover(parent1, parent2))
        else:
            child = crossover(parent1, parent2)
        
        population.append(child)

    return population


def selection(population,fitnesses, population_size=POPULATION_SIZE, survivor_fraction=SURVIVOR_FRACTION):
    population = [x for _, x in sorted(zip(fitnesses, population), key=lambda item: item[0], reverse=True)]
    if population_size * survivor_fraction < 1:
        return population[:1]
    return population[:int(population_size * survivor_fraction)]

def export(text, population, letter_freq, two_letter_freq, words_set, fitness_counter):
    best_gen = get_best_gen(text, population, letter_freq, two_letter_freq, words_set)
    create_plain(text, best_gen)
    create_perm(best_gen)
    print("Exported to " + FILE_NAME_PLAIN + " and " + FILE_NAME_PERM)
    print("Number of fitness calculations: " + str(fitness_counter))


def get_best_gen(text, population, letter_freq, two_letter_freq, words_set):
    fitnesses = [fitness(text, gen, letter_freq, two_letter_freq, words_set) for gen in population]
    return population[fitnesses.index(max(fitnesses))]
    

def create_plain(text,gen):
    text = decode(text, gen)
    with open(FILE_NAME_PLAIN, "w") as f:
        f.write(text)

def create_perm(gen):
    with open(FILE_NAME_PERM, "w") as f:
        for key, value in gen.items():
            f.write(key + " " + value + "\n")
     

def initialize_population(text,population_size):
    # get the most used letters in general
    letter_freq = get_letter_freq()
    # sort it and make it into an array
    most_used_letters = [x for _,x in sorted(zip(letter_freq.values(), letter_freq.keys()), reverse=True)]

    # get the most used letters in the text
    letter_freq = {}
    for letter in text:
        if letter in ALL_LETTERS:
            if letter in letter_freq:
                letter_freq[letter] += 1
            else:
                letter_freq[letter] = 1
    # sort it and make it into an array
    most_used_letters_text = [x for _,x in sorted(zip(letter_freq.values(), letter_freq.keys()), reverse=True)]
    # add the missing letters to the end
    for letter in ALL_LETTERS:
        if letter not in most_used_letters_text:
            most_used_letters_text.append(letter)

    # create the first gen as a permutation whereas the 1st most used letter in text goes to the 1st most used letter in general
    gen = {}
    for i in range(len(most_used_letters)):
        gen[most_used_letters_text[i]] = most_used_letters[i]

    # popultaion is a list of gens, where each gen is a dictionary(of letters, key:value)
    population = []
    for i in range(population_size):
        population.append(gen.copy())  
    return population

def decode(text, gen):
    decoded_chars = []
    for char in text:
        decoded_chars.append(gen.get(char, char))
    return ''.join(decoded_chars)


def get_letter_freq():
    frequencies = {}
    with open(FILE_NAME_LETTER_FREQ, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                frequency, letter = line.split('\t')
                letter = letter.lower()
                frequencies[letter] = float(frequency)
    return frequencies

def get_two_letter_freq():
    frequencies = {}
    with open(FILE_NAME_TWO_LETTER_FREQ, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                frequency, letters = line.split('\t')
                letters = letters.lower()
                frequencies[letters] = float(frequency)
            else:
                break
    return frequencies

def get_words_set():
    words = set()
    with open(FILE_NAME_DICT, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                words.add(line)
    return words

def fitness(text, gen, letter_freq, two_letter_freq, words_set):
    decoded_text = decode(text, gen)
    # Split into words and remove trailing punctuation
    words = (word.strip('.,;:?!') for word in decoded_text.split())
    word_score = sum(len(word) - 1 for word in words if word in words_set)
    letter_counter = Counter(decoded_text)
    letter_score = sum(letter_freq[letter] for letter in letter_counter if letter in letter_freq)
    two_letter_score = sum(two_letter_freq[two_letters] for two_letters in (decoded_text[i:i+2] for i in range(len(decoded_text)-1)) if two_letters in two_letter_freq)
    score = (WORD_SCORE_LAMBDA * word_score) + (LETTER_SCORE_LAMBDA * letter_score) + (TWO_LETTER_SCORE_LAMBDA * two_letter_score)
    return score

def mutate(gen):
    # copy gen
    new_gen = dict(gen)
    # pick a random letter
    letter = random.choice(list(new_gen.keys()))
    # pick a random letter to swap with
    swap_letter = random.choice(list(new_gen.keys()))
    # swap the letters
    new_gen[letter], new_gen[swap_letter] = new_gen[swap_letter], new_gen[letter]
    return new_gen

def crossover(gen1, gen2):
    new_gen = {}
    unused_letters = set(ALL_LETTERS)
    used_letters= set()
    duplicate_keys = set()

    for letter in gen1.keys():
        if random.randint(0, 1) == 0:
            new_gen[letter] = gen1[letter]
            if gen1[letter] in unused_letters:
                unused_letters.remove(gen1[letter])
            if gen1[letter] in used_letters:
                duplicate_keys.add(gen1[letter])
            used_letters.add(gen1[letter])
        else:
            new_gen[letter] = gen2[letter]
            if gen2[letter] in unused_letters:
                unused_letters.remove(gen2[letter])
            if gen2[letter] in used_letters:
                duplicate_keys.add(gen2[letter])
            used_letters.add(gen2[letter])
    
    # match all duplicate keys to unused letters
    for key in duplicate_keys:
        new_gen[key] = unused_letters.pop()
    return new_gen    


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_results_file_name():
    # remove the .csv 
    file_name = FILE_NAME_RESULTS[:-4]
    if os.path.exists(file_name + ".csv"):
        i =1
        while os.path.exists(file_name + str(i) + ".csv"):
            i += 1
        return file_name + str(i) + ".csv"
    else:
        return file_name + ".csv"


def normal_message(mutation_chance, population_size, survivor_fraction, index, max_runs):
            clear_screen()
            print("Running normal genetic algorithm with mutation chance: {}, population size: {}, survivor fraction: {}".format(mutation_chance, population_size, survivor_fraction))
            print("Run {}/{}".format(index, max_runs))

def darwinian_message(mutation_chance, population_size, survivor_fraction, index, max_runs, search_count):
            clear_screen()
            print("Running darwinian genetic algorithm with mutation chance: {}, population size: {}, survivor fraction: {}, search count: {}".format(mutation_chance, population_size, survivor_fraction, search_count))
            print("Run {}/{}".format(index, max_runs))

def lamirickian_message(mutation_chance, population_size, survivor_fraction, index, max_runs, search_count):
            clear_screen()
            print("Running lamirickian genetic algorithm with mutation chance: {}, population size: {}, survivor fraction: {}, search count: {}".format(mutation_chance, population_size, survivor_fraction, search_count))
            print("Run {}/{}".format(index, max_runs))

def data_to_csv(algorithms, mutation_chances, population_sizes, survivor_fractions, search_counts, text, letter_freq, two_letter_freq, words_set):
    file_name = get_results_file_name()
    current_run_index = 1
    max_runs = calc_max_runs(algorithms, mutation_chances, population_sizes, survivor_fractions, search_counts)
    with open(file_name, "w") as f:
        f.write(CSV_FILE_FIRST_ROW)
        for mutation_chance in mutation_chances:
            for population_size in population_sizes:
                for survivor_fraction in survivor_fractions:
                    for algorithm in algorithms:
                        if (algorithm == ALG_NORMAL):
                            normal_message(mutation_chance, population_size, survivor_fraction, current_run_index, max_runs)
                            last_iteration, best_score, fitness_counter = normal_genetic(text, letter_freq, two_letter_freq, words_set, False, population_size, survivor_fraction, mutation_chance)
                            f.write(f"{algorithm}, {mutation_chance}, {population_size}, {survivor_fraction}, X, {last_iteration}, {best_score}, {fitness_counter}\n")
                            current_run_index += 1
                        else:
                            for search_count in search_counts:
                                if (algorithm == ALG_DARWINIAN):
                                    darwinian_message(mutation_chance, population_size, survivor_fraction, current_run_index, max_runs, search_count)
                                    last_iteration, best_score, fitness_counter = darwinian_genetic(text, letter_freq, two_letter_freq, words_set, try_export=False, population_size=population_size, survivor_fraction=survivor_fraction, mutation_chance=mutation_chance, search_count=search_count)
                                else:
                                    lamirickian_message(mutation_chance, population_size, survivor_fraction, current_run_index, max_runs, search_count)
                                    last_iteration, best_score, fitness_counter = lamirick_genetic(text, letter_freq, two_letter_freq, words_set, try_export=False, population_size=population_size, survivor_fraction=survivor_fraction, mutation_chance=mutation_chance, search_count=search_count)
                                f.write(f"{algorithm}, {mutation_chance}, {population_size}, {survivor_fraction}, {search_count}, {last_iteration}, {best_score}, {fitness_counter}\n")
                                current_run_index += 1

def calc_max_runs(algorithms, mutation_chances, population_sizes, survivor_fractions, search_counts):
    max_runs = 0
    if ALG_NORMAL in algorithms:
        max_runs += len(mutation_chances) * len(population_sizes) * len(survivor_fractions)
    if ALG_DARWINIAN in algorithms:
        max_runs += len(mutation_chances) * len(population_sizes) * len(survivor_fractions) * len(search_counts)
    if ALG_LAMIRICK in algorithms:
        max_runs += len(mutation_chances) * len(population_sizes) * len(survivor_fractions) * len(search_counts)
    return max_runs




def main():
    special_main()


def get_vars(file_name):
    text = ""
    with open(file_name, "r") as f:
        text = f.read()
    letter_freq = get_letter_freq()
    two_letter_freq = get_two_letter_freq()
    words_set = get_words_set()
    return text, letter_freq, two_letter_freq, words_set

def normal_main():
    if (len(sys.argv) < 3) or (sys.argv[2] not in ["1", "2", "3"]):
        print(INCORRECT_ARGS_MESSAGE)
        sys.exit(1)

    file_name = sys.argv[1]
    mode = int(sys.argv[2])
    text, letter_freq, two_letter_freq, words_set = get_vars(file_name)

    if(mode == 1):
        normal_genetic(text, letter_freq, two_letter_freq, words_set)
    elif(mode == 2):
        darwinian_genetic(text, letter_freq, two_letter_freq, words_set)
    elif(mode == 3):
        lamirick_genetic(text, letter_freq, two_letter_freq, words_set)


def run_algorithm_in_parallel(algorithm, search_counts, mutation_chances, population_sizes, survivor_fractions, text, letter_freq, two_letter_freq, words_set):
    for search_count in search_counts:
        data_to_csv([algorithm], mutation_chances, population_sizes, survivor_fractions, [search_count], text, letter_freq, two_letter_freq, words_set)


def special_main():
    file_name = 'enc.txt'
    text, letter_freq, two_letter_freq, words_set = get_vars(file_name)

    algorithms = [ALG_NORMAL, ALG_DARWINIAN, ALG_LAMIRICK]
    mutation_chances = [0.05,0.1,0.2,0.3]
    population_sizes = [100, 500,1000,2000]
    survivor_fractions = [0.1, 0.2, 0.3,0.4]
    search_counts = [1,2]


    processes = []
    for algorithm in algorithms:
        for search_count in search_counts:
            process = multiprocessing.Process(
                target=run_algorithm_in_parallel,
                args=(algorithm, [search_count], mutation_chances, population_sizes, survivor_fractions, text, letter_freq, two_letter_freq, words_set)
            )
            processes.append(process)

    for process in processes:
        process.start()
        

    for process in processes:
        process.join()
    
        



if __name__ == '__main__':
    main()