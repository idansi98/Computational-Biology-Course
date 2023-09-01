import tkinter as tk
import tkinter.messagebox as msgbox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random
import copy
import sys
import matplotlib
import decimal


# Collects stats over a single simulation
class Stats:
    def __init__(self, total_people):
        self.infected_over_steps = {}
        self.heard_over_steps = {}
        self.total_people = total_people
        self.rounds_till_all_infected = 0
        self.rounds_without_improvements = 0


# PARAMETERS
DRAW_EVERY_X_ITERATIONS = 50
SPECIAL_CONFIG = False
L = 5
P = 0.9
BOARD_SIZE = 100
DIAGONAL_NEIGHBORS = False
MAX_ROUNDS_WITHOUT_IMPROVEMENTS =  (1 / P) * L * 50
MAX_SIMS = 1 # WHEN PUBLISHING IT, IT IS IMPORTANT THIS WILL BE 1
S1_Chance = 0.25
S2_Chance = 0.25
S3_Chance = 0.25
S4_Chance = 0.25



# Matrices
config = None
rumors = None
rumors_recieved = None
waiting = None
is_spreader = None
is_spreading_right_now = None

# GUI
root = None
canvas = None
rect_ids = []
button = None
iteration_label = None
percentage_label = None

# Algorithm
is_running = False
iterNum = 0
simNum = 0

# Stats
stats = []
Percentege = 0
Epsilon = 0.0000001


def main():
    # Define P
    if len(sys.argv) > 1:
        global P
        P = float(sys.argv[1])

    # Define L
    if len(sys.argv) > 2:
        global L
        L = int(sys.argv[2])
    

    # Define Special Configuration or None
    if len(sys.argv) > 3:
        global SPECIAL_CONFIG
        if sys.argv[3] == "special":
            SPECIAL_CONFIG = True
        else:
            SPECIAL_CONFIG = False

    # Define Diagonal Neighbors or None
    if len (sys.argv) > 4:
        global DIAGONAL_NEIGHBORS
        if sys.argv[4] == "diagonal":
            DIAGONAL_NEIGHBORS = True
        else:
            DIAGONAL_NEIGHBORS = False

    # Draw every X iterations
    if len (sys.argv) > 5:
        global DRAW_EVERY_X_ITERATIONS
        DRAW_EVERY_X_ITERATIONS = int(sys.argv[5])

    # Max rounds with no improvement
    if len (sys.argv) > 6:
        global MAX_ROUNDS_WITHOUT_IMPROVEMENTS
        MAX_ROUNDS_WITHOUT_IMPROVEMENTS = int(sys.argv[6])

    # Define S1-S4
    if len(sys.argv) > 10:
        global S1, S2, S3, S4
        S1 = float(sys.argv[7])
        S2 = float(sys.argv[8])
        S3 = float(sys.argv[9])
        S4 = float(sys.argv[10])
    #     decimal.getcontext().prec = 10
    #     Create Decimal objects for each number
    #     S1 = decimal.Decimal(sys.argv[7])
    #     S2 = decimal.Decimal(sys.argv[8])
    #     S3 = decimal.Decimal(sys.argv[9])
    #     S4 = decimal.Decimal(sys.argv[10])
    
        # check they add to 1.0
        if abs(S1 + S2 + S3 + S4 - 1.0) > Epsilon:
            print("S1 + S2 + S3 + S4 must equal 1.0")
            sys.exit(1)

    init()
    first_draw()
    root.mainloop()

def init():
    global iterNum, config, rumors, rumors_recieved, waiting, is_spreader,is_spreading_right_now, Percentege
    config = np.zeros([BOARD_SIZE, BOARD_SIZE])
    rumors = np.zeros([BOARD_SIZE, BOARD_SIZE])
    rumors_recieved = np.zeros([BOARD_SIZE, BOARD_SIZE])
    waiting = np.zeros([BOARD_SIZE, BOARD_SIZE])
    is_spreader = np.zeros([BOARD_SIZE, BOARD_SIZE])
    is_spreading_right_now = np.zeros([BOARD_SIZE, BOARD_SIZE])
    Percentege = 0
    iterNum = 0

    if SPECIAL_CONFIG:
        set_initial_config_special()
    else:
        place_people()
        set_initial_config()
    infect_first_person()

def place_people():
    global config, stats
    config = np.random.choice([0, 1], size=(BOARD_SIZE, BOARD_SIZE), p=[1-P, P])
    stats.append(Stats(np.count_nonzero(config)))
    
def set_initial_config():
    global config
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if config[x, y] == 1:
                options = [1, 2, 3, 4]
                weights = [S1_Chance, S2_Chance, S3_Chance, S4_Chance]
                config[x, y] = random.choices(options, weights)[0]

def set_initial_config_special():
    global config, stats
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            # sin x
            z = np.sin(x/10 +y/10) 
            # make it an int
            z = int(z * 40  + x*0.1)

            z = z%9
            # make it positive
            if z < 0:
                z = z + 5

    

            if z == 0:
                config[x, y] = 0
            elif z == 1:
                config[x, y] = 0
            elif z == 2:
                config[x, y] = 0
            elif z == 3:
                config[x, y] = 0
            elif z == 4:
                config[x, y] = 4
            elif z == 5:
                config[x, y] = 3
            elif z == 6:
                config[x, y] = 3
            elif z == 7:
                config[x, y] = 3
            elif z == 8:
                config[x, y] = 4

            if x==y:
                config[x, y] = 1
            
    stats.append(Stats(np.count_nonzero(config)))

def infect_first_person():
    global config, rumors, is_spreader
    while True:
        x = random.randint(0, BOARD_SIZE-1)
        y = random.randint(0, BOARD_SIZE-1)
        if config[x, y] != 0:
            break
    config[x, y] = 1
    rumors[x, y] = 1
    is_spreader[x, y] = 1


def infect():
    global config, rumors, rumors_recieved, waiting, is_spreader, iterNum, Percentege

    is_spreading_right_now.fill(0)

    # Check if a cell is infected
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            being_infected = (
                is_spreader[x, y] == 0) and (
                (
                    (rumors_recieved[x, y] == 1) and 
                    (
                        (config[x, y] == 1) or 
                        (config[x, y] == 2 and np.random.binomial(1, 2/3)) or 
                        (config[x, y] == 3 and np.random.binomial(1, 1/3))
                    )
                ) or (
                    (rumors_recieved[x, y] > 1) and 
                    (
                        (config[x, y] == 1) or
                        (config[x, y] == 2) or
                        (config[x, y] == 3 and np.random.binomial(1, 2/3)) or
                        (config[x, y] == 4 and np.random.binomial(1, 1/3))
                    )
                ))
            if (being_infected):
                waiting[x, y] = iterNum + L
                is_spreader[x, y] = 1
    rumors_recieved.fill(0)

    # Check if a cell is spreading right now
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if is_spreader[x, y]:
                if waiting[x, y] <= iterNum:
                    waiting[x, y] = iterNum + L 
                    neighbors = get_neighbores(x, y)
                    is_spreading_right_now[x, y] = 1
                    for (i, j) in neighbors:
                        rumors_recieved[i, j] += 1
                        if (rumors[i, j] == 0):
                            Percentege += 100 / stats[simNum].total_people
                        rumors[i, j] = 1

def get_neighbores(x, y):
    global DIAGONAL_NEIGHBORS
    if DIAGONAL_NEIGHBORS:
        return [(i, j) for (i, j) in [(x, y-1), (x, y+1), (x-1, y), (x+1, y), (x-1, y-1), (x+1, y+1), (x-1, y+1), (x+1, y-1)] if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and config[i, j] != 0]
    else:
        return [(i, j) for (i, j) in [(x, y-1), (x, y+1), (x-1, y), (x+1, y)] if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and config[i, j] != 0]

def one_step(doesDraw = False):
    global iterNum
    infect()
    update_stats()
    if doesDraw:
        draw()
    iterNum += 1
    root.update()

def multiple_steps():
    global is_running, simNum
    finished = False
    is_running = True
    button.config(text="Stop", command=stop)
    while True:
        if iterNum % DRAW_EVERY_X_ITERATIONS == DRAW_EVERY_X_ITERATIONS-1:
            one_step(True)
        else:
            one_step(False)
        finished = (stats[simNum].rounds_till_all_infected != 0) or (stats[simNum].rounds_without_improvements >= MAX_ROUNDS_WITHOUT_IMPROVEMENTS)
        if finished or not is_running:
            break
    is_running = False
    if finished:
        draw()
        root.update()
        simNum = simNum + 1
        if simNum == MAX_SIMS:
            show_stats()
            destroy_button()
        else:
            init()
            draw()
            multiple_steps()
    
  
def update_stats():
    global stats, rumors_recieved, iteration_label
    stats[simNum].heard_over_steps[iterNum] = np.count_nonzero(rumors)
    stats[simNum].infected_over_steps[iterNum] = np.count_nonzero(is_spreader)
    stats[simNum].total_people = np.count_nonzero(config)
    # check if all people are infected
    if np.count_nonzero(config == 1) == stats[simNum].total_people:
        stats[simNum].rounds_till_all_infected = iterNum
    # check if there are no improvements
    if iterNum > 1 and stats[simNum].infected_over_steps[iterNum] == stats[simNum].infected_over_steps[iterNum - 1]:
        stats[simNum].rounds_without_improvements += 1
    else:
        stats[simNum].rounds_without_improvements = 0

def show_stats():
    draw_infections_and_heard_per_iteration()

def calculate_average_stats(stats_list):
    if len(stats_list) == 0:
        return None
    elif len(stats_list) == 1:
        return stats_list[0], len(stats_list[0].infected_over_steps)
    else:
        num_stats = len(stats_list)
        total_people = sum(s.total_people for s in stats_list) // num_stats
        infected_over_steps = {}
        heard_over_steps = {}
        longest_iterations = 0
        for s in stats_list:
            longest_iterations = max(longest_iterations, len(s.infected_over_steps))
        for s in stats_list:
            list_length = len(s.infected_over_steps)
            # get the last element
            last_element_infected = s.infected_over_steps[list_length-1]
            last_element_heard = s.heard_over_steps[list_length-1]
            # add the last element to the end of the list till it reaches the longest list
            for i in range(list_length, longest_iterations):
                s.infected_over_steps[i] = last_element_infected
                s.heard_over_steps[i] = last_element_heard
        for i in range(longest_iterations):
            #print("I is " + str(i))
            #print("infected_over_steps is " + str(len(stats_list[i].infected_over_steps)))
            #print("heard_over_steps is " + str(len(stats_list[i].heard_over_steps)))
            infected_over_steps[i] = sum(s.infected_over_steps[i] for s in stats_list) // num_stats
            heard_over_steps[i] = sum(s.heard_over_steps[i] for s in stats_list) // num_stats

        new_stats = Stats(total_people)
        new_stats.infected_over_steps = infected_over_steps
        new_stats.heard_over_steps = heard_over_steps
        return new_stats, longest_iterations

def draw_infections_and_heard_per_iteration():
    global stats
        # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Calculate the average stats
    avg_stats, longest_iterations = calculate_average_stats(stats)

    # Plot the first graph on the first subplot
    ax1.plot(list(avg_stats.infected_over_steps.keys()), [100 * x / avg_stats.total_people for x in list(avg_stats.infected_over_steps.values())])
    ax1.set_title('People who believe the rumor')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Believers [%]')
    ax1.set_xlim(0, longest_iterations)

    # Plot the second graph on the second subplot
    ax2.plot(list(avg_stats.heard_over_steps.keys()), [100 * y / avg_stats.total_people for y in list(avg_stats.heard_over_steps.values())])
    ax2.set_title('People who heard the rumor')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Rumor heard [%]')
    ax2.set_xlim(0, longest_iterations)

    # Add text labels for simulation parameters on the third subplot
    ax3.text(0.5, 0.8, f'Total Population: {avg_stats.total_people}', ha='center', va='center', fontsize=12)
    ax3.text(0.5, 0.5, f'Believers: {avg_stats.infected_over_steps[longest_iterations-1]}', ha='center', va='center', fontsize=12)
    ax3.text(0.5, 0.2, f'Heard The Rumor: {avg_stats.heard_over_steps[longest_iterations-1]}', ha='center', va='center', fontsize=12)

    # Remove ticks and spines from the third subplot
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3)

    # Display the figure
    plt.show()


def draw_grid():
    global canvas
    for i in range(0, 600, 6):
        canvas.create_line(i, 0, i, 600)
        canvas.create_line(0, i, 600, i)

def add_button():
    global canvas, root, button
    button = tk.Button(root, text="Start", command=multiple_steps)
    # make it a bit more to the left
    button.pack(side=tk.TOP, padx=10)
    # make it bigger

    button.config(height=2, width=10)
    button.pack()

def destroy_button():
    global button
    button.destroy()

def stop():
    global is_running
    is_running = False
    button.config(text="Start", command=multiple_steps)

def add_label():
    global canvas, root, iteration_label, percentage_label
    iteration_label = tk.Label(root, text="Iteration: 0", font=("Helvetica", 16))
    iteration_label.pack(side=tk.TOP, padx=10, pady=10)
    # change size
    iteration_label.config(height=2, width=20)
    percentage_label = tk.Label(root, text="Rumor Spread: 0%", font=("Helvetica", 16))
    percentage_label.pack(side=tk.TOP, padx=10, pady=10)
    # change size
    percentage_label.config(height=2, width=20)

def get_color(x, y):
    # add green if more likely to believe rumor
    if (config[x, y] == 0):
        green = 0
    else:
        green = int(50 * (5-config[x, y]))
    # add blue if is already infected
    blue = int(50 * rumors[x,y] + 200 * is_spreader[x, y])
    # add red if is a spreading right now
    red = int(250 * is_spreading_right_now[x, y])
    return f"#{red:02x}{green:02x}{blue:02x}"

def draw_people_squares(destroy = True):
    global canvas, rect_ids
    if destroy:
        for rect_id in rect_ids:
            canvas.delete(rect_id)
    rect_ids = []
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            rect_id = canvas.create_rectangle(x*6, y*6, x*6+6, y*6+6, fill=get_color(x,y))
            rect_ids.append(rect_id)

def first_draw():
    global canvas, root, iteration_label, rect_ids, percentage_label
    root = tk.Tk()
    root.title("Rumors")
    canvas = tk.Canvas(root, width=600, height=600)
    canvas.pack(side=tk.LEFT, padx=10, pady=10)
    add_label()
    add_button()
    draw_grid()
    draw_people_squares(destroy=False)

def draw():
    global canvas, root, iteration_label, rect_ids, percentage_label
    iteration_label.config(text=f"Iteration:{iterNum+1}")
    percentage_label.config(text=f"Rumor Spread:{round(Percentege, 2)}%")

    draw_people_squares()
    #There's an option so instead of pressing the button it will run automatically
    #for this use: canvas.after(time in ms, function you want to run)                       




if __name__ == "__main__":
    main()




