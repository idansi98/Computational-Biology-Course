# Ido Tziony 206534299
# Idan Simai 206821258

import json
import math


FIRST_DATA_PATH = 'testnet0.txt'
SECOND_DATA_PATH = 'testnet1.txt'
FIRST_WEIGHTS_PATH = 'wnet0'
SECOND_WEIGHTS_PATH = 'wnet1'
FIRST_PREDICTION_PATH = 'prediction0.txt'
SECOND_PREDICTION_PATH = 'prediction1.txt'

FIRST_TRUE_LABELS_PATH = 'answer0.txt'
SECOND_TRUE_LABELS_PATH = 'answer1.txt'

# DEBUG true prints the accuracy
DEBUG = False

# Every 0 in the data set is replaced with -1
def normalize(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] == 0:
                x[i][j] = -1
    return x


def load_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    
    features = []
    for line in lines:
        temp_list = []
        for c in line:
            temp_list.append(int(c))
        features.append(temp_list)

    return features

    

def print_to_file(data, file_path):
    with open(file_path, 'w') as f:
        # for all predictions but the last one
        for i in range(len(data)-1):
            f.write(str(data[i]) + '\n')
        # for the last prediction
        f.write(str(data[-1]))


class Edge:
    def __init__(self, in_node, out_node, weight, type):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.type = type

    def __hash__(self):
        return hash((self.in_node, self.out_node, self.type))

    def __eq__(self, other):
        return self.in_node == other.in_node and self.out_node == other.out_node and self.type == other.type

class Genome:
    INPUT_NODE_COUNT = 16
    

    OUTPUT_NODE = 999999
    BIAS_NODE = -1
    CALCULATION_ERROR = -float('inf') 

    def __init__(self):
        pass
    
    def save_as_json(self, file_path):
        with open(file_path, 'w') as f:
            f.write(str(self.hidden_nodes_count) + "\n")
            for edge in self.edges:
                f.write(json.dumps(edge.__dict__) + "\n")
    
    def load_from_json(self, file_path):
        with open(file_path, 'r') as f:
            self.hidden_nodes_count = int(f.readline())
            self.edges = []
            for line in f:
                if line == "\n":
                    continue
                edge = Edge(**json.loads(line))
                self.edges.append(edge)
    

    def get_all_reached_nodes(self, edges_coming_from):
        reached_nodes = set(range(Genome.INPUT_NODE_COUNT)).union({Genome.BIAS_NODE})

        index = 0
        while index < len(reached_nodes):
            current_node = list(reached_nodes)[index]
            if current_node != Genome.OUTPUT_NODE:
                for edge in edges_coming_from[current_node]:
                    if edge.out_node not in reached_nodes:
                        reached_nodes.add(edge.out_node)
            index += 1
        return reached_nodes
    
    def remove_all_nodes_not_connected_to_output(self, reached_nodes, edges_coming_to):
        new_reached_nodes = set({Genome.OUTPUT_NODE})
        index = 0
        while index < len(new_reached_nodes):
            current_node = list(new_reached_nodes)[index]
            if current_node != Genome.INPUT_NODE_COUNT:
                for edge in edges_coming_to[current_node]:
                    if (edge.in_node in reached_nodes) and (edge.in_node not in new_reached_nodes):
                        new_reached_nodes.add(edge.in_node)
            index += 1
        return new_reached_nodes
    
    def get_edges_to_and_from(self):
        all_node_numbers = set(range(Genome.INPUT_NODE_COUNT + self.hidden_nodes_count)).union({Genome.BIAS_NODE}).union({Genome.OUTPUT_NODE})
        edges_coming_to = {node: [] for node in all_node_numbers}
        edges_coming_from = {node: [] for node in all_node_numbers}
        for edge in self.edges:
            if edge.weight != 0:
                edges_coming_to[edge.out_node].append(edge)
                edges_coming_from[edge.in_node].append(edge)

        return edges_coming_to, edges_coming_from
    

    def remove_edges_from_unreached_nodes(edges_coming_to, reached_nodes):
        new_edges_coming_to = {}
        # remove edges from and to unreached nodes
        for node in edges_coming_to:
            new_edges_coming_to[node] = []
            for edge in edges_coming_to[node]:
                if edge.in_node in reached_nodes and edge.out_node in reached_nodes:
                    new_edges_coming_to[node].append(edge)
   

        return new_edges_coming_to

    def create_model(self):
        # Get all relevant nodes for the calculation
        edges_coming_to, edges_coming_from = self.get_edges_to_and_from()
        reached_nodes_from_input = self.get_all_reached_nodes(edges_coming_from)
        nodes_in_calculation_path = self.remove_all_nodes_not_connected_to_output(reached_nodes_from_input, edges_coming_to)
        edges_coming_to = Genome.remove_edges_from_unreached_nodes(edges_coming_to, nodes_in_calculation_path)

        if Genome.OUTPUT_NODE not in nodes_in_calculation_path:
            self.model = None
            return
        
        nodes_in_calculation_path.remove(Genome.OUTPUT_NODE)
        
        calculated_nodes_indexes = set(range(Genome.INPUT_NODE_COUNT)).union({Genome.BIAS_NODE})

        input_nodes_of = {node: {edge.in_node for edge in edges_coming_to[node]} for node in nodes_in_calculation_path}
        nodes_to_calculate = [node for node in nodes_in_calculation_path if node not in calculated_nodes_indexes]
        order_of_calculation = []
        node_coming_and_weight = []


        # Get the order of calculation
        while len(nodes_to_calculate) > 0:
            has_progress = False
            for node in nodes_to_calculate:
                    if input_nodes_of[node].issubset(calculated_nodes_indexes):
                        calculated_nodes_indexes.add(node)
                        nodes_to_calculate.remove(node)
                        order_of_calculation.append(node)
                        node_coming_and_weight.append([(edge.in_node, edge.weight) for edge in edges_coming_to[node]])
                        has_progress = True
            if(not has_progress):
                self.model = None
                return
            
        # add for output node
        self.nodes_coming_and_weight_final = [(edge.in_node, edge.weight) for edge in edges_coming_to[Genome.OUTPUT_NODE]]
            
        
        self.order_of_calculation = order_of_calculation
        self.node_coming_and_weight = node_coming_and_weight


        
    # Uses the order of calculation to calculate the output
    def predict(self, x):
        answers = []
        for sample in x:
            calculated_nodes = sample
            calculated_nodes.extend([0 for _ in range(Genome.INPUT_NODE_COUNT)])
            calculated_nodes.extend([0 for _ in range(self.hidden_nodes_count)])
            calculated_nodes.append(1)
            


            for i in range(len(self.order_of_calculation)):
                node = self.order_of_calculation[i]
                node_coming_and_weight = self.node_coming_and_weight[i]
                for coming_node, weight in node_coming_and_weight:
                    calculated_nodes[node] += calculated_nodes[coming_node] * weight
                calculated_nodes[node] = 1 / (1 + math.exp(-calculated_nodes[node]))

            # now we have the output node
            node = Genome.OUTPUT_NODE
            outpt_sum = 0
            node_coming_and_weight = self.nodes_coming_and_weight_final
            for coming_node, weight in node_coming_and_weight:
                outpt_sum+= calculated_nodes[coming_node] * weight
            

            after_sig = 1 / (1 + math.exp(-outpt_sum))
            answers.append(1 if after_sig > 0.5 else 0)
        return answers


# For testing
def check_accuracy(genome, x, y):
    answers = genome.predict(x)
    correct_answers = 0
    for i in range(len(y)):
        if answers[i] == y[i][0]:
            correct_answers += 1
    return correct_answers / len(y) 

   
    
if __name__ == "__main__":
    best_gen_ww0 = Genome()
    best_gen_ww0.load_from_json(FIRST_WEIGHTS_PATH)
    best_gen_ww0.create_model()
    x_test_one = load_file(FIRST_DATA_PATH)
    x_test_one = normalize(x_test_one)
    print_to_file(best_gen_ww0.predict(x_test_one), FIRST_PREDICTION_PATH)

    best_gen_ww1 = Genome()
    best_gen_ww1.load_from_json(SECOND_WEIGHTS_PATH)
    best_gen_ww1.create_model()
    x_test_two = load_file(SECOND_DATA_PATH)
    x_test_two = normalize(x_test_two)
    print_to_file(best_gen_ww1.predict(x_test_two), SECOND_PREDICTION_PATH)

    if DEBUG:
        y_test_one = load_file(FIRST_TRUE_LABELS_PATH)
        y_test_two = load_file(SECOND_TRUE_LABELS_PATH)
        print("Accuracy of first model: ", check_accuracy(best_gen_ww0, x_test_one, y_test_one))
        print("Accuracy of second model: ", check_accuracy(best_gen_ww1, x_test_two, y_test_two))



