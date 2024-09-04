from copy import copy, deepcopy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))


import matplotlib.pyplot as plt
import networkx as nx

from causalgraphicalmodels import CausalGraphicalModel

from hierarchicalcausalmodels.utils.distributions import EmpiricalDistribution
from hierarchicalcausalmodels.utils.parsing_utils import source_sample
from hierarchicalcausalmodels.utils.utils import linear_functor, logit_functor, random_functor, additive_functor, \
    is_empty, distribution_functor, cleaner, extract_distributions_from_data

import numpy as np
from numba import cuda

@cuda.jit
def sample_unit_node_gpu(node, predecessors, node_function, samples, result, sizes):
    i, j = cuda.grid(2)
    if i < len(sizes) and j < sizes[i]:
        parent_samples = {parent: samples[parent][i][j] for parent in predecessors}
        result[i][j] = node_function(parent_samples)

@cuda.jit
def sample_subunit_node_gpu(node, predecessors, node_function, samples, result):
    i = cuda.grid(1)
    if i < len(samples[list(samples.keys())[0]]):
        parent_samples = {parent: samples[parent][i] for parent in predecessors}
        result[i] = node_function(parent_samples)

class HSCM:
    def __init__(self, nodes: set, edges: set, unit_nodes: set, subunit_nodes: set, sizes: list, node_functions: dict,
                 data: dict):
        # each scm comes with a size dict for sampling
        self.subunit_nodes = {"_" + k for k in
                              subunit_nodes}  # to keep track of the names of the subunit nodes with the "_" prefix
        self.unit_nodes = unit_nodes  # to keep track of the names of the unit nodes without the "_" prefix
        self.subunit_nodes_names = subunit_nodes  # to keep track of the names of the subunit nodes without the "_" prefix
        self.node_function = node_functions  # dictionary of functions indexed by the nodes
        self.nodes = {("_" + k if k in subunit_nodes else k) for k in
                      nodes}  # to keep track of the names of the nodes with the "_" prefix

        self.edges = set()
        for edge in edges:
            x_temp, y_temp = edge
            if x_temp in subunit_nodes:  # if x is a subunit node according to former notation
                x_temp = '_' + x_temp
            if y_temp in subunit_nodes:  # if y is a subunit node according to former notation
                y_temp = '_' + y_temp
            self.edges.add((x_temp, y_temp))

        self.sizes = sizes
        predecessors = dict()
        self.coeffs = dict()
        self.additive_functions = dict()

        # for predecessors
        for node in self.unit_nodes:
            for i in range(len(sizes)):
                predecessors[node + str(i)] = set()  # set of predecessors for each node
        for node in self.subunit_nodes:
            for i in range(len(sizes)):
                for j in range(sizes[i]):
                    predecessors[node + str(i) + '_' + str(j)] = set()

        for parent, child in self.edges:
            if parent in self.subunit_nodes:
                if child in self.subunit_nodes:
                    for i in range(len(sizes)):
                        for j in range(sizes[i]):
                            predecessors[child + str(i) + '_' + str(j)] = {parent + str(i) + '_' + str(
                                j)}  # if parent and child are subunit nodes ie the same person, there is only one parent
                else:
                    for i in range(len(sizes)):  # seul cas ou on a un parent subunit qui a un enfant unit
                        temp = set()
                        for j in range(sizes[i]):
                            temp.add(parent + str(i) + '_' + str(j))

                        predecessors[child + str(i)].add(frozenset(temp))
            else:
                if child in self.subunit_nodes:
                    for i in range(len(sizes)):
                        for j in range(sizes[i]):
                            predecessors[child + str(i) + '_' + str(j)].add(parent + str(i))
                else:
                    for i in range(len(sizes)):
                        predecessors[child + str(i)].add(parent + str(i))

        self.predecessors = predecessors
        self.cgm = CausalGraphicalModel(nodes=self.nodes, edges=self.edges)
        self.data = data
        self.node_distribution = dict()  # dictionary of distributions passing functions indexed by the nodes, taking all previous values as parameters, and a random(0,1) as a last parameter to perform a sampling (with a ppf like norm.ppf etc .. for instance)
        self.aggregator_functions = {}  # dictionary of functions indexed by the unit nodes, and for each, a dictionary of functions indexed by the subunit nodes with functions like np.mean, np.median, np.std etc ...

        for node in self.unit_nodes:
            self.aggregator_functions[node] = dict()

        # by default, we take the mean
        for (x, y) in self.edges:
            if x in self.subunit_nodes_names and y in self.unit_nodes:  # we use the same notation as in the predecessors without '_' prefix for the subunit nodes
                self.aggregator_functions[y]['_' + x] = lambda d: np.mean(
                    np.array(list(d)))  # not the most efficient way but to dev a better one





        self.node_experimental_distribution = dict()  # dictionary of distributions.py like indexed by the nodes to see and explore them
        self.node_theoretical_distribution = dict()  # dictionary of distributions.py like indexed by the nodes for interventions



















































    #############################################
    ################ Basic methods ##############
    #############################################
    def add_node(self, node, edges, is_unit, function):
        self.nodes.add(node)
        self.edges = self.edges | edges
        self.node_function[node] = function
        if is_unit:
            self.unit_nodes.add(node)
        else:
            self.subunit_nodes.add(node)
        if is_unit:
            for parent, child in edges:
                if parent == node:
                    if child in self.subunit_nodes:
                        for i in range(len(self.sizes)):
                            for j in range(self.sizes[i]):
                                self.predecessors[child + str(i) + '_' + str(j)].add(parent + str(i))
                    else:
                        for i in range(len(self.sizes)):
                            self.predecessors[child + str(i)].add(parent + str(i))
                elif child == node:
                    if parent in self.subunit_nodes:
                        for i in range(len(self.sizes)):
                            for j in range(self.sizes[i]):
                                self.predecessors[child + str(i)].add(parent + str(i) + '_' + str(j))
                    else:
                        for i in range(len(self.sizes)):
                            self.predecessors[child + str(i)].add(parent + str(i))
        else:
            for parent, child in edges:
                if parent == node:
                    if child in self.subunit_nodes:
                        for i in range(len(self.sizes)):
                            for j in range(self.sizes[i]):
                                self.predecessors[child + str(i) + '_' + str(j)].add(parent + str(i) + '_' + str(j))
                    else:
                        for i in range(len(self.sizes)):
                            for j in range(self.sizes[i]):
                                self.predecessors[child + str(i)].add(parent + str(i) + '_' + str(j))
                elif child == node:
                    if child in self.subunit_nodes:
                        for i in range(len(self.sizes)):
                            for j in range(self.sizes[i]):
                                self.predecessors[child + str(i)].add(parent + str(i) + '_' + str(j))
                    else:
                        for i in range(len(self.sizes)):
                            for j in range(self.sizes[i]):
                                self.predecessors[child + str(i) + '_' + str(j)].add(parent + str(i))


    def set_size(self, num_unit, size):
        self.sizes[num_unit] = size



    def print_predecessors(self):
        for node, preds in self.predecessors.items():
            print(node, preds)


    #######################
    # Do-calculus functions
    #######################
    def set_aggregator(self, edge, aggregator_function):

        print(self.aggregator_functions, 'aggregator_functions')
        # aggregator will be used for (a,b) with a subunit node and b unit node, for instance the mean of all student SAT score in a school
        parent, child = edge
        if parent in self.subunit_nodes_names and child in self.unit_nodes:  # we use the same notation as in the predecessors without '_' prefix for the subunit nodes
            self.aggregator_functions[child]['_' + parent] = aggregator_function
        else:
            print("You can't set an aggregator for this edge", edge)



    def set_all_aggregator_to_mean(self):
        for (x, y) in self.edges:
            if x in self.subunit_nodes_names and y in self.unit_nodes:
                self.aggregator_functions[y]['_' + x] = lambda d: np.mean(np.array(list(d)))

    def set_aggregator_to_mean(self, edge):
        x, y = edge
        if x in self.subunit_nodes_names and y in self.unit_nodes:
            self.aggregator_functions[y]['_' + x] = lambda d: np.mean(np.array(list(d)))





    # Distributions' calculus related methods
    def set_distributions(self, distributions):
        # those distributions.py are inverted cumulated distribution function from [0,1] to R, taking, using distributions.py(a,b) where a is parameter (predecessors values) and b sampled uniformly in [0,1]
        for node in self.unit_nodes:
            self.node_distribution[node] = distributions[node]
        for node in self.subunit_nodes_names:
            self.node_distribution['_' + node] = distributions[node]



    def set_distributions_from_generator(self, generator):
        # reminder : node_distribution is theoretical
        for node in self.unit_nodes:
            self.node_distribution[node] = lambda a, b: generator(node, a, b)
        for node in self.subunit_nodes_names:
            self.node_distribution['_' + node] = lambda a, b: generator(node, a, b)



    def set_experimental_distributions_from_data(self):
        distributions = extract_distributions_from_data(self.data, self.nodes, self.unit_nodes, self.sizes)
        for node in self.unit_nodes:
            self.node_experimental_distribution[node] = distributions[node]
        for node in self.subunit_nodes:
            self.node_experimental_distribution[node] = distributions[node]  # which is a dict


    def set_theoretical_distribution_to_node(self, node, distribution_object):
        # be careful, this distribution is a Distribution object ! (not a function)
        self.node_theoretical_distribution[node] = distribution_object
        return





    def set_function_to_node(self, node, function):
        # note that this function is a deterministic function and is used only for HSCM sampling
        self.node_function[node] = function

    def set_value_to_node(self, node, value):
        # node is given with the asker convention
        if node in self.subunit_nodes:
            print("You can't set a value for a subunit node, but a probability distribution")
        else:
            self.node_function[node] = lambda d: value

    def set_distribution_to_node(self, node, distribution):
        # be careful, this distribution is a function that take a parameter and a number sampled uniformly in [0,1], not a Distribution object
        self.node_distribution[node] = distribution
        return



    #############################################
    # Intervention related methods
    #############################################
    def soft_intervention(self, node, distribution_object):
        return

    def hard_intervention(self, node, value):

        return

    def soft_conditional_intervention(self, node, distribution_object):
        # conditioning on its parent ! a_i_j follow q_star(a|parents(a_i_j)) -> parents are subunits z_i_j or units x_i
        return































































    #############################################
    ############### CAUSAL MODELS ###############
    #############################################

    # ad-hoc models for linear and logistic regression

    def linear_model(self, coeffs):  # coeffs is a dictionary of coefficients-dicts for each node, + a mean and a std
        for dicts in self.subunit_nodes_names:
            temp = dict()
            for key in coeffs[dicts].keys():
                if key != 'mean' and key != 'std':
                    if key in self.subunit_nodes_names:
                        temp['_' + key] = coeffs[dicts][key]
                    else:
                        temp[key] = coeffs[dicts][key]
                else:
                    temp[key] = coeffs[dicts][key]

            self.coeffs['_' + dicts] = temp

        for dicts in self.unit_nodes:
            temp = dict()
            for key in coeffs[dicts].keys():
                if key != 'mean' and key != 'std':
                    if key in self.subunit_nodes_names:
                        temp['_' + key] = coeffs[dicts][key]
                    else:
                        temp[key] = coeffs[dicts][key]
                else:
                    temp[key] = coeffs[dicts][key]

            self.coeffs[dicts] = temp

        if is_empty(self.aggregator_functions):
            temp_aggregator_functions = dict()
            for (x, y) in self.edges:
                if x in self.subunit_nodes_names and y in self.unit_nodes:
                    temp_aggregator_functions[y]['_' + x] = lambda d: np.mean(np.array(list(d)))
        else:
            temp_aggregator_functions = self.aggregator_functions

        def create_lambda(node):
            return lambda d: linear_functor(d, str(node), self.coeffs, node in self.unit_nodes,
                                            temp_aggregator_functions)

        lambda_functions = {node: create_lambda(node) for node in self.nodes}
        self.node_function = deepcopy(lambda_functions)

    def logistic_model(self, coeffs):
        for dicts in self.subunit_nodes_names:
            temp = dict()
            for key in coeffs[dicts].keys():
                if key != 'mean' and key != 'std':
                    if key in self.subunit_nodes_names:
                        temp['_' + key] = coeffs[dicts][key]
                    else:
                        temp[key] = coeffs[dicts][key]
                else:
                    temp[key] = coeffs[dicts][key]

            self.coeffs['_' + dicts] = temp
        for dicts in self.unit_nodes:
            temp = dict()
            for key in coeffs[dicts].keys():
                if key != 'mean' and key != 'std':
                    temp[key] = coeffs[dicts][key]
                else:
                    temp[key] = coeffs[dicts][key]

            self.coeffs[dicts] = temp

        if is_empty(self.aggregator_functions):
            temp_aggregator_functions = dict()
            for (x, y) in self.edges:
                if x in self.subunit_nodes_names and y in self.unit_nodes:
                    temp_aggregator_functions[y]['_' + x] = lambda d: np.mean(np.array(list(d)))
        else:
            temp_aggregator_functions = self.aggregator_functions

        def create_lambda(node):
            return lambda d: logit_functor(d, str(node), self.coeffs, node in self.unit_nodes,
                                           temp_aggregator_functions)

        lambda_functions = {node: create_lambda(node) for node in self.nodes}
        self.node_function = deepcopy(lambda_functions)

    # more general models

    def additive_model(self, functions, randomness):
        # each node is a sum of the functions of its predecessors, where functions is a dictionary of functions indexed by the nodes, and for each edges subunit -> unit, function operates on set (in order to avoid using means everytime)
        # we can distinguish if random is everytime the same or depend of the nodes (type(random) == func or type(random) == dict (indexed by nodes))
        for dicts in self.subunit_nodes_names:
            temp = dict()
            for key in functions[dicts].keys():
                if key in self.subunit_nodes_names:
                    temp['_' + key] = functions[dicts][key]
                else:
                    temp[key] = functions[dicts][key]

            self.additive_functions['_' + dicts] = temp
        for dicts in self.unit_nodes:
            temp = dict()
            for key in functions[dicts].keys():
                if key in self.subunit_nodes_names:
                    temp['_' + key] = functions[dicts][key]
                else:
                    temp[key] = functions[dicts][key]
            self.additive_functions[dicts] = temp

        def create_lambda(node):
            return lambda d: additive_functor(d, str(node), self.additive_functions, node in self.unit_nodes,
                                              randomness)

        lambda_functions = {node: create_lambda(node) for node in self.nodes}
        self.node_function = deepcopy(lambda_functions)

    def random_model(self):
        # be careful, every unit could have different distributions (from whole different types of distributions to the same distribution with different parameters)
        # we ensure that the distributions.py are set, because we can use those distributions.py for intervention when we do not need to sample anything (Pearl's causality lvl 2)
        def create_lambda(node):
            return lambda d: random_functor(d, str(node), self.node_distribution, self.aggregator_functions)

        lambda_functions = {node: create_lambda(node) for node in self.nodes}
        self.node_function = deepcopy(lambda_functions)





















    # Sampling


    def sample_data(self):
        samples = {}  # 1 sample for each SCM
        for node in nx.topological_sort(self.cgm.dag):
            predecessors = self.predecessors[node]
            node_function = self.node_function[node]
            
            if node in self.unit_nodes:
                # Prepare data for GPU
                sample_size = len(self.sizes)
                d_samples = {k: cuda.to_device(v) for k, v in samples.items()}
                d_result = cuda.device_array((sample_size, max(self.sizes)))
                d_sizes = cuda.to_device(np.array(self.sizes))
                
                # Configure GPU grid
                threads_per_block = (16, 16)
                blocks_per_grid_x = (sample_size + (threads_per_block[0] - 1)) // threads_per_block[0]
                blocks_per_grid_y = (max(self.sizes) + (threads_per_block[1] - 1)) // threads_per_block[1]
                blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
                
                # Launch GPU kernel
                sample_unit_node_gpu[blocks_per_grid, threads_per_block](node, predecessors, node_function, d_samples, d_result, d_sizes)
                
                # Copy result back to host
                samples[node] = d_result.copy_to_host()
            else:
                # Subunit node processing
                sample_size = len(samples[list(samples.keys())[0]])
                d_samples = {k: cuda.to_device(v) for k, v in samples.items()}
                d_result = cuda.device_array(sample_size)
                
                # Configure GPU grid
                threads_per_block = 256
                blocks_per_grid = (sample_size + (threads_per_block - 1)) // threads_per_block
                
                # Launch GPU kernel
                sample_subunit_node_gpu[blocks_per_grid, threads_per_block](node, predecessors, node_function, d_samples, d_result)
                
                # Copy result back to host
                samples[node] = d_result.copy_to_host()
        
        self.data = samples
        return samples



































        ######################

    # data related methods
    ######################

    def load_and_clean_data(self, data):
        # we assume data is in pandas Dataframe format
        self.data = cleaner(data)

    def load_data(self, data):
        # we assume data is cleaned and in the same format as dictionary of samples indexed by nodes, unit_number, subunit_sample without prefix '_'
        # we clean data first
        self.data = data
        # to dev ---> we should update the sizes for each unit and subunits

    def set_distribution_from_data(self):
        # we don't automatically load the data
        # we assume sizes are already set
        # we assume the data is already cleaned
        for node in self.unit_nodes:
            self.node_distribution[node] = lambda d, random_sample: EmpiricalDistribution(
                {self.data[node + i] for i in range(len(self.sizes))}).ppf(random_sample)

        for node in self.subunit_nodes_names:
            # we should use d to distinguish between every distributions in every units,
            self.node_distribution['_' + node] = lambda d, random_sample: distribution_functor(d, random_sample,
                                                                                               self.data, node,
                                                                                               self.sizes)

















































    # Plotting

    def plot_data(self):
        # we plot the data for each node
        s = []
        for node in self.unit_nodes: # better things can be done ..
            print(node, )
            ax, fig = plt.subplots()
            plt.hist([self.data[node + str(i)] for i in range(len(self.sizes))], bins='auto', alpha=0.7, color='r')
            plt.xlabel('distribution of ' + node)
            #plt.xlim(min([self.data[node + str(i)] for i in range(len(self.sizes))]),max([self.data[node + str(i)] for i in range(len(self.sizes))]))
            plt.show()
            s.append([self.data[node + str(i)] for i in range(len(self.sizes))])
        return s

    def plot_subunit_distributions(self, ):
        # we plot histogram for distribution of each unit_node, and for each subunit_node in every unit (as schools)
        for node in self.subunit_nodes:
            for i in range(len(self.sizes)):
                plt.hist([self.data[node + str(i) + '_' + str(j)] for j in range(self.sizes[i])], bins='auto',alpha=0.7, color='r')
                plt.xlabel('distribution of ' + node + ' in unit ' + str(i))
                plt.show()
        return

    def plot_unit_distributions(self):
        # we plot histogram for distribution of each unit_node
        for node in self.unit_nodes:
            plt.hist([self.data[node + str(i)] for i in range(len(self.sizes))], bins='auto', alpha=0.7, color='r')
            plt.xlabel('distribution of ' + node)
            plt.show()
        return








































    # to dev
    # to do -> randomness uniformize notation

    def ate(self, treatment, outcome, data):
        return

    def clean_data(self):
        return

    def asymptotic(self, n_unit, n_subunit):
        temp = self.sizes
        self.sizes = [n_subunit for _ in range(n_unit)]
        self.sample_data()
        self.sizes = temp

    def true_effect(self, node, distribution):
        self.node_distribution[node] = distribution

    def estimate_effect(self, node, distribution):
        self.node_distribution[node] = distribution

    def regression_estimate(self, node, distribution):
        self.node_distribution[node] = distribution

    def estimate_q(self, node, distribution):
        self.node_distribution[node] = distribution

    def augment(self, variable):

        return

    def collapse(self):
        nodes = copy(self.unit_nodes)
        edges = copy(self.edges)

        for node in self.subunit_nodes:
            nodes.add('Q_' + node)
        for edge in self.edges:
            if edge[1] in self.subunit_nodes:
                edges.remove((edge[0], edge[1]))
                if edge[0] in self.subunit_nodes:
                    edges.add(('Q_' + edge[0], 'Q_' + edge[1]))
                else:
                    edges.add((edge[0], 'Q_' + edge[1]))

        graph = CausalGraphicalModel(nodes=nodes, edges=edges)
        return graph  # return a cgm, where each subunit node is replaced by a unique node for simplicity and dataviz
