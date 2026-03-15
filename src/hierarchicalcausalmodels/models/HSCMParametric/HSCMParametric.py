# We assume every transitions is parametric, and we can set the distributions.py for each node, and the functions for each node
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))




from copy import copy, deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from causalgraphicalmodels import CausalGraphicalModel

from hierarchicalcausalmodels.utils.distributions import Distribution,EmpiricalDistribution
from hierarchicalcausalmodels.utils.parsing_utils import source_sample
from hierarchicalcausalmodels.utils.utils import linear_functor, logit_functor, random_functor, additive_functor, \
    is_empty, distribution_functor, cleaner, extract_distributions_from_data


class HSCMParametric:
    def __init__(self, nodes: set, edges: set, unit_nodes: set, subunit_nodes: set, sizes: list, node_functions: dict,
                 data: dict,observed_nodes: set=None):
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
        if observed_nodes is not None:
            self.cgm.observed_variables = {node for node in self.nodes if self.observed_nodes[node]}
            self.cgm.unobserved_variables = {node for node in self.nodes if not self.observed_nodes[node]}
        else:
            self.cgm.observed_variables = nodes
        
        
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




        self.collapsed = CausalGraphicalModel(nodes=self.nodes, edges=self.edges)
        self.augmented = CausalGraphicalModel(nodes=self.nodes, edges=self.edges)
        self.marginalized = CausalGraphicalModel(nodes=self.nodes, edges=self.edges)

        self.params = dict()  # dictionary of parameters indexed by the nodes, list of parameters for each node, parameters can be a tuple
        self.node_experimental_distribution = dict()  # dictionary of distributions.py like indexed by the nodes to see and explore them
        self.node_theoretical_distribution = dict()  # dictionary of distributions.py like indexed by the nodes for interventions ! here is the key for parametric things






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
        # we assume that node is a sub
        # we can intervene on a node by setting its distribution to a new one
        self.node_theoretical_distribution[node] = distribution_object
        self.node_function[node] = distribution_object.ppf()
        return



    def hard_intervention(self, node, value):
        distribution_object = EmpiricalDistribution({value})
        self.node_theoretical_distribution[node] = distribution_object
        self.node_function[node] = distribution_object.ppf()
        return

# intervenes on a node by replacing its mechanism with a new distribution
# conditioned on its parents, i.e. node follows q*(node | parents(node))
# distribution_object is a callable taking a dict of parent values and returning a sample
def soft_conditional_intervention(self, node, distribution_object):
    former_distrib = self.node_theoretical_distribution.get(node)
    former_function = self.node_function.get(node)
    self.node_theoretical_distribution[node] = distribution_object
    self.node_function[node] = lambda d: distribution_object.ppf(d, np.random.uniform(0, 1))
    return former_distrib, former_function

    def set_soft_intervention(self, node, distribution_object):
        former_distrib = self.node_theoretical_distribution[node]
        self.node_theoretical_distribution[node] = distribution_object
        former_node_function = self.node_function[node]
        self.node_function[node] = lambda d: distribution_object.ppf(d, np.random.uniform(0, 1))
        return former_distrib, former_node_function
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
            
        def observed_node(self, node):
            self.observed_nodes[node] = True
            return

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
            if node in self.unit_nodes:
                for i in range(len(self.sizes)):  # we must distinguish between unit and subunit nodes
                    parent_samples = dict()
                    for parent in self.predecessors[node + str(i)]:
                        if isinstance(parent, frozenset):  # if parent is a subunit node and node is a unit_node
                            parent_samples[source_sample(list(parent)[0])] = {samples[parents] for parents in
                                                                              parent}  # if parent is a subunit node, we take a set of all values of the subunit node, and his name is in parent.keys()[0][:-3]
                        else:  # if parent is a unit node
                            parent_samples[parent] = samples[parent]
                    # print(parent_samples, 'parent_samples')
                    samples[node + str(i)] = self.node_function[node](parent_samples)
            else:
                for i in range(len(self.sizes)):
                    for j in range(self.sizes[i]):
                        parent_samples = {
                            parent: samples[parent]
                            for parent in self.predecessors[node + str(i) + '_' + str(j)]
                        }
                        # print(parent_samples, 'parent_samples')
                        samples[node + str(i) + '_' + str(j)] = self.node_function[node](parent_samples)
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
        for node in self.unit_nodes:  # better things can be done ..
            print(node, )
            plt.hist([self.data[node + str(i)] for i in range(len(self.sizes))], bins='auto', alpha=0.7, color='r')
            plt.xlabel('distribution of ' + node)
            plt.xlim(min([self.data[node + str(i)] for i in range(len(self.sizes))]),
                     max([self.data[node + str(i)] for i in range(len(self.sizes))]))
            plt.show()
            s.append([self.data[node + str(i)] for i in range(len(self.sizes))])
        return s

    def plot_subunit_distributions(self, ):
        # we plot histogram for distribution of each unit_node, and for each subunit_node in every unit (as schools)
        for node in self.subunit_nodes:
            for i in range(len(self.sizes)):
                plt.hist([self.data[node + str(i) + '_' + str(j)] for j in range(self.sizes[i])], bins='auto',
                         alpha=0.7, color='r')
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

    def collapse(self):
        nodes = copy(self.unit_nodes)
        edges = copy(self.edges)
        subunit_to_unit = {}

        for subunit in self.subunit_nodes:
            # Create a new unit endogenous variable for each subunit variable
            new_unit = 'Q_' + subunit
            nodes.add(new_unit)
            subunit_to_unit[subunit] = new_unit

            # Observing variable to add ! we suppose that every variable are observed

            # Disconnect the unit parents from the subunit variable and connect them to the new unit endogenous variable
            for parent, child in list(edges):
                if child == subunit:
                    edges.remove((parent, child))
                    if parent in self.unit_nodes:
                        edges.add((parent, new_unit))
                    elif parent in self.subunit_nodes:
                        edges.add((subunit_to_unit[parent], new_unit))

            # Connect the new unit endogenous variable to each direct unit descendant
            for parent, child in list(edges):
                if parent == subunit:
                    edges.remove((parent, child))
                    edges.add((new_unit, child))

            # Erase the subunit variable
            nodes.remove(subunit)

        # Erase the inner plate (if applicable)
        # Add logic to erase the inner plate if needed

        graph = CausalGraphicalModel(nodes=nodes, edges=edges)
        self.collapsed = graph
        return graph # this is a HCGM


    def is_identifiable(self, treatment, outcome, data): # to optimize
        # we have to check if there is no subunit level cofounder
        boolean = True
        # to optimize because it's super bad, better structure to be used (predecessors to be affined)
        for node in self.subunit_nodes_names:
            for parent in self.predecessors[node]:
                if parent in self.subunit_nodes_names:
                    for edge in self.edges:
                        if edge[0] == parent and edge[1] != node :
                            boolean = False


        return


    def marginalize(self):

        return

    # we have to well define what is doable in a hierarchical causal model

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










    def augment(self, augmentation_variable, mechanism):
        level = mechanism['level']
        parents = mechanism.get('parents', [])
        children = mechanism.get('children', [])
        function = mechanism.get('function', lambda d: 0.0)

        is_subunit = (level == 'subunit')
        internal_name = '_' + augmentation_variable if is_subunit else augmentation_variable

        self.nodes.add(internal_name)
        if is_subunit:
            self.subunit_nodes.add(internal_name)
            self.subunit_nodes_names.add(augmentation_variable)
        else:
            self.unit_nodes.add(augmentation_variable)
            self.aggregator_functions[augmentation_variable] = {}

        def _int(name):
            return '_' + name if name in self.subunit_nodes_names else name

        for p in parents:
            self.edges.add((_int(p), internal_name))
        for c in children:
            self.edges.add((internal_name, _int(c)))

        if is_subunit:
            for i in range(len(self.sizes)):
                for j in range(self.sizes[i]):
                    key = internal_name + str(i) + '_' + str(j)
                    self.predecessors[key] = set()
                    for p in parents:
                        if p in self.subunit_nodes_names:
                            self.predecessors[key].add(_int(p) + str(i) + '_' + str(j))
                        else:
                            self.predecessors[key].add(p + str(i))
        else:
            for i in range(len(self.sizes)):
                key = augmentation_variable + str(i)
                self.predecessors[key] = set()
                for p in parents:
                    if p in self.subunit_nodes_names:
                        self.predecessors[key].add(frozenset(_int(p) + str(i) + '_' + str(j) for j in range(self.sizes[i])))
                    else:
                        self.predecessors[key].add(p + str(i))

        for c in children:
            if c in self.subunit_nodes_names:
                for i in range(len(self.sizes)):
                    for j in range(self.sizes[i]):
                        c_key = _int(c) + str(i) + '_' + str(j)
                        if is_subunit:
                            self.predecessors[c_key].add(internal_name + str(i) + '_' + str(j))
                        else:
                            self.predecessors[c_key].add(augmentation_variable + str(i))
            else:
                for i in range(len(self.sizes)):
                    if is_subunit:
                        self.predecessors[c + str(i)].add(frozenset(internal_name + str(i) + '_' + str(j) for j in range(self.sizes[i])))
                    else:
                        self.predecessors[c + str(i)].add(augmentation_variable + str(i))

        if is_subunit:
            for c in children:
                if c in self.unit_nodes:
                    self.aggregator_functions[c][internal_name] = lambda d: np.mean(np.array(list(d)))

        self.node_function[internal_name] = function
        self.cgm = CausalGraphicalModel(nodes=self.nodes, edges=self.edges)



def is_identifiable(self, treatment, outcome):
    collapsed_graph = self.collapse()

    # treatment and outcome in collapsed graph use unit names directly
    # Q_ prefix for collapsed subunit nodes
    t = treatment if treatment in self.unit_nodes else 'Q__' + treatment
    o = outcome if outcome in self.unit_nodes else 'Q__' + outcome

    # check if a valid backdoor adjustment set exists
    adjustment_sets = collapsed_graph.get_all_backdoor_adjustment_sets(t, o)

    if adjustment_sets:
        return True, adjustment_sets
    else:
        return False, set()


def marginalize(self, variables=None):
    # if no variables specified, marginalize all subunit nodes
    if variables is None:
        variables = set(self.subunit_nodes_names)

    nodes = set(self.unit_nodes)
    edges = set()

    # only keep unit-level edges, rewiring through marginalized subunit nodes
    for node in self.unit_nodes:
        for parent, child in self.edges:
            # unit -> unit edges, keep as is
            if parent in self.unit_nodes and child in self.unit_nodes:
                edges.add((parent, child))
            # subunit -> unit: connect subunit's unit parents directly to the unit child
            if '_' + parent in self.subunit_nodes and child in self.unit_nodes and parent in variables:
                for grandparent, p in self.edges:
                    if p == '_' + parent and grandparent in self.unit_nodes:
                        edges.add((grandparent, child))
            # unit -> subunit -> unit chain
            if parent in self.unit_nodes and '_' + child in self.subunit_nodes and child in variables:
                for p2, c2 in self.edges:
                    if p2 == '_' + child and c2 in self.unit_nodes:
                        edges.add((parent, c2))

    # deduplicate and remove self-loops
    edges = {(p, c) for p, c in edges if p != c}

    graph = CausalGraphicalModel(nodes=nodes, edges=edges)
    self.marginalized = graph
    return graph
# Estimates the Average Treatment Effect (ATE) of treatment on outcome.
# Uses backdoor adjustment: check identifiability, pick the smallest valid
# adjustment set, then estimate E[Y|do(T=1)] - E[Y|do(T=0)] by sampling
# adjustment variables from the observational distribution and intervening
# on treatment directly via node_function. Adapted to the unit-level structure.
def ate(self, treatment, outcome, n_samples=1000):
    # check identifiability first
    identifiable, adj_sets = self.is_identifiable(treatment, outcome)
    if not identifiable:
        print("Effect is not identifiable, no valid adjustment set found.")
        return None

    # pick the smallest adjustment set
    adjustment_set = min(adj_sets, key=len)

    # internal names
    t = treatment if treatment in self.unit_nodes else '_' + treatment
    o = outcome if outcome in self.unit_nodes else '_' + outcome

    # estimate E[Y | do(T=1)] - E[Y | do(T=0)] by backdoor adjustment
    results = {0: [], 1: []}

    for t_val in [0, 1]:
        for _ in range(n_samples):
            # sample the adjustment variables from the model
            sample = self.sample_data()

            # collect adjustment variable values across units
            adj_values = {}
            for adj in adjustment_set:
                adj_int = adj if adj in self.unit_nodes else '_' + adj
                adj_values[adj_int] = [sample[adj_int + str(i)] for i in range(len(self.sizes))]

            # intervene on treatment for each unit
            y_vals = []
            for i in range(len(self.sizes)):
                parent_samples = {t: t_val}
                for adj in adjustment_set:
                    adj_int = adj if adj in self.unit_nodes else '_' + adj
                    parent_samples[adj_int] = adj_values[adj_int][i]
                y_vals.append(self.node_function[o](parent_samples))

            results[t_val].append(np.mean(y_vals))

    ate_value = np.mean(results[1]) - np.mean(results[0])
    return ate_value

# Estimates the interventional distribution Q = P(Y|do(T=t)) by intervening
# on treatment and sampling outcome values across all units.
def estimate_q(self, treatment, outcome, t_val, n_samples=1000):
    t = treatment if treatment in self.unit_nodes else '_' + treatment
    o = outcome if outcome in self.unit_nodes else '_' + outcome

    y_samples = []
    for _ in range(n_samples):
        sample = self.sample_data()
        for i in range(len(self.sizes)):
            parent_samples = {parent: sample[parent] for parent in self.predecessors[o + str(i)]}
            parent_samples[t + str(i)] = t_val
            y_samples.append(self.node_function[o](parent_samples))

    return np.array(y_samples)


# Estimates the Conditional ATE E[Y|do(T=1),C=c] - E[Y|do(T=0),C=c]
# for a given conditioning variable and value.
# Builds on estimate_q, filtering samples where the condition holds.
def cate(self, treatment, outcome, condition_node, condition_value, n_samples=1000, tol=0.1):
    c = condition_node if condition_node in self.unit_nodes else '_' + condition_node
    t = treatment if treatment in self.unit_nodes else '_' + treatment
    o = outcome if outcome in self.unit_nodes else '_' + outcome

    results = {0: [], 1: []}
    for t_val in [0, 1]:
        for _ in range(n_samples):
            sample = self.sample_data()
            for i in range(len(self.sizes)):
                # only keep samples where condition is satisfied
                if abs(sample[c + str(i)] - condition_value) > tol:
                    continue
                parent_samples = {parent: sample[parent] for parent in self.predecessors[o + str(i)]}
                parent_samples[t + str(i)] = t_val
                results[t_val].append(self.node_function[o](parent_samples))

    if not results[0] or not results[1]:
        print("Not enough samples satisfying the condition, try increasing n_samples or tol.")
        return None

    return np.mean(results[1]) - np.mean(results[0])

# Plots the causal DAG with unit nodes in blue and subunit nodes in red.
# Optionally plots the collapsed or marginalized graph instead.
def plot_causal_graph(self, which='full'):
    if which == 'collapsed':
        graph = self.collapsed.dag
        title = 'Collapsed graph'
    elif which == 'marginalized':
        graph = self.marginalized.dag
        title = 'Marginalized graph'
    else:
        graph = self.cgm.dag
        title = 'Full causal graph'

    color_map = []
    for node in graph.nodes():
        if node in self.unit_nodes:
            color_map.append('steelblue')
        else:
            color_map.append('salmon')

    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=True, node_color=color_map,
            node_size=1500, font_size=10, font_color='white',
            arrows=True, arrowsize=20)
    plt.title(title)
    plt.show()



# estimates causal effect of treatment on outcome using regression adjustment
# fits a linear model on sampled data: outcome ~ treatment + adjustment variables
# returns the coefficient of treatment as the effect estimate
def regression_estimate(self, treatment, outcome, n_samples=500):
    identifiable, adj_sets = self.is_identifiable(treatment, outcome)
    if not identifiable:
        print("Effect is not identifiable.")
        return None

    adjustment_set = min(adj_sets, key=len)
    t = treatment if treatment in self.unit_nodes else '_' + treatment
    o = outcome if outcome in self.unit_nodes else '_' + outcome

    X, Y = [], []
    for _ in range(n_samples):
        sample = self.sample_data()
        for i in range(len(self.sizes)):
            row = [sample[t + str(i)]]
            for adj in adjustment_set:
                adj_int = adj if adj in self.unit_nodes else '_' + adj
                row.append(sample[adj_int + str(i)])
            X.append(row)
            Y.append(sample[o + str(i)])

    X, Y = np.array(X), np.array(Y)
    # add intercept
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    # OLS: beta = (X'X)^-1 X'Y
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    # coefficient at index 1 is the treatment effect
    return beta[1]



# validates and normalizes self.data keys to match internal naming convention
# checks that all expected unit and subunit keys are present
# removes any keys that don't match the expected format
def clean_data(self):
    expected_keys = set()
    for node in self.unit_nodes:
        for i in range(len(self.sizes)):
            expected_keys.add(node + str(i))
    for node in self.subunit_nodes:
        for i in range(len(self.sizes)):
            for j in range(self.sizes[i]):
                expected_keys.add(node + str(i) + '_' + str(j))

    missing = expected_keys - set(self.data.keys())
    extra = set(self.data.keys()) - expected_keys

    if missing:
        print("Missing keys in data:", missing)
    if extra:
        print("Removing unexpected keys:", extra)
        for key in extra:
            del self.data[key]

    return missing, extra