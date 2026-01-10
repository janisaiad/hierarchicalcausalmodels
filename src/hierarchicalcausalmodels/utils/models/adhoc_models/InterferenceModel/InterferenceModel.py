from causalgraphicalmodels.cgm import CausalGraphicalModel


class InterferenceUnitModel:
    def __init__(self, nodes, edges, prior, law_a, law_b, law_y, sizes):
        self.cgm = CausalGraphicalModel(nodes=nodes, edges=edges)
        self.prior = prior  # a list of prior for each unit
        self.law_a = law_a  # take an argument and return a random number with a as a parameter
        self.law_y = law_y
        self.law_b = law_b
        self.sizes = sizes

    def sample(self):
        nb_units = self.sizes[0]
        nb_subunits = self.sizes[1]
        data = {}
        for k in range(nb_units):
            data['U' + str(k)] = self.prior()  # we sample the prior
            for j in range(nb_subunits):
                data['A' + str(k) + '_' + str(j)] = self.law_a(
                    data['U' + str(k)])  # we use the unit data to sample a disbution

            data['B' + str(k)] = self.law_b(
                [data['U' + str(k)]] + [data['A' + str(k) + '_' + str(j)] for j in range(nb_subunits)])

            for j in range(nb_subunits):
                data['Y' + str(k) + '_' + str(j)] = self.law_y(data['A' + str(k) + '_' + str(j)], data[
                    'U' + str(k)], data['B'+str(k)])  # we use the subunit data for this example
        return data
