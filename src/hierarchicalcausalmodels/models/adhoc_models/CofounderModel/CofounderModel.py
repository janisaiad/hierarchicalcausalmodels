from causalgraphicalmodels.cgm import CausalGraphicalModel


class CofounderUnitModel:
    def __init__(self, nodes, edges, prior, law_a, law_y, sizes):
        self.cgm = CausalGraphicalModel(nodes=nodes, edges=edges)
        self.prior = prior  # a list of prior for each unit
        self.law_a = law_a  # take an argument and return a random number with a as a parameter
        self.law_y = law_y
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
                data['Y' + str(k) + '_' + str(j)] = self.law_y(data['A' + str(k) + '_' + str(j)], data[
                    'U' + str(k)])  # we use the subunit data for this example
        return data
