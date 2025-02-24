# Get the data from the training set
# Including protein names and sequences, protein - protein interaction relationships, interaction confidence, virulence factors, and vectorization of protein sequences (pure one - hot and improved version)
# Get vecT5 code
# Process the initial protein sequences to obtain the matrix of protein sequences
# Create a class to store protein information. The protein objects are finally divided into training objects and test objects.
# There are two methods for matrix - converting proteins.
# In the main function, read information from different files according to the needs of the training set and test set to create strain information objects.
# Finally, make specific divisions in utils.

import string
import dgl
import numpy as np
import torch


def get_acid_embeddings_ctc():
    acid_embedding = {}
    file = open(r"vec5_CTC.txt", encoding="utf - 8")
    lines = file.readlines()
    for line in lines:
        temp_embedding = line.split("\t")
        acid_embedding[temp_embedding[0]] = temp_embedding[1].strip("\n")
    return acid_embedding


def get_acid_embeddings_one_hot():
    letters = list(string.ascii_uppercase)
    one_hot_dict = {}
    for i, letter in enumerate(letters):
        one_hot_vector = [0] * 26
        one_hot_vector[i] = 1
        one_hot_dict[letter] = one_hot_vector
    return one_hot_dict


def embeddings_normalization(protein_embedding, max_embedding_length, acid_dim):
    if len(protein_embedding) > max_embedding_length:
        return protein_embedding[:max_embedding_length]
    elif len(protein_embedding) < max_embedding_length:
        return np.concatenate((protein_embedding, np.zeros((max_embedding_length - len(protein_embedding), acid_dim))))
    return protein_embedding


def change_name_to_id(protein_sequences):
    i = 0
    protein_name_to_id = {}
    protein_id_to_name = {}
    for protein in protein_sequences:
        protein_name_to_id[protein] = i
        protein_id_to_name[i] = protein
        i += 1
    return protein_name_to_id, protein_id_to_name


class PathogenicBacteria:

    def __init__(self, protein_sequences, protein_links, virulence_factors):

        self.protein_sequences = protein_sequences
        self.protein_links = protein_links
        self.virulence_factors = virulence_factors

        self.features_matrix_ctc = None
        self.features_matrix_one_hot = None
        self.adj_matrix = None
        self.vf_labels = None
        self.num_protein = None
        self.protein_weight_matrix = None

    def protein_matrix_ctc(self):
        protein_num = len(self.protein_sequences)
        protein_embeddings = np.zeros((protein_num, 2000, 13))
        max_embedding_length = 2000

        acid_embeddings = get_acid_embeddings_ctc()
        for protein_name in self.protein_sequences:
            protein_embedding = []
            protein_sequence = self.protein_sequences[protein_name]
            for acid in protein_sequence:
                temp = acid_embeddings[acid]
                tem = np.array([float(x) for x in temp.split()])
                protein_embedding.append(tem)

            cur_protein = np.array(protein_embedding)
            acid_dim = cur_protein.shape[1]
            protein_name_to_id, protein_id_to_name = change_name_to_id(self.protein_sequences)
            p_id = protein_name_to_id[protein_name]

            protein_embeddings[p_id] = embeddings_normalization(cur_protein, max_embedding_length, acid_dim)

        self.features_matrix_ctc = np.array(protein_embeddings)

    def protein_matrix_one_hot(self):
        protein_num = len(self.protein_sequences)
        protein_embeddings = np.zeros((protein_num, 2000, 26))
        max_embedding_length = 2000

        acid_embeddings = get_acid_embeddings_one_hot()
        for protein_name in self.protein_sequences:
            protein_embedding = []
            protein_sequence = self.protein_sequences[protein_name]
            for acid in protein_sequence:
                temp = acid_embeddings[acid]
                # tem = np.array([float(x) for x in temp.split()])
                protein_embedding.append(temp)

            cur_protein = np.array(protein_embedding)
            acid_dim = cur_protein.shape[1]
            protein_name_to_id, protein_id_to_name = change_name_to_id(self.protein_sequences)
            p_id = protein_name_to_id[protein_name]

            protein_embeddings[p_id] = embeddings_normalization(cur_protein, max_embedding_length, acid_dim)

        self.features_matrix_one_hot = np.array(protein_embeddings)

    def protein_adj_weight_matrix(self):
        self.num_protein = len(self.protein_sequences)
        adj_dense = np.empty((self.num_protein, self.num_protein), dtype=int)
        adj_weight = np.empty((self.num_protein, self.num_protein), dtype=float)
        protein_name_to_id, protein_id_to_name = change_name_to_id(self.protein_sequences)

        for protein_link in self.protein_links:
            m = protein_name_to_id[protein_link[0]]
            n = protein_name_to_id[protein_link[1]]
            adj_dense[m][n] = 1
            adj_dense[n][m] = adj_dense[m][n]
            adj_weight[m][n] = protein_link[2] / 1000
            adj_weight[n][m] = adj_weight[m][n]

        self.adj_matrix = adj_dense
        self.protein_weight_matrix = adj_weight

    def set_vf_labels(self):
        vf_factors_labels = []
        for protein in self.protein_sequences:
            if protein in self.virulence_factors:
                vf_factors_labels.append(1)  # Virulence factors are labeled as 1
            else:
                vf_factors_labels.append(0)

        self.vf_labels = np.array(vf_factors_labels)

    def get_protein_matrix_ctc(self):
        self.protein_matrix_ctc()
        return self.features_matrix_ctc

    def get_protein_matrix_one_hot(self):
        self.protein_matrix_one_hot()
        return self.features_matrix_one_hot

    def get_adj_weight_matrix(self):
        self.protein_adj_weight_matrix()
        return self.adj_matrix, self.protein_weight_matrix

    def get_num_protein(self):
        self.num_protein = len(self.protein_sequences)
        return self.num_protein

    def get_vf_labels(self):
        self.set_vf_labels()
        return self.vf_labels


def protein_sequences_reader(file_name):
    max_length = 0
    file = open(file_name, encoding="utf - 8")
    protein_sequence = {}

    lines = file.readlines()
    temp_locus_tag_str = ""
    temp_protein_sequence = ""
    flag = 0
    for line in lines:
        if line.startswith(">"):
            if flag == 1:
                protein_sequence[locus_tag] = temp_protein_sequence
                if len(temp_protein_sequence) > max_length:
                    max_length = len(temp_protein_sequence)
            temp_locus_tag_str = line.split('.')
            # Get the locus tag of each protein
            locus_tag = temp_locus_tag_str[1].strip('\n')
            temp_protein_sequence = ""
            flag = 0
        elif line == lines[-1]:
            temp_protein_sequence += line.strip('\n')
            protein_sequence[locus_tag] = temp_protein_sequence
            if len(temp_protein_sequence) > max_length:
                max_length = len(temp_protein_sequence)
        else:
            temp_protein_sequence += line.strip('\n')
            flag = 1
    file.close()
    return protein_sequence


def new_protein_sequences_reader(protein_links, protein_sequences):
    new_protein_equences = {}
    for protein_link in protein_links:
        if protein_link[0] in protein_sequences.keys() and protein_link[1] in protein_sequences.keys():
            if protein_link[0] not in new_protein_equences.keys():
                new_protein_equences[protein_link[0]] = protein_sequences[protein_link[0]]
            if protein_link[1] not in new_protein_equences.keys():
                new_protein_equences[protein_link[1]] = protein_sequences[protein_link[1]]

    return new_protein_equences


def link_processing(file_name):
    file = open(file_name, encoding="utf - 8")
    lines = file.readlines()
    protein_links = []

    for line in lines:
        temp_sequence = line.split(" ")
        if int(temp_sequence[2].strip("\n")) >= 400:
            protein_link = []
            protein1 = temp_sequence[0].split('.')[1]
            protein2 = temp_sequence[1].split('.')[1]
            protein_link.append(protein1)
            protein_link.append(protein2)
            protein_link.append(int(temp_sequence[2].strip('\n')))
            protein_links.append(protein_link)
    file.close()
    return protein_links


def get_virulence_factors(file_name):
    file = open(file_name, encoding="utf - 8")
    lines = file.readlines()
    virulence_factor = []

    for line in lines:
        temp = line.strip("\n")
        virulence_factor.append(temp)
    file.close()
    return virulence_factor


def construct_graph(adj, features, edge_weight, labels):
    adj = torch.tensor(adj, dtype=torch.float32)
    src, dst = torch.nonzero(adj, as_tuple=True)
    graph = dgl.graph((src, dst))
    graph.ndata['features'] = torch.tensor(features, dtype=torch.float32)
    edge_weights = edge_weight[torch.nonzero(adj, as_tuple=True)].squeeze()
    graph.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)
    graph.ndata['label'] = torch.tensor(labels)

    return graph


def get_data():

    graph_list = []
    # Salmonella
    # salmonella_protein_sequences = protein_sequences_reader("Salmonella/99287.protein.sequences.v12.0.fa")
    # salmonella_protein_links = link_processing("Salmonella/99287.protein.links.v12.0.txt")
    # salmonella_vfs = get_virulence_factors("Salmonella/VFs.txt")
    # new_sal_sequences = new_protein_sequences_reader(salmonella_protein_links, salmonella_protein_sequences)
    # salmonella = PathogenicBacteria(new_sal_sequences, salmonella_protein_links, salmonella_vfs)
    # sal_adj, sal_weight = salmonella.get_adj_weight_matrix()
    # salmonella_graph = construct_graph(sal_adj, salmonella.get_protein_matrix_ctc(),
    #                                    sal_weight, salmonella.get_vf_labels())
    # graph_list.append(salmonella_graph)
    # Campylobacter jejuni
    # campylobacterjejuni_protein_sequences = protein_sequences_reader("Campylobacter_jejuni/192222.protein.sequences.v12.0.fa")
    # campylobacterjejuni_protein_links = link_processing("Campylobacter_jejuni/192222.protein.links.v12.0.txt")
    # campylobacterjejuni_vfs = get_virulence_factors("Campylobacter_jejuni/VFs.txt")
    # new_cam_sequences = new_protein_sequences_reader(campylobacterjejuni_protein_links, campylobacterjejuni_protein_sequences)
    # campylobacterjejuni = PathogenicBacteria(new_cam_sequences, campylobacterjejuni_protein_links,
    #                                          campylobacterjejuni_vfs)
    # cam_adj, cam_weight = campylobacterjejuni.get_adj_weight_matrix()
    # campylobacterjejuni_graph = construct_graph(cam_adj, campylobacterjejuni.get_protein_matrix_ctc(),
    #                                             cam_weight, campylobacterjejuni.get_vf_labels())
    # graph_list.append(campylobacterjejuni_graph)

    # Staphylococcus aureus
    staphylococcusaureus_protein_sequences = protein_sequences_reader("Staphylococcus_aureus/93061.protein.sequences.v12.0.fa")
    staphylococcusaureus_protein_links = link_processing("Staphylococcus_aureus/93061.protein.links.v12.0.txt")
    staphylococcusaureus_vfs = get_virulence_factors("Staphylococcus_aureus/VFs.txt")
    new_sta_sequences = new_protein_sequences_reader(staphylococcusaureus_protein_links, staphylococcusaureus_protein_sequences)
    staphylococcusaureus = PathogenicBacteria(new_sta_sequences,
                                              staphylococcusaureus_protein_links, staphylococcusaureus_vfs)
    sta_adj, sta_weight = staphylococcusaureus.get_adj_weight_matrix()
    staphylococcusaureus_graph = construct_graph(sta_adj, staphylococcusaureus.get_protein_matrix_ctc(),
                                                 sta_weight, staphylococcusaureus.get_vf_labels())
    graph_list.append(staphylococcusaureus_graph)

    # # Shigella
    # Shigella_protein_sequences = protein_sequences_reader(
    #     "Shigella/844609.protein.sequences.v12.0.fa")
    # Shigella_protein_links = link_processing("Shigella/844609.protein.links.v12.0.txt")
    # Shigella_vfs = get_virulence_factors("Shigella/VFs.txt")
    # new_shi_sequences = new_protein_sequences_reader(Shigella_protein_links,
    #                                                  Shigella_protein_sequences)
    # Shigella = PathogenicBacteria(new_shi_sequences, Shigella_protein_links, Shigella_vfs)
    # shi_adj, shi_weight = Shigella.get_adj_weight_matrix()
    # Shigella_graph = construct_graph(shi_adj, Shigella.get_protein_matrix_ctc(),
    #                                              shi_weight, Shigella.get_vf_labels())
    # graph_list.append(Shigella_graph)

    return graph_list

