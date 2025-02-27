import math
import torch
import torchvision
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
from torch import nn
import torch.nn.functional as F


class ProteinEmbedding(nn.Module):
    def __init__(self, protein_size, acid_size, cnn_hidden, pool_size):
        # Initialize the ProteinEmbedding module
        super(ProteinEmbedding, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=acid_size, out_channels=cnn_hidden, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.biGRU = nn.GRU(cnn_hidden, cnn_hidden, bidirectional=True, batch_first=True, num_layers=1)
        self.maxpool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(math.floor(protein_size / pool_size), 256)
        self.dropout = nn.Dropout(p=0.5)  # Add Dropout

    def forward(self, x):
        # Forward pass of the ProteinEmbedding module
        x_1 = x.transpose(1, 2)
        x_1 = self.conv1d(x_1.float())
        x_1 = self.bn1(x_1)
        x_1 = self.maxpool1d(x_1)
        x_1 = x_1.transpose(1, 2)
        x_1, _ = self.biGRU(x_1)
        x_1 = self.global_avgpool1d(x_1)
        x_1 = x_1.squeeze()
        x = self.fc1(x_1)
        x = self.dropout(x)  # Add Dropout
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize the Encoder module
        super(Encoder, self).__init__()
        self.sageconv1 = SAGEConv(input_dim, hidden_dim, 'pool')
        self.sageconv2 = SAGEConv(hidden_dim, output_dim, 'pool')


    def forward(self, blocks, x):
        # Forward pass of the Encoder module
        h = self.sageconv1(blocks[0], x, blocks[0].edata['weight'])
        h = F.relu(h)
        h = self.sageconv2(blocks[1], h, blocks[1].edata['weight'])
        h = F.relu(h)
        return h


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize the Decoder module
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        # Forward pass of the Decoder module
        x = F.relu(self.fc1(h))
        x_reconstructed = self.fc2(x)
        return x_reconstructed


class Classifier(nn.Module):
    def __init__(self, hidden_dim):
        # Initialize the Classifier module
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=0.5)  # Add Dropout

    def forward(self, h):
        # Forward pass of the Classifier module
        h = self.dropout(h)  # Add Dropout
        logits = self.fc1(h)
        return logits


class ContrastiveSampler:
    def __init__(self, noise_level=0.1):
        # Initialize the ContrastiveSampler
        self.noise_level = noise_level

    def generate_gaussian_view(self, features):
        # Generate a Gaussian-perturbed view of the input features
        noise = torch.randn_like(features) * self.noise_level
        return features + noise

    def get_positive_samples(self, features_g):
        # Get positive samples
        return features_g

    def get_negative_samples(self, features, current_indices):
        # Get negative samples
        batch_size = features.shape[0]
        negative_samples = []

        for i in range(batch_size):
            # Get all nodes in the current batch as negative samples, excluding the current node
            negative_indices = torch.cat((torch.arange(0, i), torch.arange(i + 1, batch_size)))

            negative_samples.append(features[negative_indices])

        # Stack the negative samples for each node into a single tensor
        return torch.stack(negative_samples)


class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize the Projector module
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, h):
        # Forward pass of the Projector module
        h = F.relu(self.bn(self.fc1(h)))
        h = self.fc2(h)
        return h


class Model(nn.Module):
    def __init__(self, protein_size, acid_size, gnn_in_size, hidden_size, gnn_out_size,
                 pool_size=3, cnn_hidden=1, noise_level, num_negatives, temperature):
        # Initialize the Model
        super(Model, self).__init__()

        self.protein_embedding = ProteinEmbedding(protein_size, acid_size, cnn_hidden, pool_size)
        self.encoder = Encoder(gnn_in_size, hidden_size, gnn_out_size)
        self.decoder = Decoder(gnn_out_size, hidden_size, gnn_in_size)
        self.classifier = Classifier(gnn_out_size)

        # Add Projector for contrastive learning
        self.projector = Projector(gnn_out_size, hidden_size, gnn_out_size)

        self.contrastive_sampler = ContrastiveSampler(noise_level)
        self.temperature = temperature

    def info_nce_loss(self, query, positive_key, negative_keys):
        # Compute the InfoNCE loss
        positive_sim = F.cosine_similarity(query, positive_key, dim=1)
        positive_sim = torch.exp(positive_sim / self.temperature)

        negative_sim = F.cosine_similarity(query.unsqueeze(1), negative_keys, dim=2)
        negative_sim = torch.exp(negative_sim / self.temperature)

        nce_denom = positive_sim + negative_sim.sum(dim=1)

        nce_loss = -torch.log(positive_sim / nce_denom).mean()

        return nce_loss

    def forward(self, blocks, x, additional_features):
        # Forward pass of the Model
        x = self.protein_embedding(x)
        additional_features = self.protein_embedding(additional_features)

        # Generate Gaussian noise view
        x_g = self.contrastive_sampler.generate_gaussian_view(x)

        # Pass original features through the encoder
        h = self.encoder(blocks, x)
        h_g = self.encoder(blocks, x_g)

        # Apply projector before calculating contrastive loss
        h_projected = self.projector(h)
        h_g_projected = self.projector(h_g)

        # Get positive and negative samples, excluding the current node
        positives_h = self.contrastive_sampler.get_positive_samples(h_g_projected)
        negatives_h_from_h = self.contrastive_sampler.get_negative_samples(h_projected,
                                                                           torch.arange(h_projected.size(0)))
        negatives_h_g_from_h_g = self.contrastive_sampler.get_negative_samples(h_g_projected,
                                                                               torch.arange(h_g_projected.size(0)))

        # Combine negative samples from two views
        negatives_h = torch.cat([negatives_h_from_h, negatives_h_g_from_h_g], dim=1)

        x_reconstructed = self.decoder(h)
        logits = self.classifier(h)

        # Compute InfoNCE loss
        contrastive_loss = self.info_nce_loss(h_projected, positives_h, negatives_h)

        return x_reconstructed, logits, additional_features, contrastive_loss, h


def compute_loss(x, x_reconstructed, logits, labels, triplet_loss, lambda_param, triplet_lambda):
    # Compute the total loss
    mse_loss = F.mse_loss(x_reconstructed, x, reduction='mean')
    focal_loss = torchvision.ops.sigmoid_focal_loss(logits, labels, reduction='mean')
    loss = focal_loss + triplet_lambda * triplet_loss + lambda_param * mse_loss
    return loss
