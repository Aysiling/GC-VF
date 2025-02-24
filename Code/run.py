import random
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, recall_score, precision_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm
import dgl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from utils import load_graph, find_best_threshold
from model_new import Model, compute_loss
import torchvision

# Set random seed
seed = 1
dgl.random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Learning rate
lr = 1e-3
# Total number of epochs
num_epoch = 131
auc_test_rounds = 100
# Weight decay
weight_decay = 1e-4
# Hyperparameter λ
lambda_param = 1.0
# Hyperparameter for contrastive learning
contrastive_lambda = 1.0
noise_level = 0.1
temperature = 0.07

# Create the model and move it to the GPU
model = Model(2000, 13, 256, 128, 64, noise_level=noise_level, temperature=temperature).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
sampler = dgl.dataloading.NeighborSampler([4, 4])

# Load and merge graphs
graphs = load_graph()
full_graph = dgl.batch(graphs).to(device)  # Ensure the graph is on the GPU
num_nodes = full_graph.num_nodes()

# Get labels
labels = full_graph.ndata['label'].cpu().numpy()

# Stratified split into training, validation, and test sets
train_val_indices, test_indices = train_test_split(np.arange(num_nodes), test_size=0.2, stratify=labels,
                                                   random_state=seed)
train_indices, val_indices = train_test_split(train_val_indices, test_size=0.25, stratify=labels[train_val_indices],
                                              random_state=seed)

# Train
with tqdm(total=num_epoch) as pbar:
    pbar.set_description('Training')
    best_val_auc = float('-inf')
    best_model = None

    for epoch in range(num_epoch):
        model.train()
        train_nids = torch.from_numpy(train_indices).to(torch.int64).to(device)
        subgraph_loader = dgl.dataloading.DataLoader(full_graph, train_nids, sampler, batch_size=300, shuffle=True)

        total_loss = 0
        total_mse_loss = 0
        total_focal_loss = 0
        total_contrastive_loss = 0
        count = 0

        for input_nodes, output_nodes, blocks in subgraph_loader:
            blocks = [block.to(device) for block in blocks]  # Move subgraphs to the GPU
            input_features = blocks[0].srcdata['features']
            output_labels = blocks[-1].dstdata['label']
            output_features = blocks[-1].dstdata['features']
            x_reconstructed, logits, x, contrastive_loss = model(blocks, input_features, output_features)
            loss = compute_loss(x_reconstructed, x, logits.squeeze(), output_labels.float(), contrastive_loss,
                                lambda_param, contrastive_lambda)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            total_mse_loss += F.mse_loss(x_reconstructed, x, reduction='mean').item()
            total_focal_loss += torchvision.ops.sigmoid_focal_loss(logits.squeeze(), output_labels.float(),
                                                                   reduction='mean').item()
            total_contrastive_loss += contrastive_loss.item()
            count += 1

        mean_train_loss = total_loss / count
        mean_mse_loss = total_mse_loss / count
        mean_focal_loss = total_focal_loss / count
        mean_contrastive_loss = total_contrastive_loss / count

        if epoch % 10 == 0 and epoch != 0:
            # Validation
            model.eval()
            val_nids = torch.from_numpy(val_indices).to(torch.int64).to(device)
            subgraph_loader = dgl.dataloading.DataLoader(full_graph, val_nids, sampler, batch_size=300, shuffle=True)

            total_val_loss = 0
            count = 0
            true_labels = []
            pred_scores = []

            with torch.no_grad():
                for input_nodes, output_nodes, blocks in subgraph_loader:
                    blocks = [block.to(device) for block in blocks]  # Move subgraphs to the GPU
                    input_features = blocks[0].srcdata['features']
                    output_labels = blocks[-1].dstdata['label']
                    output_features = blocks[-1].dstdata['features']
                    x_reconstructed, logits, x, contrastive_loss = model(blocks, input_features, output_features)
                    val_loss = compute_loss(x_reconstructed, x, logits.squeeze(), output_labels.float(),
                                            contrastive_loss, lambda_param, contrastive_lambda)

                    total_val_loss += val_loss.item()
                    count += 1

                    pred_scores.extend(torch.sigmoid(logits).cpu().numpy())
                    true_labels.extend(output_labels.cpu().numpy())

            mean_val_loss = total_val_loss / count
            val_auc = roc_auc_score(true_labels, pred_scores)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model = model.state_dict()

            pbar.set_postfix(val_loss=mean_val_loss, val_auc=val_auc)
        else:
            pbar.set_postfix(train_mse_loss=mean_mse_loss, train_focal_loss=mean_focal_loss)

        pbar.update(1)

# Save the best model
torch.save(best_model, 'best_model.pkl')

# Test
model.load_state_dict(torch.load('best_model.pkl'))
model.eval()

test_nids = torch.from_numpy(test_indices).to(torch.int64).to(device)

# Variables for calculating results
accuracy_full_rounds = []
sensitivity_full_rounds = []  # Sensitivity (Sn)
specificity_full_rounds = []  # Specificity (Sp)
precision_full_rounds = []  # Precision
f1_full_rounds = []  # F1 score
auc_pr_full_rounds = []  # AUPRC
auc_roc_full_rounds = []  # AUROC
mcc_full_rounds = []  # MCC

# Testing phase
with tqdm(total=auc_test_rounds) as pbar_test:
    pbar_test.set_description('Testing')

    for round in range(auc_test_rounds):
        subgraph_loader = dgl.dataloading.DataLoader(full_graph, test_nids, sampler, batch_size=300, shuffle=True)

        true_labels = []
        pred_scores = []

        for input_nodes, output_nodes, blocks in subgraph_loader:
            blocks = [block.to(device) for block in blocks]  # Move subgraphs to the GPU
            input_features = blocks[0].srcdata['features']
            output_labels = blocks[-1].dstdata['label']
            output_features = blocks[-1].dstdata['features']

            with torch.no_grad():
                x_reconstructed, logits, additional_features, _ = model(blocks, input_features, output_features)
                pred_scores.extend(torch.sigmoid(logits).cpu().numpy())
                true_labels.extend(output_labels.cpu().numpy())

        best_threshold = find_best_threshold(true_labels, pred_scores)
        pred_labels = (np.array(pred_scores) >= best_threshold).astype(int)

        # Calculate various metrics
        auc_roc_full_rounds.append(roc_auc_score(true_labels, pred_scores))
        precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
        auc_pr_full_rounds.append(auc(recall, precision))  # AUPRC

        sensitivity_full_rounds.append(recall_score(true_labels, pred_labels))  # Sensitivity (Sn)
        specificity_full_rounds.append(recall_score(true_labels, pred_labels, pos_label=0))  # Specificity (Sp)
        precision_full_rounds.append(precision_score(true_labels, pred_labels))  # Precision
        f1_full_rounds.append(f1_score(true_labels, pred_labels))  # F1 score
        accuracy_full_rounds.append(accuracy_score(true_labels, pred_labels))  # Accuracy
        mcc_full_rounds.append(matthews_corrcoef(true_labels, pred_labels))  # MCC

        pbar_test.update(1)

# Calculate average metrics
accuracy = np.mean(accuracy_full_rounds)
sensitivity = np.mean(sensitivity_full_rounds)
specificity = np.mean(specificity_full_rounds)
precision = np.mean(precision_full_rounds)
f1 = np.mean(f1_full_rounds)
auc_pr = np.mean(auc_pr_full_rounds)
auc_roc = np.mean(auc_roc_full_rounds)
mcc = np.mean(mcc_full_rounds)

# Output results
print(f"Accuracy: {accuracy:.4f}, Sensitivity (Sn): {sensitivity:.4f}, Specificity (Sp): {specificity:.4f}, "
      f"Precision: {precision:.4f}, F1 Score: {f1:.4f}, AUPRC: {auc_pr:.4f}, AUROC: {auc_roc:.4f}, MCC: {mcc:.4f}")
