import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers, StronglyEntanglingLayers

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

SAVE_PATH = "/Users/soardr/FLIQ/models/"

train_dataset = pd.read_csv("Datasets/drug+induced+autoimmunity+prediction/DIA_trainingset_RDKit_descriptors.csv")
test_dataset = pd.read_csv("Datasets/drug+induced+autoimmunity+prediction/DIA_testset_RDKit_descriptors.csv")

X = train_dataset.columns[113:]
X = train_dataset[X]

X_test = test_dataset.columns[113:]
X_test = test_dataset[X_test]

n_train = X.shape[0]
n_test = X_test.shape[0]

def load_quantum_embeddings():
    q_train_images = np.load(SAVE_PATH + "q_train_images__6_3.npy")
    q_test_images = np.load(SAVE_PATH + "q_test_images__6_3.npy")

    return q_train_images, q_test_images

y_train = train_dataset['Label']
y_test = test_dataset['Label']

# q_train_images, q_test_images = get_and_save_embeddings()
q_train_images, q_test_images = load_quantum_embeddings()

X_train, X_val, Y_train, Y_val = train_test_split(q_train_images, y_train, 
                                                test_size=0.2, 
                                                shuffle=True, 
                                                stratify=y_train, 
                                                random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val.values, dtype=torch.long)

X_test_tensor = torch.tensor(q_test_images, dtype=torch.float32)
Y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

window_size = 6
n_layers = 3

dev = qml.device("default.qubit", wires=window_size)
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, window_size))

@qml.qnode(dev)
def circuit(phi):
    for j in range(window_size):
        qml.RY(np.pi * int(phi[j]) / 10, wires=j)
    
    # RandomLayers(rand_params, wires=list(range(window_size)))
    StronglyEntanglingLayers(rand_params, wires=list(range(window_size)))

    return [qml.expval(qml.PauliZ(j)) for j in range(window_size)]

def quanv_fixed(fr_vector, stride=1):
    fr_vector_shape = fr_vector.shape
    # print("fr_vector_shape:", fr_vector_shape)
    # assert fr_vector_shape == (1, 85)  # (batch_size, 1, 85)

    # out = np.zeros((14, 14, 4))
    out = np.zeros((85 - stride + 1, window_size))

    for num_oper in range(0, 85 - window_size, stride):
        temp_fr_vector = [fr_vector[num_oper + i] for i in range(window_size)]
        q_results = circuit(
            temp_fr_vector
        )

        for j in range(window_size):
            out[num_oper // stride, j] = q_results[j]

    return out

def get_and_save_embeddings():
    q_train_images = []
    print("Quantum pre-processing of train images:")
    # for idx, img in enumerate(train_images):
    for idx, img in X.iterrows():
        print("{}/{}        ".format(idx + 1, n_train), end="\r")
        q_train_images.append(quanv_fixed(img))
    q_train_images = np.asarray(q_train_images)

    q_test_images = []
    print("\nQuantum pre-processing of test images:")
    # for idx, img in enumerate(test_images):
    for idx, img in X_test.iterrows():
        print("{}/{}        ".format(idx + 1, n_test), end="\r")
        q_test_images.append(quanv_fixed(img))
    q_test_images = np.asarray(q_test_images)

    # Save pre-processed images
    np.save(SAVE_PATH + "q_train_images__6_3__v2.npy", q_train_images)
    np.save(SAVE_PATH + "q_test_images__6_3__v2.npy", q_test_images)

    return q_train_images, q_test_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else device

device = torch.device(device)
print(device)

class Classical_QEmbeds(nn.Module):
    def __init__(self, output_dim):
        super(Classical_QEmbeds, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=window_size, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(32, 64)
        # self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, output_dim)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)

        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)

        return x

def train_model():
    model__Classical_QEmbeds = Classical_QEmbeds(2).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model__Classical_QEmbeds.parameters(), lr=0.001)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)

    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    num_epochs = 100

    for epoch in range(num_epochs):
        model__Classical_QEmbeds.train()
        running_loss = 0.0

        for batch in training_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model__Classical_QEmbeds(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(training_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    model__Classical_QEmbeds.eval()
    val_loss = 0.0
    correct = 0

    all_pred = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model__Classical_QEmbeds(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            all_pred.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = correct / len(val_dataset)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(all_labels, all_pred, digits=4))

    acc = accuracy_score(all_labels, all_pred)
    print(f"Accuracy: {acc:.4f}")

    return model__Classical_QEmbeds

model__Classical_QEmbeds = train_model()
# model__Classical_QEmbeds = torch.load("/Users/soardr/FLIQ/models/Quanv1D_ClassicalFFN_StronglyEntanglingLayers_kernel.pt", weights_only=False)
# model__Classical_QEmbeds.eval()

test_preds_model = np.argmax(model__Classical_QEmbeds(X_test_tensor.to(device)).cpu().detach().numpy(), axis=1)

print("Prediction Classification Report:")
print(classification_report(Y_test_tensor.numpy(), test_preds_model.numpy(), digits=4))

acc = accuracy_score(Y_test_tensor.numpy(), test_preds_model.numpy())
print(f"Prediction Accuracy: {acc:.4f}")
