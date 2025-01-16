import torch
import gpytorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURES = ['volt1', 'volt2', 'amp1', 'amp2', 'FQtyL', 'FQtyR', 'E1 FFlow',
            'E1 OilT', 'E1 OilP', 'E1 RPM', 'E1 CHT1', 'E1 CHT2', 'E1 CHT3',
            'E1 CHT4', 'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4', 'OAT', 'IAS',
            'VSpd', 'NormAc', 'AltMSL']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPCNNLSTM(nn.Module):
    def __init__(self, input_dim, lstm_hidden=64):
        super(GPCNNLSTM, self).__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.predictor_model = CRNN(7, lstm_hidden, n_class=2, leakyRelu=True).to(device)
        self.interpolation_model = None
        self.params = [{'params': self.predictor_model.parameters()}]
        self.optimizer = optim.AdamW(self.params, lr=0.0001)
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.35, 0.65]).to(device), reduction='mean')
        self.output_transform = nn.Softmax(dim=1)

    def init_interpolation_model(self, input_data):
        x, ind, y = input_data
        self.interpolation_model = MTGPInterpolationModel((x, ind), y, self.likelihood).to(device)
        self.params.append({'params': self.interpolation_model.covar_module.parameters(), 'lr': 0.001})
        self.params.append({'params': self.interpolation_model.mean_module.parameters(), 'lr': 0.001})
        self.params.append({'params': self.interpolation_model.likelihood.parameters()})

    def update(self, input_data, labels):
        batch_size = len(input_data)
        losses = []
        loss = 0.0
        for j in range(batch_size):
            x, ind, y = input_data[j]

            self.predictor_model.train()
            self.optimizer.zero_grad()

            with gpytorch.settings.detach_test_caches(False), \
                 gpytorch.settings.use_toeplitz(True), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True), \
                 gpytorch.settings.max_root_decomposition_size(12):
                self.interpolation_model.set_train_data((x, ind), y, strict=False)
                self.interpolation_model.train()
                self.interpolation_model.eval()

                sample = []
                x_eval = torch.linspace(0, labels[j].shape[1] - 1, labels[j].shape[1]).type(torch.FloatTensor).to(device)
                for ii in range(7):
                    task_idx = torch.full_like(x_eval, dtype=torch.long, fill_value=ii).to(device)
                    gp_output = self.interpolation_model(x_eval, task_idx)
                    f_samples = gp_output.rsample(torch.Size([10]))
                    sample_mean = f_samples.mean(0).squeeze(-1)
                    sample.append(sample_mean)

                    del task_idx

            vital_features = torch.stack(sample).unsqueeze(0).to(device)
            output = self.predictor_model(vital_features).squeeze(0)
            output = self.output_transform(output)
            loss += self.loss(output, labels[j].squeeze())

            del vital_features
            del x_eval

        loss = loss / batch_size
        losses.append(loss.cpu().item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor_model.parameters(), 0.25)

        self.optimizer.step()
        del output
        torch.cuda.empty_cache()
        del loss

        return np.mean(losses)

    def predict(self, input_data, seq_length):
        self.interpolation_model.eval()
        self.predictor_model.eval()

        vital_features, lab_features, baseline_features = input_data
        with gpytorch.settings.detach_test_caches(state=True), \
             gpytorch.settings.use_toeplitz(False), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True), \
             gpytorch.settings.max_root_decomposition_size(10):
            sample = []
            x_eval = torch.linspace(0, seq_length, seq_length).type(torch.FloatTensor).to(device)
            for ii in range(7):
                task_idx = torch.full_like(x_eval, dtype=torch.long, fill_value=ii).to(device)
                with torch.no_grad():
                    gp_output = self.interpolation_model(x_eval, task_idx)
                    f_samples = gp_output.rsample(torch.Size([10]))
                sample_mean = f_samples.mean(0).squeeze(-1)
                sample.append(sample_mean)

                del task_idx

        vital_features = torch.stack(sample).unsqueeze(0).to(device)
        output = self.predictor_model(vital_features).squeeze(0)
        output = self.output_transform(output)
        del x_eval
        del vital_features
        return output

class MTGPInterpolationModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MTGPInterpolationModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=7, rank=1)

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class LSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(LSTM, self).__init__()
        self.rnn = nn.GRU(nIn, nHidden, bidirectional=False, batch_first=True)
        self.embedding = nn.Linear(nHidden, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        output = self.embedding(recurrent[:, -1, :])  # Apply the embedding layer to the last time step
        return output


class CRNN(nn.Module):
    def __init__(self, input_dim, lstm_hidden, n_class, leakyRelu=True):
        super(CRNN, self).__init__()

        self.fc_att = nn.Linear(lstm_hidden, 1)
        self.att_softmax = nn.Softmax(dim=1)
        self.embedding = nn.Linear(lstm_hidden, n_class)

        kernel_size = [3, 3, 3, 3]
        pad_size = [1, 1, 1, 1]
        shift_size = [1, 1, 1, 1]
        in_size = [16, 32, 32, 64]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False, dropout=False):
            nIn = input_dim if i == 0 else in_size[i - 1]
            nOut = in_size[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv1d(nIn, nOut, kernel_size[i], shift_size[i], pad_size[i]))
            if dropout:
                cnn.add_module('dropout{0}'.format(i), nn.Dropout(0.1))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm1d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        convRelu(1)
        convRelu(2)
        convRelu(3, True)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            LSTM(7, lstm_hidden, lstm_hidden), LSTM(lstm_hidden, lstm_hidden, lstm_hidden))

    def forward(self, input):
        conv = input.permute(2, 0, 1)
        output = self.rnn(conv)
        output = self.embedding(output).squeeze(1)
        return output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Function to extend sequence
def extend_sequence(model, input_seq, num_steps_to_extend):
    model.eval()
    extended_sequence = input_seq.detach().cpu().numpy()
    # print("extended sequnce shape : ",extended_sequence.shape)
    for _ in range(num_steps_to_extend):
        # print("-----")
        next_value = model(input_seq).detach().cpu().numpy()

        next_value = next_value.reshape(1, 1, -1)  # Ensure next_value has the same number of dimensions
        # print("next value shape : ",next_value.shape)
        extended_sequence = np.concatenate((extended_sequence, next_value),axis=1)
        print("extended sequnce shape : ",extended_sequence.shape(1))
        # print("total sequence shape : ",extended_sequence.shape+ input_seq.shape)
        input_seq = torch.tensor(extended_sequence[:, -input_seq.size(1):, :], dtype=torch.float32)  # Ensure input_seq has the correct dimensions
        # print("input seq shape : ",input_seq.shape)
    return extended_sequence


def main():
    data = pd.read_csv('/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/planes/cleaned_flights/34_cleaned.csv')
    max_seq_len = 9930

 # Select the group with id 147 for demonstration
    group = data.groupby('id').get_group(147)
    group_values = group[['E1 CHT1']].values  # Use only 'E1 CHT1' feature for demonstration
    labels = group['before_after'].values

    #Print the original sequence shape
    print(group_values.shape)
    # Prepare the data for LSTM
    time_steps = 10
    X, y = create_dataset(group_values, time_steps)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define the LSTM model
    input_size = X_train.shape[2]
    hidden_size = 50
    output_size = input_size

    model = LSTM(input_size, hidden_size, output_size).to(device)

    # Train the model (example training loop)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    # Plot training and validation loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

    # Extend the sequence to max_seq_len
    num_steps_to_extend = max_seq_len - len(group_values)
    input_seq = torch.tensor(group_values[-time_steps:], dtype=torch.float32).unsqueeze(0).to(device)
    extended_sequence = extend_sequence(model, input_seq, num_steps_to_extend)

    # Squeeze the extra dimension from extended_sequence
    extended_sequence = extended_sequence.squeeze(0)

    # Combine original and extended sequences
    full_sequence = np.concatenate((group_values, extended_sequence), axis=0)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(group_values)), group_values[:, 0], label='Original Sequence')
    ax.plot(np.arange(len(group_values), len(full_sequence)), full_sequence[len(group_values):, 0], label='Extended Sequence')
    ax.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('E1 CHT1')
    plt.title('Original and Extended Sequences')
    plt.show()

    # Plot the entire predicted sequence overlay
    predicted_sequence = model(torch.tensor(full_sequence[:time_steps].reshape(1, time_steps, -1), dtype=torch.float32).to(device)).detach().cpu().numpy()
    for i in range(1, len(full_sequence) - time_steps):
        next_pred = model(torch.tensor(full_sequence[i:i+time_steps].reshape(1, time_steps, -1), dtype=torch.float32).to(device)).detach().cpu().numpy()
        predicted_sequence = np.concatenate((predicted_sequence, next_pred), axis=0)

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(full_sequence)), full_sequence[:, 0], label='Real Sequence')
    ax.plot(np.arange(len(predicted_sequence)), predicted_sequence[:, 0], label='Predicted Sequence', linestyle='dashed')
    ax.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('E1 CHT1')
    plt.title('Real vs Predicted Sequence')
    plt.show()



if __name__ == "__main__":
    main()
