import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from random import random
import matplotlib.pyplot as plt

# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        
    def forward(self, encoder_inputs):
        outputs, hidden = self.rnn(encoder_inputs)
        return outputs, hidden

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, initial_input, encoder_outputs, hidden, targets, teacher_force_probability):
        decoder_sequence_length = len(targets)
        outputs = torch.zeros(decoder_sequence_length, 1, 1)
        input_at_t = initial_input
        
        for t in range(decoder_sequence_length):            
            output, hidden = self.rnn(input_at_t, hidden)
            output = self.out(output)
            outputs[t] = output
            
            teacher_force = random() < teacher_force_probability
            input_at_t = targets[t].unsqueeze(0) if teacher_force else output

        return outputs

# Define the Seq2Seq class
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, lr):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.loss_function = nn.L1Loss()
    
    def forward(self, encoder_inputs, targets, teacher_force_probability):
        encoder_outputs, hidden = self.encoder(encoder_inputs)
        outputs = self.decoder(encoder_inputs[:, -1, :].unsqueeze(1), encoder_outputs, hidden, targets, teacher_force_probability)
        return outputs

    def compute_loss(self, outputs, targets):
        loss = self.loss_function(outputs, targets)
        return loss
    
    def optimize(self, outputs, targets):
        self.optimizer.zero_grad()
        loss = self.compute_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

# Function to create dataset
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Function to extend sequence
def extend_sequence(model, input_seq, num_steps_to_extend, teacher_force_probability):
    model.eval()
    extended_sequence = input_seq.detach().cpu().numpy()
    for _ in range(num_steps_to_extend):
        targets = torch.zeros(1, 1, 1)  # Dummy targets for the decoder
        next_value = model(input_seq, targets, teacher_force_probability).detach().cpu().numpy()
        next_value = next_value.reshape(1, 1, -1)  # Ensure next_value has the same number of dimensions
        extended_sequence = np.concatenate((extended_sequence, next_value), axis=1)  # Concatenate along the time steps axis
        input_seq = torch.tensor(extended_sequence[:, -input_seq.size(1):, :], dtype=torch.float32).unsqueeze(0)  # Ensure input_seq has the correct dimensions
    return extended_sequence

# Main function
if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('/Users/manasdubey2022/Desktop/NGAFID/Codebase/data/planes/cleaned_flights/34_cleaned.csv')
    max_seq_len = 9930

    # Select the group with id 147 for demonstration
    group = data.groupby('id').get_group(147)
    group_values = group[['E1 CHT1']].values  # Use only 'E1 CHT1' feature for demonstration

    # Prepare the data for Seq2Seq
    time_steps = 10
    X, y = create_dataset(group_values, time_steps)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = y.reshape(y.shape[0], 1, y.shape[1])  # Reshape y to match the decoder's expected input shape
    print(X.shape, y.shape)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define the Seq2Seq model
    hidden_size = 50
    lr = 0.001
    seq2seq = Seq2Seq(Encoder(hidden_size), Decoder(hidden_size), lr)

    # Training loop
    num_epochs = 20
    teacher_force_probability = 0.5
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            outputs = seq2seq(batch_X, batch_y, teacher_force_probability)
            seq2seq.optimize(outputs, batch_y)
        loss = seq2seq.compute_loss(outputs, batch_y)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Extend the sequence
    num_steps_to_extend = 100
    input_seq = torch.tensor(group_values[-time_steps:], dtype=torch.float32).unsqueeze(0)
    extended_sequence = extend_sequence(seq2seq, input_seq, num_steps_to_extend, teacher_force_probability)

    # Squeeze the extra dimension from extended_sequence
    extended_sequence = extended_sequence.squeeze(0)

    # Print shapes for debugging
    print("Final extended sequence shape:", extended_sequence.shape)
    print("Group values shape:", group_values.shape)

    # Combine original and extended sequences
    full_sequence = np.concatenate((group_values, extended_sequence), axis=0)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(group_values)), group_values[:, 0], label='Original Sequence')
    ax.plot(np.arange(len(group_values), len(full_sequence)), full_sequence[len(group_values):, 0], label='Extended Sequence')
    ax.legend()
    plt.show()