# RNN-GRU-LSTM

This repository contains custom implementations of Recurrent Neural Network (RNN) and Gated Recurrent Unit (GRU) models using PyTorch. These implementations provide a clear understanding of the internal mechanisms of these architectures.

## Model Architectures

### 1. Custom RNN

The CustomRNN is a basic recurrent neural network implementation that processes sequential data. It maintains a hidden state that carries information across time steps.

#### Architecture Details:
```python
class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size + hidden_size, output_size)
```

#### Forward Pass Formulation:
The RNN processes inputs using the following equations:

1. Combined Input: `combined = [x_t; h_{t-1}]`
   - `x_t`: Current input (size: input_size)
   - `h_{t-1}`: Previous hidden state (size: hidden_size)
   
2. Hidden State Update: `h_t = σ(W_h · combined + b_h)`
   - `σ`: Sigmoid activation function
   - `W_h`: Hidden state weights
   - `b_h`: Hidden state bias

3. Output Generation: `y_t = W_o · combined + b_o`
   - `W_o`: Output weights
   - `b_o`: Output bias

### 2. Custom GRU

The CustomGRU implements a Gated Recurrent Unit, which is more sophisticated than standard RNN and includes update and reset gates to better control information flow.

#### Architecture Details:
```python
class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.in2update = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2reset = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2out = nn.Linear(input_size + hidden_size, hidden_size)
```

#### Forward Pass Formulation:
The GRU processes inputs using the following equations:

1. Combined Input: `combined = [x_t; h_{t-1}]`
   - `x_t`: Current input
   - `h_{t-1}`: Previous hidden state

2. Update Gate: `z_t = σ(W_z · combined + b_z)`
   - Controls how much of the new state should be used
   - `σ`: Sigmoid activation function

3. Reset Gate: `r_t = σ(W_r · combined + b_r)`
   - Controls how much of the previous state to forget

4. Candidate Hidden State: `h̃_t = tanh(W_h · [x_t; (r_t ⊙ h_{t-1})] + b_h)`
   - `⊙`: Element-wise multiplication
   - Creates a candidate state using reset gate

5. Final Hidden State: `h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t`
   - Updates the hidden state using the update gate

6. Output Generation: `y_t = W_o · [x_t; h_t] + b_o`
   - Produces the final output using the new hidden state

## Initialization

Both models use Kaiming uniform initialization for their hidden states:
```python
def init_hidden(self):
    return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))
```

## Key Differences

1. **Gating Mechanism**: 
   - RNN uses a simple update rule with a single transformation
   - GRU uses update and reset gates to control information flow

2. **Memory Retention**:
   - RNN may struggle with long-term dependencies
   - GRU's gating mechanism helps better preserve relevant information

3. **Parameter Count**:
   - RNN has fewer parameters and is simpler to train
   - GRU has more parameters but offers better control over information flow

## Usage Example

```python
# Initialize models
rnn = CustomRNN(input_size=59, hidden_size=256, output_size=256)
gru = CustomGRU(input_size=59, hidden_size=256, output_size=256)

# Initialize hidden states
hidden_rnn = rnn.init_hidden()
hidden_gru = gru.init_hidden()

# Forward pass
output_rnn, hidden_rnn = rnn(input_tensor, hidden_rnn)
output_gru, hidden_gru = gru(input_tensor, hidden_gru)
```