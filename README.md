# Custom RNN Architectures Implementation

This repository contains PyTorch implementations of three recurrent neural network architectures built from scratch: vanilla RNN, GRU (Gated Recurrent Unit), and LSTM (Long Short-Term Memory).

## Vanilla RNN

### Architecture
The vanilla RNN is the simplest form of recurrent neural network, processing sequential data by maintaining a hidden state that is updated at each time step.

```python
class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size + hidden_size, output_size)
```

### Forward Pass Equations
At each time step t, the RNN performs the following computations:

1. **Combined Input:**
   ```
   combined_t = [x_t, h_{t-1}]
   ```
   where `[·,·]` denotes concatenation

2. **Hidden State Update:**
   ```
   h_t = σ(W_h · combined_t + b_h)
   ```
   where σ is the sigmoid activation function

3. **Output Computation:**
   ```
   y_t = W_o · combined_t + b_o
   ```

## Gated Recurrent Unit (GRU)

### Architecture
GRU introduces update and reset gates to better control information flow through the network.

```python
class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.in2update = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2reset = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2out = nn.Linear(input_size + hidden_size, hidden_size)
```

### Forward Pass Equations
The GRU computes the following at each time step t:

1. **Gate Computations:**
   ```
   combined_t = [x_t, h_{t-1}]
   r_t = σ(W_r · combined_t + b_r)    # reset gate
   z_t = σ(W_z · combined_t + b_z)    # update gate
   ```

2. **Candidate Hidden State:**
   ```
   reset_hidden = [x_t, r_t ⊙ h_{t-1}]
   h̃_t = tanh(W_h · reset_hidden + b_h)
   ```
   where ⊙ denotes element-wise multiplication

3. **Hidden State Update:**
   ```
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
   ```

4. **Output Computation:**
   ```
   combined_output = [x_t, h_t]
   y_t = W_o · combined_output + b_o
   ```

## Long Short-Term Memory (LSTM)

### Architecture
LSTM uses three gates and a memory cell to control information flow and maintain long-term dependencies.

```python
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.in2cell = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2in = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2frg = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2out = nn.Linear(input_size + hidden_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, output_size)
```

### Forward Pass Equations
The LSTM computes the following at each time step t:

1. **Combined Input:**
   ```
   combined_t = [x_t, h_{t-1}]
   ```

2. **Gate Computations:**
   ```
   f_t = σ(W_f · combined_t + b_f)    # forget gate
   i_t = σ(W_i · combined_t + b_i)    # input gate
   o_t = σ(W_o · combined_t + b_o)    # output gate
   ```

3. **Cell State Update:**
   ```
   c̃_t = tanh(W_c · combined_t + b_c)    # candidate cell state
   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t       # new cell state
   ```

4. **Hidden State and Output:**
   ```
   h_t = o_t ⊙ tanh(c_t)
   y_t = W_y · h_t + b_y
   ```

## Initialization
All models use Kaiming uniform initialization for their hidden states, which helps prevent vanishing/exploding gradients at the start of training.

## Usage
To use any of these models:

1. Initialize the model with appropriate dimensions:
```python
model = CustomRNN(input_size=64, hidden_size=128, output_size=10)
# or
model = CustomGRU(input_size=64, hidden_size=128, output_size=10)
# or
model = CustomLSTM(input_size=64, hidden_size=128, output_size=10)
```

2. Initialize the hidden state:
```python
hidden = model.init_hidden()
```

3. Forward pass:
```python
output, hidden = model(x, hidden)  # For RNN/GRU
# or
output, hidden, cell = model(x, hidden, cell)  # For LSTM
```