import torch
import torch.nn as nn

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.in2cell = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2in = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2frg = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2out = nn.Linear(input_size + hidden_size, hidden_size)

        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state=None, cell_state=None):
        batch_size, _ = x.size()

        if not hidden_state or not cell_state:
            hidden_state, cell_state = self.init_hidden(batch_size)

        cmb = torch.cat((x, hidden_state), dim=1)

        forget = torch.sigmoid(self.in2frg(cmb))
        inp = torch.sigmoid(self.in2in(cmb))
        cell_cand = torch.tanh(self.in2cell(cmb))

        cell_state = forget * cell_state + inp * cell_cand

        out_gate = torch.sigmoid(self.in2out(cmb))
        hidden_state = out_gate * torch.tanh(cell_state)

        output = self.out_layer(hidden_state)
        return output, hidden_state, cell_state

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(batch_size, self.hidden_size)
        cell_state = torch.zeros(batch_size, self.hidden_size)
        return hidden_state, cell_state
