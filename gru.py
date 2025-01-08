import torch 
import torch.nn as nn

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.in2update = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2reset = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2out = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, hidden_state):
        # x (1, 59), hid (1, 256), comb (1, 315)
        combined = torch.cat((x, hidden_state), 1)
        reset = torch.sigmoid(self.in2reset(combined)) 
        update = torch.sigmoid(self.in2update(combined)) 

        reset_hid = torch.cat((x, reset * hidden_state), 1)
        h_tilde = torch.tanh(self.in2hidden(reset_hid)) 
        
        hidden = (1-update) * hidden_state + update * h_tilde
        combined_output = torch.cat((x, hidden), dim=1)
        output = self.in2out(combined_output)

        return output, hidden # output, hidden
            
    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))