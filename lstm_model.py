import torch
import torch.nn as nn

#LSTM network
class Network(nn.Module):
    
    def __init__(self, input_size, hidden_units, layers_num, dropout_prob=0):
        super().__init__()
        self.hidden_units = hidden_units
        self.input_size = input_size
        self.rnn = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True,
                           bidirectional=False
                           )
        self.out = nn.Linear(hidden_units, int(input_size)) #add a linear layer to produce an output with the same numbe of units as the input (number of environmental states)
        
    def forward(self, x, state=None):
        x, rnn_state = self.rnn(x, state)
        outy = self.out(x)
        
        return outy, rnn_state