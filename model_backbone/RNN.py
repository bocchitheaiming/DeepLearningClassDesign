# Implemention of RNN model for text classification
import torch
import torch.nn as nn

class myRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=128, output_size=2, num_layers=1, dropout=0.3):
        super(myRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) 

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.rnn_layers = nn.ModuleList()
        
        for layer in range(num_layers):
            if layer == 0:
                input_to_hidden = nn.Linear(embedding_dim, hidden_size, bias=False)
            else:
                input_to_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
            
            hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=True)
            
            self.rnn_layers.append(nn.ModuleDict({
                'input_to_hidden': input_to_hidden,
                'hidden_to_hidden': hidden_to_hidden
            }))
        
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # batch_size, seq_len, _ = x.size()
        batch_size, seq_len = x.size()
        x_emb = self.embedding(x)
        
        hidden_states = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x_emb.dtype)
            hidden_states.append(h)
        
        for t in range(seq_len):
            if t < seq_len:
                current_input = x_emb[:, t, :]  # (batch_size, embedding_dim or hidden_size)
            
            for layer in range(self.num_layers):
                rnn_layer = self.rnn_layers[layer]

                if layer == 0:
                    layer_input = current_input
                else:
                    layer_input = hidden_states[layer - 1]
                
                new_hidden = self.activation(rnn_layer['input_to_hidden'](layer_input) + rnn_layer['hidden_to_hidden'](hidden_states[layer])) # h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
                
                if layer < self.num_layers - 1:
                    new_hidden = self.dropout(new_hidden)
                hidden_states[layer] = new_hidden
        
        final_hidden = hidden_states[-1]  # (batch_size, hidden_size)
        output = self.output_layer(final_hidden)  # (batch_size, output_size)
        return output
    
    def init_weights(self):
        # initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

class myRNN_use_pytorch(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=128, output_size=2, num_layers=1, dropout=0.3):
        super(myRNN_use_pytorch, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.output_layer = nn.Linear(hidden_size*2, output_size)
        
    def forward(self, x):
        x_emb = self.embedding(x)
        output, _ = self.rnn(x_emb)
        output = torch.cat((output[:, -1, :], output[:, 0, :]), dim=1)
        output = self.output_layer(output)
        return output
    
    def init_weights(self):
        # initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

class myLSTM_use_pytorch(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=128, output_size=2, num_layers=1, dropout=0.3):
        super(myLSTM_use_pytorch, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.output_layer = nn.Linear(hidden_size*2, output_size)
        
    def forward(self, x):
        x_emb = self.embedding(x)
        output, _ = self.lstm(x_emb)  # output shape: (batch_size, seq_len, hidden_size)
        output = torch.cat((output[:, 0, :], output[:, 1, :]), dim=1)
        output = self.output_layer(output)
        return output
    
    def init_weights(self):
        # initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, output_size=2):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        hn = torch.cat((hn[0], hn[1]), dim=1)
        out = self.dropout(hn)
        return self.fc(out)
    
    def init_weights(self):
        # initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)