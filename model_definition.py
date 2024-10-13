class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size,time_window):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.time_window = time_window
    def forward(self, x):
        h = torch.zeros(x.size(0), self.lstm.hidden_size).to(device)
        c = torch.zeros(x.size(0), self.lstm.hidden_size).to(device)

        enc_outputs = []

        for t in range(x.size(1)):
            # print(self.time_window)
            h, c = self.lstm(x[:, -x.size(1)+t, :], (h, c))
            enc_outputs.append(h.unsqueeze(1))

        enc_outputs = torch.cat(enc_outputs, dim=1)
        return enc_outputs[:, -32:, :]

class TemporalAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TemporalAttention, self).__init__()
        self.U = nn.Linear(input_size, hidden_size)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        h = h.repeat(x.size(1), 1, 1).transpose(0, 1)
        x = self.U(x)
        h = self.W(h)
        e = torch.tanh(x + h)
        attention_weights = torch.softmax(self.v(e), dim=1)
        return attention_weights

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.temporal_attention = TemporalAttention(hidden_size, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, enc_outputs):
        h = torch.zeros(enc_outputs.size(0), self.lstm.hidden_size).to(enc_outputs.device)
        c = torch.zeros(enc_outputs.size(0), self.lstm.hidden_size).to(enc_outputs.device)

        dec_outputs = []
        for t in range(enc_outputs.size(1)):
            temporal_weights = self.temporal_attention(enc_outputs, h)
            context_vector = torch.sum(temporal_weights * enc_outputs, dim=1)
            h, c = self.lstm(context_vector, (h, c))
            out = self.fc(h)
            dec_outputs.append(out.unsqueeze(1))

        dec_outputs = torch.cat(dec_outputs, dim=1)
        return dec_outputs

class SETANet_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,time_window):
        super(SETANet_Model, self).__init__()
        self.encoder = Encoder(input_size,hidden_size,time_window)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, x):
       # print(x.shape)
        enc_outputs = self.encoder(x)
        #print(enc_outputs.shape)   
        dec_output = self.decoder(enc_outputs)
        return dec_output

class SETANet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_models=1):
        super(SETANet, self).__init__()
        self.models = nn.ModuleList([SETANet_Model(input_size, hidden_size, output_size,time_window=192),SETANet_Model(input_size, hidden_size, output_size,time_window=48),SETANet_Model(input_size, hidden_size, output_size,time_window=48)])
        self.meta_learner = nn.Linear(num_models * output_size, output_size)
    def forward(self, x):
        # 对集成中的每个模型进行前向传播
        dec_outputs = [model(x) for model in self.models]

        stacked_outputs = torch.cat(dec_outputs, dim=2)

        final_output = self.meta_learner(stacked_outputs)
        
        return final_output