from functools import cmp_to_key

from utils_lstm import get_args, get_bb_types, get_device

import torch
import torch.nn as nn

# Basic settings
torch.manual_seed(470)
torch.cuda.manual_seed(470)

args = get_args()
BB_TYPES = get_bb_types()
device = get_device()

class SlideEncoder(nn.Module):
    def __init__(self):
        super(SlideEncoder, self).__init__()
        ninp = args.ninp
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        self.embed = nn.Embedding(len(BB_TYPES), args.embedding_size, args.padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(ninp, nhid, nlayers, bias=True).float()

    def forward(self, x, states, lengths=None):
        """
        Args:
            x: tensor(B, L, 5)
            states: List[Tuple(h_0, c_0), ..., Tuple(h_{B-1}, c_{B-1})]
            lengths: tensor(B)
        """
        idxs = [*range(0, len(lengths))]
        def by_lengths(p1, p2):
            return lengths[p2] - lengths[p1]

        idxs = sorted(idxs, key=cmp_to_key(by_lengths))

        x = x[:, idxs]
        lengths = lengths[idxs]
        
        input = x[:, :, :-1]
        types = x[:, :, -1:].long()
        types = torch.squeeze(self.embed(types))
        input = torch.cat((input, types), dim=-1)

        output = self.dropout(input)
        

        output = torch.nn.utils.rnn.pack_padded_sequence(output, lengths.cpu())

        h_0 = torch.stack([h for (h, _) in states], dim=0)
        c_0 = torch.stack([c for (_, c) in states], dim=0)

        (output, context_vector) = self.lstm(output.to(device), (h_0, c_0))
        output, lengths = torch.nn.utils.rnn.pad_packed_sequence(output, total_length = args.max_seq_length)
        return (output, context_vector)

class SlideDeckEncoder(nn.Module):
    def __init__(self):
        super(SlideDeckEncoder, self).__init__()
        self.slide_encoder = SlideEncoder()

        input_size = args.nhid * args.slide_deck_N
        output_size = args.slide_deck_embedding_size

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        return

    def _get_init_states(self, x):
        init_states = [
            (torch.zeros((x.size(1), args.nhid)).to(x.device),
            torch.zeros((x.size(1), args.nhid)).to(x.device))
            for _ in range(args.nlayers)
        ]
        return init_states
    
    def forward(self, xs, lengths):
        states = None
        embedding = []
        for i, x in enumerate(xs):
            if states is None:
                states = self._get_init_states(x)
            length = lengths[i]
            output, states = self.slide_encoder(x, states, length)
            output = output[length.long() - 1,:,:]
            idxs = torch.arange(args.batch_size)
            output = output[idxs, idxs, :]
            embedding.append(output.squeeze())
        
        output = torch.cat(embedding, dim=-1)
        output = self.relu(self.linear(output))
        return output

class Generator(nn.Module):
    def __init__(self, embed_weights=None, ganlike=True):
        super(Generator, self).__init__()
        self.ganlike = ganlike
        self.embed = nn.Embedding(len(BB_TYPES), args.embedding_size, padding_idx=0)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=args.latent_vector_dim + args.embedding_size, hidden_size=args.nhid, num_layers=2, 
            batch_first=True, dropout=args.dropout, bias=True)
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(args.slide_deck_embedding_size, args.nhid)
        self.linear2 = nn.Linear(args.nhid, 4)
        if embed_weights is not None:
            self.embed.weight.data = embed_weights
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            # if normalize:
            #     layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.gen_model = nn.Sequential(
            *block(args.nhid, 32, normalize=False),
            *block(32, 32),
            nn.Linear(32, 4),
        )

    def forward(self, x, z, slide_deck_embedding, lengths=None):
        """

        Args:
            x (tensor): bb labels, (Batch_size, Sequence_size)
            z (tensor): latent vector, (Batch_size, latent_vector_dim)
            slide_deck_embedding (tensor): slide_deck_embedding vector, (Batch_size, slide_deck_embedding_dim)
            lengths (tensor): (Batch_size,)

        Returns:
            bb sequence: (tensor), (Batch_size, Sequence_size, 5)
        """
        # print(x.shape, z.shape, slide_deck_embedding.shape, lengths)
        x = x.int()
        (Batch_size, Sequence_size) = x.shape
        temp_input_1 = self.dropout(self.embed(x))   # Batch_size, Sequence_size, input_size
        # print(temp_input_1)
        # print("1",temp_input_1.shape)
        temp_input_2 = z.unsqueeze(1).repeat((1, Sequence_size, 1))
        # print("2",temp_input_2.shape)
        input_1 = torch.cat((temp_input_2, temp_input_1), dim=-1)
        # print(input_1.shape)
        input_1 = torch.nn.utils.rnn.pack_padded_sequence(input_1, lengths.cpu(), batch_first=True)
        # print(input_1.data.shape)
        # print("3",input_1.shape)
        hidden_0 = self.dropout(self.linear1(slide_deck_embedding)).unsqueeze(0).repeat((2, 1, 1))
        # print("4",hidden_0.shape)
        c_0 = torch.zeros(size=(2, Batch_size, args.nhid)).to(device)
        # print("5",c_0.shape)
        output, (h_n, c_n) = self.lstm(input_1.to(device), (hidden_0, c_0))
        # print(output.data.shape)
        output, length = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=args.max_seq_length)

        # output = output.transpose(0, 1)
        if self.ganlike:
            # output = output.transpose(1, 2)
            # print(output.shape)
            output = self.gen_model(output)
        else:
            output = self.linear2(output)
        output = self.tanh(output)
        return output, (h_n, c_n)

class Discriminator(nn.Module):
    def __init__(self, embed_weights=None):
        super(Discriminator, self).__init__()

        self.embed = nn.Embedding(len(BB_TYPES), args.embedding_size, padding_idx=0)
        self.dropout = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(input_size = args.ninp, hidden_size=args.nhid, num_layers=args.nlayers, 
            batch_first=True, bias=True)
        self.linear1 = nn.Linear(args.slide_deck_embedding_size, args.nhid)
        self.d_model = nn.Sequential(
            nn.Linear(args.nhid, args.nhid//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.nhid//2, args.nhid//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.nhid//2, 1)
        )
        if embed_weights is not None:
            self.embed.weight.data = embed_weights


    def forward(self, x, bb, slide_deck_embedding, lengths=None):
        """

        Args:
            x (tensor): type labels, (Batch_size, Sequence_size)
            bb (tensor): (Batch_size, Sequence_size, 4)
            slide_deck_embedding (tensor): slide_deck_embedding vector, (Batch_size, slide_deck_embedding_dim)
            length (tensor): (Batch_size,)

        Returns:
            
        """
        x = x.int()
        
        (Batch_size, _) = x.shape
        temp_input_1 = self.dropout(self.embed(x))   # Batch_size, Sequence_size, input_size
        input_1 = torch.cat((bb, temp_input_1), dim=-1)
        input_1 = torch.nn.utils.rnn.pack_padded_sequence(input_1, lengths.cpu(), batch_first=True)

        h_0 = self.dropout(self.linear1(slide_deck_embedding)).unsqueeze(0).repeat((2, 1, 1))
        c_0 = torch.zeros(size=(2,Batch_size, args.nhid)).to(device)
        output, (h_n, c_n) = self.lstm(input_1.to(device), (h_0, c_0))
        output, length = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=args.max_seq_length)
        
        output = output[:,length-1,:].squeeze()
        idxs = torch.arange(args.batch_size)
        output = output[idxs, idxs, :]
        #print(output[:, :10])
        output = self.d_model(output.squeeze())
        #print(output[:, :10])
        return output