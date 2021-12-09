import torch
import torch.nn as nn

class TransformerWithToken(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        token_mask = torch.zeros(1, 1, dtype=torch.bool)
        self.register_buffer('token_mask', token_mask)

        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
            ), num_layers=num_layers)

    def forward(self, x, src_key_padding_mask):
        # x: [N, B, E]
        # padding_mask: [B, N]
        #   `False` for valid values
        #   `True` for padded values

        B = x.size(1)

        token = self.token.expand(-1, B, -1)
        x = torch.cat([token, x], dim=0)

        token_mask = self.token_mask.expand(B, -1)
        padding_mask = torch.cat([token_mask, src_key_padding_mask], dim=1)

        x = self.core(x, src_key_padding_mask=padding_mask)

        return x

class SlideDecoder(nn.Module):
    def __init__(
        self, num_label, dim_slide, max_bbox,
        d_model=512, nhead=8, num_layers=4
    ):
        super().__init__()

        self.max_bbox = max_bbox
        self.dec_fc_slide = nn.Linear(dim_slide, d_model)

        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=d_model // 2)
        self.dec_transformer = nn.TransformerEncoder(te,
                                                     num_layers=num_layers)

        self.fc_out_cls = nn.Linear(d_model, num_label)
        self.fc_out_bbox = nn.Linear(d_model, 4)
        self.relu = nn.LeakyReLU()
    
    def forward(self, slide_enc, padding_mask):
        # B x E
        B, _ = slide_enc.size()
        N = self.max_bbox

        x = self.dec_fc_slide(slide_enc)

        x = x.unsqueeze(0).expand(N, -1, -1)
        t = self.pos_token[:N].expand(-1, B, -1)
        x = torch.cat([x, t], dim=-1)
        x = self.relu(self.dec_fc_in(x))
        x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
        x = x.permute(1, 0, 2)[~padding_mask]
        # B, N, E
        # logit_cls: [M, L]    bbox_pred: [M, 4]
        logit_cls = self.fc_out_cls(x)
        bbox_pred = torch.tanh(self.fc_out_bbox(x))

        return logit_cls, bbox_pred

class SlideEncoder(nn.Module):
    def __init__(
        self, num_label, dim_slide, padding_idx,
        d_model=512, nhead=8, num_layers=4
    ):
        super().__init__()
        
        #encoder
        self.emb_label = nn.Embedding(num_label, d_model, padding_idx)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)

        self.enc_transformer = TransformerWithToken(d_model=d_model,
                                                    dim_feedforward=d_model // 2,
                                                    nhead=nhead, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, dim_slide)

        self.relu = nn.LeakyReLU()
    
    def forward(self, bbox, label, padding_mask):
        # B x N x E
        B, N, _ = bbox.size()
        b = self.fc_bbox(bbox)
        l = self.emb_label(label)
        x = self.enc_fc_in(torch.cat([b, l], dim=-1))
        x = self.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, src_key_padding_mask=padding_mask)
        x = x[0]

        slide_enc = self.fc_out(x)
        return slide_enc

class SlideDeckEncoder(nn.Module):
    def __init__(self, num_label, dim_slide, num_slide, padding_idx,
        d_model=512, nhead=8, num_layers=4):
        super().__init__()

        self.encoder = SlideEncoder(
            num_label, dim_slide, padding_idx,
            d_model, nhead, num_layers
        )
        
        self.fc_inner = nn.Linear(dim_slide * num_slide, dim_slide)
        self.fc_out = nn.Linear(dim_slide, dim_slide)
        
        self.relu = nn.LeakyReLU()

    def forward(self, bboxes, labels, padding_masks):
        # M x B x N x E
        slides = []
        for (i, label) in enumerate(labels):
            padding_mask = padding_masks[i]
            bbox = bboxes[i]
            output = self.encoder(bbox, label, padding_mask)
            slides.append(output)
        output = torch.cat(slides, dim=-1)
        output = self.relu(self.fc_inner(output))
        deck_enc = self.fc_out(output)
        return deck_enc

class Generator(nn.Module):
    def __init__(self, dim_latent, num_label, dim_slide, padding_idx,
        d_model=512, nhead=8, num_layers=4
    ):
        super().__init__()

        self.fc_z = nn.Linear(dim_latent, d_model // 2)
        
        self.emb_label = nn.Embedding(num_label, d_model // 2, padding_idx)


        self.fc_deck = nn.Linear(dim_slide, d_model)

        self.fc_in = nn.Linear(d_model * 2, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=d_model // 2)
        self.transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, 4)

        self.relu = nn.LeakyReLU()

    def forward(self, z, label, deck_enc, padding_mask):
        # B x N x E
        z = self.fc_z(z)
        l = self.emb_label(label)
        d = self.fc_deck(deck_enc)
        d = d.unsqueeze(1).expand(-1, l.size(1), -1)
        x = torch.cat([z, l, d], dim=-1)

        x = self.relu(self.fc_in(x)).permute(1, 0, 2)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = self.fc_out(x.permute(1, 0, 2))
        x = torch.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_label, dim_slide, max_bbox, padding_idx,
        d_model=512, nhead=8, num_layers=4,
        #encoder_state_dict=None, decoder_state_dict=None
    ):
        super().__init__()
        self.encoder = SlideEncoder(
            num_label, dim_slide, padding_idx,
            d_model, nhead, num_layers
        )
        # if encoder_state_dict is not None:
        #     self.encoder.load_state_dict(encoder_state_dict)
        self.fc_out_deck = nn.Linear(dim_slide * 2, dim_slide)
        self.fc_out_disc = nn.Linear(dim_slide, 1)
        self.relu = nn.LeakyReLU()

        # decoder
        self.decoder = SlideDecoder(num_label, dim_slide, max_bbox, d_model, nhead, num_layers)
        # if decoder_state_dict is not None:
        #     self.decoder.load_state_dict(decoder_state_dict)

    def forward(self, bbox, label, deck_enc, padding_mask, reconst=False):
        slide_enc = self.encoder(bbox, label, padding_mask)
        # logit_disc: [B,]

        x = torch.cat([deck_enc, slide_enc], dim=-1)
        x = self.fc_out_deck(x)
        x = self.relu(x)
        logit_disc = self.fc_out_disc(x).squeeze(-1)

        if not reconst:
            return logit_disc
        else:
            (logit_cls, bbox_pred) = self.decoder(slide_enc, padding_mask)
            return logit_disc, logit_cls, bbox_pred