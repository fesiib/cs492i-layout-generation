import torch
import torch.nn as nn

from model_layoutGAN import SlideEncoder, SlideDecoder, TransformerWithToken, Generator, Discriminator

class SlideDeckEncoder(nn.Module):
    def __init__(self, num_label, dim_slide, small_dim_slide, num_slide, padding_idx,
        d_model=512, nhead=8, num_layers=4):
        super().__init__()

        self.encoder = SlideEncoder(
            num_label, dim_slide, padding_idx,
            d_model, nhead, num_layers
        )
        
        self.fc_out = nn.Linear(dim_slide, small_dim_slide)
        
        self.relu = nn.LeakyReLU()

    def forward(self, bboxes, labels, padding_masks):
        # M x B x N x E
        slides = []
        for (i, label) in enumerate(labels):
            padding_mask = padding_masks[i]
            bbox = bboxes[i]
            output = self.encoder(bbox, label, padding_mask)
            slides.append(output)
        output = torch.stack(slides, dim=0)
        output = torch.max(output, dim=0).values
        deck_enc = self.fc_out(output)
        return deck_enc

class CombinedGenerator(nn.Module):
    def __init__(self, dim_latent, num_label, small_dim_slide, padding_idx,
        d_model=512, nhead=8, num_layers=4
    ):
        super().__init__()

        self.generator = Generator(dim_latent, num_label, small_dim_slide, padding_idx, d_model, nhead, num_layers)

        self.fc_deck = nn.Linear(small_dim_slide, d_model)
        self.fc_in = nn.Linear(d_model*2, d_model)

    def forward(self, z, label, deck_enc, padding_mask):
        # B x N x E
        z = self.generator.fc_z(z)
        l = self.generator.emb_label(label)
        d = self.fc_deck(deck_enc)
        d = d.unsqueeze(1).expand(-1, l.size(1), -1)
        x = torch.cat([z, l, d], dim=-1)

        x = self.generator.relu(self.fc_in(x)).permute(1, 0, 2)

        x = self.generator.transformer(x, src_key_padding_mask=padding_mask)

        x = self.generator.fc_out(x.permute(1, 0, 2))
        x = torch.sigmoid(x)
        return x

class CombinedDiscriminator(nn.Module):
    def __init__(self, num_label, dim_slide, small_dim_slide, max_bbox, padding_idx,
        d_model=512, nhead=8, num_layers=4,
        #encoder_state_dict=None, decoder_state_dict=None
    ):
        super().__init__()

        self.discriminator = Discriminator(num_label, dim_slide, max_bbox, padding_idx, d_model, nhead, num_layers)

        # if encoder_state_dict is not None:
        #     self.encoder.load_state_dict(encoder_state_dict)
        self.fc_out_deck = nn.Linear(dim_slide + small_dim_slide, dim_slide)

        # decoder
        # if decoder_state_dict is not None:
        #     self.decoder.load_state_dict(decoder_state_dict)

    def forward(self, bbox, label, deck_enc, padding_mask, reconst=False):
        slide_enc = self.discriminator.encoder(bbox, label, padding_mask)
        # logit_disc: [B,]

        x = torch.cat([deck_enc, slide_enc], dim=-1)
        x = self.fc_out_deck(x)
        x = self.discriminator.relu(x)
        logit_disc = self.discriminator.fc_out_disc(x).squeeze(-1)

        if not reconst:
            return logit_disc
        else:
            (logit_cls, bbox_pred) = self.discriminator.decoder(slide_enc, padding_mask)
            return logit_disc, logit_cls, bbox_pred