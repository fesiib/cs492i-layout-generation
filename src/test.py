import torch

from utils import SortByRefSlide, draw_bbs, get_device, get_z

device = get_device()

def test(models, optimizers, loader):
    for model in models:
        models[model].eval()

    batch = list(loader)[0]
    batch = SortByRefSlide(batch)
    shape = batch['shape'].to(device)
    x_slide_deck = batch["slide_deck"].to(device)
    length_ref = batch["length_ref_types"].to(device)
    ref_types = batch["ref_types"].to(device)
    ref_slide = batch["ref_slide"].to(device)

    x_slide_deck = torch.transpose(x_slide_deck, 0, 1)
    x_slide_deck = torch.transpose(x_slide_deck, 1, 2)
    lengths_slide_deck = torch.transpose(batch["lengths_slide_deck"], 0, 1).to(device)

    optimizers["encoder"].zero_grad()
    optimizers["discriminator"].zero_grad()

    slide_deck_embedding = models['encoder'](x_slide_deck, lengths_slide_deck)

    # Sample noise as generator input
    z = get_z()

    fake_layouts_bbs = models['generator'](ref_types, z, slide_deck_embedding, length_ref)[0].detach()
    real_layouts_bbs = ref_slide[:,:,:-1]
    draw_bbs(shape,real_layouts_bbs[4], ref_types[4])
    draw_bbs(shape, fake_layouts_bbs[4], ref_types[4])