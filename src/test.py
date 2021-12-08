import torch

from utils import SortByRefSlide, draw_bbs, get_device, get_z

device = get_device()

def test(models, optimizers, loader):
    for model in models:
        models[model].eval()
    print(list(loader))
    batch = list(loader)[0]
    batch = SortByRefSlide(batch)
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
    draw_bbs((1,1),real_layouts_bbs[4])
    draw_bbs((1,1), fake_layouts_bbs[4])