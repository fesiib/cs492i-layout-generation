from utils import SortByRefSlide


def test():
    for model in models:
        models[model].eval()

    batch = list(test_loader)[0]
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

    batch_size, _ = ref_types.shape

    # Sample noise as generator input
    z = torch.autograd.Variable(Tensor(np.random.normal(0, 1, (batch_size, args.latent_vector_dim))))
    fake_layouts_bbs = models['generator'](ref_types, z, slide_deck_embedding, length_ref)[0].detach()
    real_layouts_bbs = ref_slide[:,:,:-1]