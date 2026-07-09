if __name__ == "__main__":
    import torch, os
    from torch.utils.data import DataLoader
    from torch import nn, optim
    from tqdm import tqdm
    from dataset import PairedDataset, default_transforms
    from model import UNetGenerator, PatchDiscriminator
    from torchvision.utils import save_image

    # --------------------
    # Hyperparameters
    # --------------------
    BATCH_SIZE = 8           # increase if GPU has memory, decrease if OOM
    IMG_SIZE = 128           # <-- changed to 128
    LR = 2e-4
    EPOCHS = 10
    LAMBDA_L1 = 100
    DATA_ROOT = "edges2shoes/train"
    CHECKPOINT_DIR = "checkpoints"
    OUTPUT_DIR = "outputs"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_PIN_MEMORY = torch.cuda.is_available()

    # Dataset and loader
    dataset = PairedDataset(DATA_ROOT, transforms=default_transforms(IMG_SIZE))
    num_workers = 4 if os.name != 'nt' else 0
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=num_workers, pin_memory=USE_PIN_MEMORY)

    # Models
    G = UNetGenerator(in_channels=3, out_channels=3).to(DEVICE)
    D = PatchDiscriminator(in_channels=6).to(DEVICE)

    # Losses and optimizers
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5,0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5,0.999))

    # Prepare fixed batch for sample images
    fixed_batch = next(iter(loader))
    fixed_A = fixed_batch['A'].to(DEVICE)
    fixed_B = fixed_batch['B'].to(DEVICE)

    global_step = 0
    for epoch in range(1, EPOCHS+1):
        loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in loop:
            real_A = batch['A'].to(DEVICE)
            real_B = batch['B'].to(DEVICE)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            D.zero_grad()
            pred_real = D(real_A, real_B)
            valid = torch.ones_like(pred_real, device=DEVICE)
            loss_D_real = criterion_GAN(pred_real, valid)

            fake_B = G(real_A)
            pred_fake = D(real_A, fake_B.detach())
            fake = torch.zeros_like(pred_fake, device=DEVICE)
            loss_D_fake = criterion_GAN(pred_fake, fake)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            G.zero_grad()
            pred_fake_for_G = D(real_A, fake_B)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, valid)
            loss_G_L1 = criterion_L1(fake_B, real_B) * LAMBDA_L1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            opt_G.step()

            global_step += 1

            loop.set_postfix({
                "loss_D": loss_D.item(),
                "loss_G": loss_G.item(),
                "loss_G_GAN": loss_G_GAN.item(),
                "loss_G_L1": loss_G_L1.item()
            })

            # Save sample images
            if global_step % 500 == 0:
                G.eval()
                with torch.no_grad():
                    fake = G(fixed_A)
                    n = min(4, fixed_A.size(0))
                    out_list = []
                    for i in range(n):
                        inp = (fixed_A[i] * 0.5 + 0.5)
                        gen = (fake[i] * 0.5 + 0.5)
                        tgt = (fixed_B[i] * 0.5 + 0.5)
                        row = torch.cat([inp, gen, tgt], dim=2)
                        out_list.append(row)
                    grid = torch.stack(out_list)
                    save_image(grid, os.path.join(OUTPUT_DIR, f"step_{global_step}.png"))
                G.train()

        # Save checkpoints each epoch
        torch.save({
            'G': G.state_dict(), 'D': D.state_dict(),
            'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict(), 'epoch': epoch
        }, os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch}.pth"))
