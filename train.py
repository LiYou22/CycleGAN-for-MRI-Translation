import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.cyclegan import CycleGAN
from dataset import MRIT1T2Dataset
from utils.seed_utils import set_seed, seed_worker
from utils.visualize import visualize_progress, plot_history

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class Config:
    # Dataset
    t1_dir = 'data/IXI-T1'
    t2_dir = 'data/IXI-T2'
    slice_mode = 'middle'
    paired = True
    transform = None
    cache_size = 2
    train_split_ratio = 0.8
    
    # Training
    batch_size = 1
    num_epochs = 30
    generator_type = 'resnet'
    resume_training = True
    seed = 42
    
    # Leanring rate
    lr_g = 2e-4
    lr_d = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    
    # others
    checkpoint_root_dir = 'checkpoints/cyclegan'
    vis_root_dir = 'visualizations/cyclegan'
    log_interval = 100

    @property
    def checkpoint_dir(self):
        return os.path.join(self.checkpoint_root_dir, self.generator_type)

    @property
    def vis_dir(self):
        return os.path.join(self.vis_root_dir, self.generator_type)


def train_epoch(model, dataloader, epoch, device, vis_dir, log_interval=100):
    model.train()
    start_time = time.time()
    epoch_G_losses = []
    epoch_D_X_losses = []
    epoch_D_Y_losses = []
    epoch_cycle_losses = []
    epoch_identity_losses = []

    for i, batch in enumerate(dataloader):
        real_X = batch['T1'].to(device)
        real_Y = batch['T2'].to(device)

        fake_Y, rec_X, fake_X, rec_Y = model.optimize(real_X, real_Y)

        epoch_G_losses.append(model.loss_G.item())
        epoch_D_X_losses.append(model.loss_D_X.item())
        epoch_D_Y_losses.append(model.loss_D_Y.item())
        epoch_cycle_losses.append((model.loss_cycle_X + model.loss_cycle_Y).item())
        epoch_identity_losses.append((model.loss_idt_X + model.loss_idt_Y).item())

        if i % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch} | Step {i}/{len(dataloader)} | Elapsed: {elapsed:.2f} sec")
            generated_images = {
                'fake_Y': fake_Y,
                'fake_X': fake_X,
                'rec_X': rec_X,
                'rec_Y': rec_Y
            }
            visualize_progress(epoch, i, batch, generated_images, vis_dir)
            start_time = time.time()
    
    avg_g_loss = np.mean(epoch_G_losses)
    avg_d_x_loss = np.mean(epoch_D_X_losses)
    avg_d_y_loss = np.mean(epoch_D_Y_losses)
    avg_cycle_loss = np.mean(epoch_cycle_losses)
    avg_identity_loss = np.mean(epoch_identity_losses)

    model.history['G_losses'].append(avg_g_loss)
    model.history['D_X_losses'].append(avg_d_x_loss)
    model.history['D_Y_losses'].append(avg_d_y_loss)
    model.history['cycle_losses'].append(avg_cycle_loss)
    model.history['identity_losses'].append(avg_identity_loss)

    print(f"Training Losses this epoch:")
    print(f"  G_loss: {avg_g_loss:.4f}, D_X_loss: {avg_d_x_loss:.4f}, D_Y_loss: {avg_d_y_loss:.4f}")
    print(f"  Cycle_loss: {avg_cycle_loss:.4f}, Identity_loss: {avg_identity_loss:.4f}")

    return {
        'G_loss': avg_g_loss,
        'D_X_loss': avg_d_x_loss,
        'D_Y_loss': avg_d_y_loss,
        'cycle_loss': avg_cycle_loss,
        'identity_loss': avg_identity_loss
    }



def evaluate_model(model, dataloader, device):
    model.eval()
    t1_to_t2_psnr = []
    t1_to_t2_ssim = []
    t2_to_t1_psnr = []
    t2_to_t1_ssim = []

    val_g_losses = []
    val_d_x_losses = []
    val_d_y_losses = []
    val_cycle_losses = []
    val_identity_losses = []

    with torch.no_grad():
        for batch in dataloader:
            real_t1 = batch['T1'].to(device)
            real_t2 = batch['T2'].to(device)
            
            # Forward pass
            fake_t2, rec_t1, fake_t1, rec_t2 = model(real_t1, real_t2)
            fake_t2 = fake_t2.to(device)
            real_t2 = real_t2.to(device)
            fake_t1 = fake_t1.to(device)
            real_t1 = real_t1.to(device)

            # Discrinator Loss
            pred_real_x = model.D_X(real_t1)
            loss_real_x = model.criterion_GAN(pred_real_x, torch.ones_like(pred_real_x))
            pred_fake_x = model.D_X(fake_t1)
            loss_fake_x = model.criterion_GAN(pred_fake_x, torch.zeros_like(pred_fake_x))
            d_x_loss = (loss_real_x + loss_fake_x) * 0.5
            
            pred_real_y = model.D_Y(real_t2)
            loss_real_y = model.criterion_GAN(pred_real_y, torch.ones_like(pred_real_y))
            pred_fake_y = model.D_Y(fake_t2)
            loss_fake_y = model.criterion_GAN(pred_fake_y, torch.zeros_like(pred_fake_y))
            d_y_loss = (loss_real_y + loss_fake_y) * 0.5

            # Generator Loss
            g_xtoy_loss = model.criterion_GAN(pred_fake_y, torch.ones_like(pred_fake_y))
            g_ytox_loss = model.criterion_GAN(pred_fake_x, torch.ones_like(pred_fake_x))

            # Cycle loss
            cycle_x_loss = model.criterion_cycle(rec_t1, real_t1) * model.lambda_cycle
            cycle_y_loss = model.criterion_cycle(rec_t2, real_t2) * model.lambda_cycle
            
            # Identity Loss
            idt_x = model.G_YtoX(real_t1)
            idt_y = model.G_XtoY(real_t2)
            idt_loss_x = model.criterion_identity(idt_x, real_t1) * model.lambda_id
            idt_loss_y = model.criterion_identity(idt_y, real_t2) * model.lambda_id

            # G loss            
            g_loss = (g_xtoy_loss + g_ytox_loss + cycle_x_loss + cycle_y_loss + idt_loss_x + idt_loss_y)

            val_g_losses.append(g_loss.item())
            val_d_x_losses.append(d_x_loss.item())
            val_d_y_losses.append(d_y_loss.item())
            val_cycle_losses.append((cycle_x_loss + cycle_y_loss).item())
            val_identity_losses.append((idt_loss_x + idt_loss_y).item())

            val_g_losses.append(g_loss.item())
            val_d_x_losses.append(d_x_loss.item())
            val_d_y_losses.append(d_y_loss.item())
            val_cycle_losses.append((cycle_x_loss + cycle_y_loss).item())
            val_identity_losses.append((idt_loss_x + idt_loss_y).item())

            # Compute PSNR and SSIM metrics
            psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

            t1_to_t2_psnr_val = psnr_metric(fake_t2, real_t2)
            t1_to_t2_ssim_val = ssim_metric(fake_t2, real_t2)
            t1_to_t2_psnr.append(t1_to_t2_psnr_val.cpu().item())
            t1_to_t2_ssim.append(t1_to_t2_ssim_val.cpu().item())

            t2_to_t1_psnr_val = psnr_metric(fake_t1, real_t1)
            t2_to_t1_ssim_val = ssim_metric(fake_t1, real_t1)
            t2_to_t1_psnr.append(t2_to_t1_psnr_val.cpu().item())
            t2_to_t1_ssim.append(t2_to_t1_ssim_val.cpu().item())

    # Compute the mean of losses and metrics
    avg_g_loss = np.mean(val_g_losses)
    avg_d_x_loss = np.mean(val_d_x_losses)
    avg_d_y_loss = np.mean(val_d_y_losses)
    avg_cycle_loss = np.mean(val_cycle_losses)
    avg_identity_loss = np.mean(val_identity_losses)

    avg_t1_to_t2_psnr = np.mean(t1_to_t2_psnr)
    avg_t1_to_t2_ssim = np.mean(t1_to_t2_ssim)
    avg_t2_to_t1_psnr = np.mean(t2_to_t1_psnr)
    avg_t2_to_t1_ssim = np.mean(t2_to_t1_ssim)

    avg_psnr = (avg_t1_to_t2_psnr + avg_t2_to_t1_psnr) / 2
    avg_ssim = (avg_t1_to_t2_ssim + avg_t2_to_t1_ssim) / 2

    model.history['val_psnr'].append(avg_psnr)
    model.history['val_ssim'].append(avg_ssim)
    model.history['val_t1_to_t2_psnr'] = model.history.get('val_t1_to_t2_psnr', []) + [avg_t1_to_t2_psnr]
    model.history['val_t1_to_t2_ssim'] = model.history.get('val_t1_to_t2_ssim', []) + [avg_t1_to_t2_ssim]
    model.history['val_t2_to_t1_psnr'] = model.history.get('val_t2_to_t1_psnr', []) + [avg_t2_to_t1_psnr]
    model.history['val_t2_to_t1_ssim'] = model.history.get('val_t2_to_t1_ssim', []) + [avg_t2_to_t1_ssim]

    model.history['val_G_losses'].append(avg_g_loss)
    model.history['val_D_X_losses'].append(avg_d_x_loss)
    model.history['val_D_Y_losses'].append(avg_d_y_loss)
    model.history['val_cycle_losses'].append(avg_cycle_loss)
    model.history['val_identity_losses'].append(avg_identity_loss)

    print(f"Evaluation Metrics: ")
    print(f"  T1->T2: PSNR: {avg_t1_to_t2_psnr:.2f}, SSIM: {avg_t1_to_t2_ssim:.2f}")
    print(f"  T2->T1: PSNR: {avg_t2_to_t1_psnr:.2f}, SSIM: {avg_t2_to_t1_ssim:.2f}")
    print(f"  Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.2f}")
    
    print(f"Validation Losses:")
    print(f"  G_loss: {avg_g_loss:.4f}, D_X_loss: {avg_d_x_loss:.4f}, D_Y_loss: {avg_d_y_loss:.4f}")
    print(f"  Cycle_loss: {avg_cycle_loss:.4f}, Identity_loss: {avg_identity_loss:.4f}")

    return {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        't1_to_t2_psnr': avg_t1_to_t2_psnr,
        't1_to_t2_ssim': avg_t1_to_t2_ssim,
        't2_to_t1_psnr': avg_t2_to_t1_psnr,
        't2_to_t1_ssim': avg_t2_to_t1_ssim,
        'g_loss': avg_g_loss,
        'd_x_loss': avg_d_x_loss,
        'd_y_loss': avg_d_y_loss,
        'cycle_loss': avg_cycle_loss,
        'identity_loss': avg_identity_loss
    }


def save_checkpoint(model, checkpoint_path, epoch):
    checkpoint = {
        'epoch': epoch,
        'G_XtoY_state': model.G_XtoY.state_dict(),
        'G_YtoX_state': model.G_YtoX.state_dict(),
        'D_X_state': model.D_X.state_dict(),
        'D_Y_state': model.D_Y.state_dict(),
        'opt_G_state': model.opt_G.state_dict(),
        'opt_D_X_state': model.opt_D_X.state_dict(),
        'opt_D_Y_state': model.opt_D_Y.state_dict(),
        'scheduler_G_state': model.scheduler_G.state_dict(),
        'scheduler_D_X_state': model.scheduler_D_X.state_dict(),
        'scheduler_D_Y_state': model.scheduler_D_Y.state_dict(),
        'history': model.history
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def train_and_validate(model, train_loader, val_loader, num_epochs, device, checkpoint_dir, vis_dir, resume_training=True):
    start_epoch = 0
    best_ssim = 0.0

    best_model_dir = os.path.join(checkpoint_dir, 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)

    # Resume training if checkpoints exist
    if resume_training:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and 'checkpoint_epoch_' in f]
        if checkpoints:
            epoch_nums = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
            latest_epoch_idx = epoch_nums.index(max(epoch_nums))
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[latest_epoch_idx])
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.G_XtoY.load_state_dict(checkpoint['G_XtoY_state'])
            model.G_YtoX.load_state_dict(checkpoint['G_YtoX_state'])
            model.D_X.load_state_dict(checkpoint['D_X_state'])
            model.D_Y.load_state_dict(checkpoint['D_Y_state'])
            model.opt_G.load_state_dict(checkpoint['opt_G_state'])
            model.opt_D_X.load_state_dict(checkpoint['opt_D_X_state'])
            model.opt_D_Y.load_state_dict(checkpoint['opt_D_Y_state'])
            model.scheduler_G.load_state_dict(checkpoint['scheduler_G_state'])
            model.scheduler_D_X.load_state_dict(checkpoint['scheduler_D_X_state'])
            model.scheduler_D_Y.load_state_dict(checkpoint['scheduler_D_Y_state'])
            model.history = checkpoint['history']
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n=== Starting Epoch {epoch}/{num_epochs-1} ===")
        avg_losses = train_epoch(model, train_loader, epoch, device, vis_dir)
        
        # Update learning rate schedulers
        model.scheduler_G.step()
        model.scheduler_D_X.step()
        model.scheduler_D_Y.step()

        # Run evaluation on the validation set
        print(f"Running evaluation for epoch {epoch}...")

        # Store the checkpoints with best SSIM
        metrics = evaluate_model(model, val_loader, device)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            print(f"Regular checkpoint saving (every 5 epochs)...")
            save_checkpoint(model, save_path, epoch)
        
        if metrics['avg_ssim'] > best_ssim:
            print(f"Average SSIM gets improved from {best_ssim:.4f} to {metrics['avg_ssim']:.4f}. Save best checkpoint.")
            best_ssim = metrics['avg_ssim']
        
            with open(os.path.join(best_model_dir, 'best_ssim.txt'), 'w') as f:
                f.write(str(best_ssim))
            
            best_model_path = os.path.join(best_model_dir, 'best_model.pth')
            save_checkpoint(model, best_model_path, epoch)

            print(f"Best model saved, SSIM={best_ssim:.4f}, epoch={epoch}")


def main():
    # Initialize hyperparameter configuration
    config = Config()

    # Set seed
    set_seed(config.seed)

    # Prepare dataset
    dataset = MRIT1T2Dataset(
        t1_dir=config.t1_dir,
        t2_dir=config.t2_dir,
        slice_mode=config.slice_mode,
        paired=config.paired,
        transform=config.transform,
        cache_size=config.cache_size
    )
    
    all_files = dataset.data_files
    random.shuffle(all_files)
    split_index = int(config.train_split_ratio * len(all_files))
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    train_dataset = MRIT1T2Dataset(
        t1_dir=config.t1_dir,
        t2_dir=config.t2_dir,
        slice_mode=config.slice_mode,
        paired=config.paired,
        transform=config.transform,
        cache_size=config.cache_size,
        file_list=train_files
    )
    val_dataset = MRIT1T2Dataset(
        t1_dir=config.t1_dir,
        t2_dir=config.t2_dir,
        slice_mode=config.slice_mode,
        paired=config.paired,
        transform=config.transform,
        cache_size=config.cache_size,
        file_list=val_files
    )

    g = torch.Generator()
    g.manual_seed(config.seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Initialize model
    cyclegan = CycleGAN(generator_type=config.generator_type, device=device)

    # prepare checkpoint and visualization directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.vis_dir, exist_ok=True)
    
    # Train
    train_and_validate(
        cyclegan,
        train_loader,
        val_loader,
        config.num_epochs,
        device,
        checkpoint_dir=config.checkpoint_dir,
        vis_dir=config.vis_dir,
        resume_training=config.resume_training
    )

    # Plot loss
    plot_history(cyclegan.history, out_dir=os.path.join(config.vis_dir, "plots"))


if __name__ == "__main__":
    main()
