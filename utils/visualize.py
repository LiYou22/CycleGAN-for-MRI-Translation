import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def visualize_progress(epoch, step, batch, generated_images, vis_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # First row: T1 -> T2 -> Reconstructed T1
    axes[0, 0].imshow(batch['T1'][0, 0].cpu().numpy().T, cmap='gray', origin='lower')
    axes[0, 0].set_title('Original T1')
    axes[0, 1].imshow(generated_images['fake_Y'][0, 0].detach().cpu().numpy().T, cmap='gray', origin='lower')
    axes[0, 1].set_title('Generated T2')
    axes[0, 2].imshow(generated_images['rec_X'][0, 0].detach().cpu().numpy().T, cmap='gray', origin='lower')
    axes[0, 2].set_title('Reconstructed T1')
    
    # Second row: T2 -> T1 -> Reconstructed T2
    axes[1, 0].imshow(batch['T2'][0, 0].cpu().numpy().T, cmap='gray', origin='lower')
    axes[1, 0].set_title('Original T2')
    axes[1, 1].imshow(generated_images['fake_X'][0, 0].detach().cpu().numpy().T, cmap='gray', origin='lower')
    axes[1, 1].set_title('Generated T1')
    axes[1, 2].imshow(generated_images['rec_Y'][0, 0].detach().cpu().numpy().T, cmap='gray', origin='lower')
    axes[1, 2].set_title('Reconstructed T2')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.suptitle(f'Epoch {epoch}, Step {step}')
    save_path = os.path.join(vis_dir, f'progress_epoch_{epoch}_step_{step}.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved progress visualization to {save_path}")


def plot_history(history, out_dir=None):
    paired_metrics = [
        ('G_losses', 'val_G_losses', 'Generator Loss'),
        ('D_X_losses', 'val_D_X_losses', 'Discriminator X Loss'),
        ('D_Y_losses', 'val_D_Y_losses', 'Discriminator Y Loss'),
        ('cycle_losses', 'val_cycle_losses', 'Cycle Consistency Loss'),
        ('identity_losses', 'val_identity_losses', 'Identity Loss'),
    ]
    
    for train_key, val_key, title in paired_metrics:
        if train_key in history and val_key in history and history[train_key] and history[val_key]:
            epochs = range(1, len(history[train_key]) + 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history[train_key], 'b-', label='Training')
            plt.plot(epochs, history[val_key], 'r-', label='Validation')
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                save_path = os.path.join(out_dir, f'{train_key}_vs_{val_key}.png')
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Saved comparison plot to {save_path}")
            
            plt.close()
    
    val_metrics = [
        ('val_psnr', 'Average PSNR'),
        ('val_ssim', 'Average SSIM'),
        ('val_t1_to_t2_psnr', 'T1->T2 PSNR'),
        ('val_t1_to_t2_ssim', 'T1->T2 SSIM'),
        ('val_t2_to_t1_psnr', 'T2->T1 PSNR'),
        ('val_t2_to_t1_ssim', 'T2->T1 SSIM'),
    ]
    
    for key, title in val_metrics:
        if key in history and history[key]:
            epochs = range(1, len(history[key]) + 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history[key], 'g-')
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel('Value')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                save_path = os.path.join(out_dir, f'{key}.png')
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Saved metric plot to {save_path}")
            
            plt.close()
