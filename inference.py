import os
import random
import torch
import matplotlib.pyplot as plt
from models.cyclegan import CycleGAN
from dataset import MRIT1T2Dataset
from utils.seed_utils import set_seed


def generate_from_t1(model, t1_tensor, device='cpu'):
    model.eval()
    with torch.no_grad():
        t1_tensor = t1_tensor.to(device)
        fake_t2 = model.G_XtoY(t1_tensor)
        rec_t1 = model.G_YtoX(fake_t2 )
    return fake_t2, rec_t1 


def generate_from_t2(model, t2_tensor, device='cpu'):
    model.eval()
    with torch.no_grad():
        t2_tensor = t2_tensor.to(device)
        fake_t1 = model.G_YtoX(t2_tensor)
        rec_t2 = model.G_XtoY(fake_t1)
    return fake_t1, rec_t2


def visualize_inference(model, real_t1_tensor, real_t2_tensor, device='cpu', save_path=None):
    model.eval()
    with torch.no_grad():
        fake_t2, rec_t1 = generate_from_t1(model, real_t1_tensor, device=device)
        fake_t1, rec_t2 = generate_from_t2(model, real_t2_tensor, device=device)

    real_t1 = real_t1_tensor.cpu().numpy()[0, 0]
    real_t2 = real_t2_tensor.cpu().numpy()[0, 0]
    fake_t2 = fake_t2.cpu().numpy()[0, 0]
    fake_t1 = fake_t1.cpu().numpy()[0, 0]
    rec_t1 = rec_t1.cpu().numpy()[0, 0]
    rec_t2 = rec_t2.cpu().numpy()[0, 0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    
    # T1 → T2 mapping: Real T1, Generated T2, Reconstructed T1
    axes[0, 0].imshow(real_t1.T, cmap='gray', origin='lower')
    axes[0, 0].set_title('Real T1')
    
    axes[0, 1].imshow(fake_t2.T, cmap='gray', origin='lower')
    axes[0, 1].set_title('Generated T2')

    axes[0, 2].imshow(rec_t1.T, cmap='gray', origin='lower')
    axes[0, 2].set_title('Reconstructed T1')
    
    # T2 → T1 mapping: Real T2, Generated T1, Reconstructed T2
    axes[1, 0].imshow(real_t2.T, cmap='gray', origin='lower')
    axes[1, 0].set_title('Real T2')
    
    axes[1, 1].imshow(fake_t1.T, cmap='gray', origin='lower')
    axes[1, 1].set_title('Generated T1')

    axes[1, 2].imshow(rec_t2.T, cmap='gray', origin='lower')
    axes[1, 2].set_title('Reconstructed T2')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu", generator_type: str = "resnet"):
    model = CycleGAN(generator_type=generator_type, device=device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    except FileNotFoundError as e:
        print(f"Falied to load the checkpoint")
        raise e
    
    model.G_XtoY.load_state_dict(checkpoint['G_XtoY_state'])
    model.G_YtoX.load_state_dict(checkpoint['G_YtoX_state'])

    model.D_X = None
    model.D_Y = None
    model.opt_G = None
    model.opt_D_X = None
    model.opt_D_Y = None
    model.scheduler_G = None
    model.scheduler_D_X = None
    model.scheduler_D_Y = None
    model.criterion_GAN = None

    model.eval()
    model.to(device)
    return model


def main():
    set_seed(42)

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    generator_type = "resnet"

    # Load best model
    checkpoint_dir = f"../checkpoints/cyclegan/{generator_type}/best_model"
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    model = load_model_from_checkpoint(best_model_path, device=device, generator_type=generator_type)
    
    dataset = MRIT1T2Dataset(
        t1_dir='./data/IXI-T1',
        t2_dir='./data/IXI-T2',
        slice_mode='middle',
        paired=True,
        transform=None,
        cache_size=2
    )
    indices = random.sample(range(len(dataset)), 5)

    save_dir = f"../visualizations/cyclegan/{generator_type}/inference_best_model"
    os.makedirs(save_dir, exist_ok=True)

    for i in indices:
        sample = dataset[i]
        real_t1_tensor = sample["T1"].unsqueeze(0)
        real_t2_tensor = sample["T2"].unsqueeze(0)

        save_path = os.path.join(save_dir, f"inference_sample_{i}.png")
        visualize_inference(model, real_t1_tensor, real_t2_tensor, device=device, save_path=save_path)

if __name__ == "__main__":
    main()
