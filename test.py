from model import VQVAE, ResnetVQVAE
import pytorch_lightning as pl
import torch
from torchvision.utils import save_image
import config
# model = ResnetVQVAE()
model = ResnetVQVAE.load_from_checkpoint('./lightning_logs/cmdw9nn2/checkpoints/epoch=599-step=94200.ckpt',
                                                 input_size=config.input_size,
                                                 hidden_units=config.hidden_units,
                                                 latent_size=config.latent_size,
                                                 emb_size=config.emb_size,
                                                 beta=config.beta,
                                                 lr=config.lr)
rd_tensor =torch.randn(1, 3, 32, 32)
# out = model(rd_tensor)
if next(model.parameters()).is_cuda:
    rd_tensor = rd_tensor.cuda()

# Disable gradient computation since you're only doing inference
with torch.no_grad():
    out, _, _ = model(rd_tensor)

# Save the image
# If `out` has more than one image, you'll need to index it, e.g., out[0]
save_image(out.squeeze(0), 'output_image.png')
