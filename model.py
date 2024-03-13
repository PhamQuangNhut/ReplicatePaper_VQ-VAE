import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
from torch import nn
import wandb

class ResidualBlocK(nn.Module) :
  def __init__(self, hidden_units) :
    super().__init__()
    self.relu1 = nn.ReLU()
    self.conv1 = nn.Conv2d(in_channels=hidden_units,
                           out_channels=hidden_units,
                           kernel_size=3, padding=1)
    self.relu2 = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels=hidden_units,
                           out_channels=hidden_units,
                           kernel_size=1)
  def forward(self, x) :
    _x = self.conv1(self.relu1(x))
    _x = self.conv2(self.relu2(_x))
    return x + _x
class VQVAE(pl.LightningModule) :
  def __init__(self,
               input_size: int = 64,
               hidden_units: int = 256,
               latent_size: int = 3,
               emb_size: int = 2,
               beta: int = 0.2,
               lr: float = 2e-4
               ) :
    super().__init__()
    self.save_hyperparameters()
    self.loss_fn = nn.MSELoss()
    self.beta = beta
    self.lr = lr
    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=3,
                  out_channels=16,
                  kernel_size=4,
                  stride=2,
                  padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(in_channels=16,
                  out_channels=4,
                  kernel_size=4,
                  stride=2,
                  padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        # ResidualBlocK(hidden_units),
        # ResidualBlocK(hidden_units)
    )
    self.pre_quant_conv = nn.Conv2d(in_channels=4,
                                    out_channels=emb_size,
                                    kernel_size=1)
    self.emb = nn.Embedding(num_embeddings=latent_size, embedding_dim=emb_size)
    self.post_quant_conv = nn.Conv2d(in_channels=emb_size,
                                     out_channels=4,
                                     kernel_size=1)
    self.decoder = nn.Sequential(
        # ResidualBlocK(hidden_units),
        # ResidualBlocK(hidden_units),
        nn.ConvTranspose2d(in_channels=4,
                  out_channels=16,
                  kernel_size=4,
                  stride=2,
                  padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=16,
                  out_channels=3,
                  kernel_size=4,
                  stride=2,
                  padding=1),
        nn.Tanh()
    )
  def forward(self, x) :
    x = self.encoder(x)
    x = self.pre_quant_conv(x)
    b, c, h, w = x.shape
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(x.size(0), -1, x.size(-1))
    emb_w = self.emb.weight[None:].repeat((x.size(0), 1, 1))
    dist = torch.cdist(x, emb_w)
    min_encoding_indices = torch.argmin(dist, dim=-1)
    quant_out = torch.index_select(input=self.emb.weight, dim=0, index=min_encoding_indices.view(-1))
    x = x.reshape((-1, x.size(-1)))
    commitment_loss = torch.mean((x.detach() - quant_out)**2)
    codebook_loss = torch.mean((x - quant_out.detach())**2)
    vq_loss = codebook_loss + commitment_loss * self.beta
    quant_out = x + (quant_out - x).detach()
    quant_out = quant_out.reshape(b, h, w, c)
    quant_out = quant_out.permute(0, 3, 1, 2)
    quant_out = self.post_quant_conv(quant_out)
    output = self.decoder(quant_out)
    
    return output, vq_loss, quant_out
  def training_step(self, batch, batch_idx):
    output, vq_loss, quant_out = self._common_step(batch, batch_idx)
    wandb.log({"Training Loss": vq_loss + self.loss_fn(output, batch)})
    return vq_loss + self.loss_fn(output, batch)
  def validation_step(self, batch, batch_idx) :
    output, vq_loss, quant_out = self._common_step(batch, batch_idx)
    # to_pil = transforms.ToPILImage()
    # columns = ["inputs", "reconstructions"]
    # img1 = to_pil(batch[0])
    # img2 = to_pil(output[0])
    # my_data = [
    #     ["inputs", wandb.Image(batch)],
    #     ["reconstructions", wandb.Image(output)],
    # ]
    # wandb_logger.log_table(key="my_samples", columns=columns, data=my_data)
    if batch_idx % 1000 == 0 :
      wandb.log({
                "output": [wandb.Image(output[0])]
                }, commit=False)
    wandb.log({"Validating Loss": vq_loss + self.loss_fn(output, batch)})
    return vq_loss + self.loss_fn(output, batch)
  def test_step(self, batch, batch_idx) :
    output, vq_loss, quant_out = self._common_step(batch, batch_idx)
    wandb.log({"Test Loss": vq_loss + self.loss_fn(output, batch)})
    return vq_loss + self.loss_fn(output, batch)
  def _common_step(self, batch, batch_idx) :
    x = batch
    output, vq_loss, quant_out = self.forward(x)

    return output, vq_loss, quant_out
  def configure_optimizers(self) :
    return torch.optim.Adam(params=self.parameters(), lr = self.lr)
  def on_epoch_end(self):
        # Log images
    sample_input = torch.randn(1, 3, self.hparams.input_size, self.hparams.input_size)
    sample_output, _, sample_latent = self.forward(sample_input)

    sample_input_grid = torchvision.utils.make_grid(sample_input)
    sample_output_grid = torchvision.utils.make_grid(sample_output)
    # sample_latent_grid = torchvision.utils.make_grid(sample_latent)

    wandb.log({"Sample Input": [wandb.Image(sample_input_grid, caption="Sample Input")]})
    wandb.log({"Sample Output": [wandb.Image(sample_output_grid, caption="Sample Output")]})
    # wandb.log({"Sample Latent Space": [wandb.Image(sample_latent_grid, caption="Sample Latent Space")]})
    
class ResnetVQVAE(pl.LightningModule) :
  def __init__(self,
               input_size: int = 64,
               hidden_units: int = 256,
               latent_size: int = 3,
               emb_size: int = 2,
               beta: int = 0.2,
               lr: float = 2e-4
               ) :
    super().__init__()
    self.loss_fn = nn.MSELoss()
    self.save_hyperparameters()
    self.beta = beta
    self.lr = lr
    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=3,
                  out_channels=hidden_units,
                  kernel_size=4,
                  stride=2,
                  padding=1),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=4,
                  stride=2,
                  padding=1),
        ResidualBlocK(hidden_units),
        ResidualBlocK(hidden_units)
    )
    self.pre_quant_conv = nn.Conv2d(in_channels=hidden_units,
                                    out_channels=emb_size,
                                    kernel_size=1)
    self.emb = nn.Embedding(num_embeddings=latent_size, embedding_dim=emb_size)
    self.post_quant_conv = nn.Conv2d(in_channels=emb_size,
                                     out_channels=hidden_units,
                                     kernel_size=1)
    self.decoder = nn.Sequential(
        ResidualBlocK(hidden_units),
        ResidualBlocK(hidden_units),
        nn.ConvTranspose2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=4,
                  stride=2,
                  padding=1),
        nn.ConvTranspose2d(in_channels=hidden_units,
                  out_channels=3,
                  kernel_size=4,
                  stride=2,
                  padding=1),
    )
  def forward(self, x) :
    x = self.encoder(x)
    x = self.pre_quant_conv(x)
    b, c, h, w = x.shape
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(x.size(0), -1, x.size(-1))
    emb_w = self.emb.weight[None:].repeat((x.size(0), 1, 1))
    dist = torch.cdist(x, emb_w)
    min_encoding_indices = torch.argmin(dist, dim=-1)
    quant_out = torch.index_select(input=self.emb.weight, dim=0, index=min_encoding_indices.view(-1))
    x = x.reshape((-1, x.size(-1)))
    commitment_loss = torch.mean((x.detach() - quant_out)**2)
    codebook_loss = torch.mean((x - quant_out.detach())**2)
    vq_loss = codebook_loss + commitment_loss * self.beta
    quant_out = x + (quant_out - x).detach()
    quant_out = quant_out.reshape(b, h, w, c)
    quant_out = quant_out.permute(0, 3, 1, 2)
    quant_out = self.post_quant_conv(quant_out)
    output = self.decoder(quant_out)
    return output, vq_loss, quant_out
  def training_step(self, batch, batch_idx):
    output, vq_loss, quant_out = self._common_step(batch, batch_idx)
    wandb.log({"Training Loss": self.loss_fn(output, batch) + vq_loss * self.beta})
    return self.loss_fn(output, batch) + vq_loss * self.beta
    # wandb.log({"Training Loss": vq_loss })
    # return vq_loss 
  def validation_step(self, batch, batch_idx) :
    output, vq_loss, quant_out = self._common_step(batch, batch_idx)
    # to_pil = transforms.ToPILImage()
    # columns = ["inputs", "reconstructions"]
    # img1 = to_pil(batch[0])
    # img2 = to_pil(output[0])
    # my_data = [
    #     ["inputs", wandb.Image(batch)],
    #     ["reconstructions", wandb.Image(output)],
    # ]
    # wandb_logger.log_table(key="my_samples", columns=columns, data=my_data)
    if batch_idx % 10000 == 0 :
      visualize_data = torch.cat((batch[0:4], output[0:4]), dim=0)
      wandb.log({
                "visualize": [wandb.Image(visualize_data)]
                }, commit=False)
    # wandb.log({"Validating Loss": vq_loss + self.loss_fn(output, batch)})
    # return vq_loss + self.loss_fn(output, batch)
    wandb.log({"Validating Loss": self.loss_fn(output, batch) + vq_loss * self.beta})
    return self.loss_fn(output, batch) + vq_loss * self.beta
  def test_step(self, batch, batch_idx) :
    output, vq_loss, quant_out = self._common_step(batch, batch_idx)
    wandb.log({"Testing Loss": self.loss_fn(output, batch) + vq_loss * self.beta})
    return self.loss_fn(output, batch) + vq_loss * self.beta
    # wandb.log({"Test Loss": vq_loss + self.loss_fn(output, batch)})
    # return vq_loss + self.loss_fn(output, batch)
  def _common_step(self, batch, batch_idx) :
    x = batch
    output, vq_loss, quant_out = self.forward(x)

    return output, vq_loss, quant_out
  def configure_optimizers(self) :
    return torch.optim.Adam(params=self.parameters(), lr = self.lr)
  # def on_epoch_end(self):
  #       # Log images
  #   sample_input = torch.randn(1, 3, self.hparams.input_size, self.hparams.input_size)
  #   sample_output, _, sample_latent = self.forward(sample_input)

  #   sample_input_grid = torchvision.utils.make_grid(sample_input)
  #   sample_output_grid = torchvision.utils.make_grid(sample_output)
  #   # sample_latent_grid = torchvision.utils.make_grid(sample_latent)

  #   wandb.log({"Sample Input": [wandb.Image(sample_input_grid, caption="Sample Input")]})
  #   wandb.log({"Sample Output": [wandb.Image(sample_output_grid, caption="Sample Output")]})
    # wandb.log({"Sample Latent Space": [wandb.Image(sample_latent_grid, caption="Sample Latent Space")]})