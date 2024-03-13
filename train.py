import pytorch_lightning as pl
from dataset import DataModule
from model import VQVAE, ResnetVQVAE
import config
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb

dm = DataModule(train_dir=config.train_dir,
                     test_dir=config.test_dir, 
                     transform=config.transform, 
                     batch_size=config.batch_size, 
                     num_workers=config.num_workers)
model = ResnetVQVAE(input_size=config.input_size,
              hidden_units=config.hidden_units,
              latent_size=config.latent_size,
              emb_size=config.emb_size,
              beta=config.beta,
              lr=config.lr).to(config.device)
if __name__ == "__main__" : 
     wandb.init(project='VQ-VAE')
     wandb_logger = WandbLogger(log_model="all")
     trainer = pl.Trainer(accelerator = config.accelerator,
                         min_epochs=config.min_epochs,
                         max_epochs=config.max_epochs,
                         logger = wandb_logger

                         #  callbacks = [MyPrintingCallBack(), EarlyStopping(monitor="val_loss")]
                         )
     trainer.fit(model, dm)
     trainer.validate(model, dm)
     trainer.test(model, dm)
     wandb.finish()