from models import HexaGAN
from data import MissingDataImputationDataLoader
from config import settings
from losses import LossStages

from torchvision.utils import make_grid
import tqdm
import torch
import wandb
import math

hexagan = HexaGAN()
train_dataloader = MissingDataImputationDataLoader(train=True)
val_dataloader = MissingDataImputationDataLoader(train=False)
loss_stages = LossStages()

NUM_CLASSES = settings.model.num_classes
C, H, W = INPUT_SHAPE = settings.model.input_shape
N_FEATURES = C * H * W
DEVICE = settings.training.device

GLOBAL_EPOCH = 0
TRAIN_GLOBAL_STEP = 0
BEST_VAL_SCORE = math.inf

if settings.flags.use_wandb:
    wandb.init(project=settings.wandb.project)
    wandb.config.update(settings.model_dump())

def train_missing_data_imputation():
    loss_disc_data, loss_gen_mi, loss_reconstruct = loss_stages.missing_data_imputation
    
    for _ in range(settings.training.n_critic):
        batch = train_dataloader.get_disc_batch()
        x, m, y = batch['x'].to(DEVICE), batch['m'].to(DEVICE), batch['y'].to(DEVICE)
        
        x_hat, _, _ = hexagan.impute(x, m)
        d_prob = hexagan.discriminate(x_hat=x_hat, y=y)

        loss = loss_disc_data(d_prob=d_prob[:, :-1], m=m)
        
        hexagan.discriminator_mi_opt.zero_grad()
        loss.backward()
        hexagan.discriminator_mi_opt.step()
    
    batch = train_dataloader.get_gen_batch()
    x, m, y= batch['x'].to(DEVICE), batch['m'].to(DEVICE), batch['y'].to(DEVICE)
    
    x_hat, _, _ = hexagan.impute(x, m)
    d_prob = hexagan.discriminate(x_hat, y)
    
    l_g_mi = loss_gen_mi(d_prob=d_prob[:, :-1], m=m)
    l_rec = loss_reconstruct(x=x, x_hat=x_hat, m=m)
    loss = l_g_mi + settings.training.alpha1 * l_rec
    
    hexagan.generator_mi_opt.zero_grad()
    hexagan.encoder_opt.zero_grad()
    loss.backward()
    hexagan.generator_mi_opt.step()
    hexagan.encoder_opt.step()

@torch.no_grad()
def val_missing_data_imputation():
    loss_disc_data, loss_gen_mi, loss_reconstruct = loss_stages.missing_data_imputation
    
    for batch in tqdm.tqdm(iter(val_dataloader.gen_loader.loader), desc="Validation"):
        x, m, y = batch['x'].to(DEVICE), batch['m'].to(DEVICE), batch['y'].to(DEVICE)
        
        x_hat, _, _ = hexagan.impute(x, m)
        d_prob = hexagan.discriminate(x_hat=x_hat, y=y)

        loss_disc_data(d_prob=d_prob[:, :-1], m=m)
        loss_gen_mi(d_prob=d_prob[:, :-1], m=m)
        loss_reconstruct(x=x, x_hat=x_hat, m=m)


@torch.no_grad()
def visualize_mi():
    if settings.flags.use_wandb:
        samples = val_dataloader.dataset.sample()
        x, m = samples['x'].to(DEVICE), samples['m'].to(DEVICE)
        x_hat, _, x_tilde = hexagan.impute(x, m)
        
        caption = 'Top: original, Middle: non-imputed, Bottom: imputed'
        imgs = torch.cat([x, x_tilde, x_hat], dim=0).view(-1, *settings.model.input_shape).cpu()
        
        grids = [
            make_grid(torch.stack([x, x_tilde, x_hat], dim=0), nrow=1, pad_value=1)
            for x, x_tilde, x_hat in zip(imgs[:NUM_CLASSES], imgs[NUM_CLASSES:2*NUM_CLASSES], imgs[2*NUM_CLASSES:])
        ]
        grid = make_grid(torch.stack(grids), nrow=len(grids), pad_value=1).permute(1, 2, 0)
        
        wandb.log({'train/visualizations/MI': wandb.Image(grid.numpy(), caption=caption)}, step=TRAIN_GLOBAL_STEP)

def main():
    global GLOBAL_EPOCH, TRAIN_GLOBAL_STEP, BEST_VAL_SCORE
    hexagan.to(DEVICE)
    hexagan.configure_optimizers()
    
    if settings.flags.use_wandb:
        wandb.watch(hexagan)
    
    for _ in range(settings.training.num_epochs):
        hexagan.train()
        for _ in tqdm.tqdm(range(len(train_dataloader)), desc="Training"):
            if settings.flags.missing_imputation:
                train_missing_data_imputation()

            if settings.flags.use_wandb:
                if TRAIN_GLOBAL_STEP % settings.wandb.train_log_interval == 0 and TRAIN_GLOBAL_STEP != 0:
                    for loss in loss_stages.missing_data_imputation:
                        name, results = loss.name, loss.result()
                        wandb.log({f'train/{name}': results}, step=TRAIN_GLOBAL_STEP)

                if TRAIN_GLOBAL_STEP % (settings.wandb.train_log_interval * 10) == 0 and TRAIN_GLOBAL_STEP != 0:
                    visualize_mi()
            
            TRAIN_GLOBAL_STEP += 1
        
        hexagan.eval()
        if settings.flags.missing_imputation:
            val_missing_data_imputation()
            epoch_score = 0
            for loss in loss_stages.missing_data_imputation:
                name, results = loss.name, loss.result()
                epoch_score += results
                wandb.log({'epoch': GLOBAL_EPOCH, f'val/{name}': results})
        
        if epoch_score < BEST_VAL_SCORE:
            if settings.flags.use_wandb:
                if settings.wandb.save_model:
                    wandb.save(hexagan.state_dict())
        
        GLOBAL_EPOCH += 1
    
    if settings.flags.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
