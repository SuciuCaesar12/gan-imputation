from data import *
from config import settings
from models import *

from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from pathlib import Path
import numpy as np
import wandb
import torch


SPLITS = ['train', 'test', 'mi']
DEVICE = settings.device


def split_stage():
    if settings.flags.split:
        run = wandb.init(
            project=settings.wandb.project, 
            job_type='data', 
            name='split-stage',
            config=settings.mnist.model_dump()
        )
        
        datasets = load_stratified_split()
        
        data_splits = wandb.Artifact(
            name='data-splits', type='data',
            description='MNIST dataset split into `train`, `test` and `mi` (missing imputation) datasets'
        )
        
        for s, (data, labels) in zip(SPLITS, datasets):
            with data_splits.new_file(s + '.npz', mode='wb') as f:
                np.savez(f, data=data, labels=labels)
        
        run.log_artifact(data_splits)
        run.finish()
    else:
        print('[INFO] Skipping data split stage')


def missing_masks_stage():
    if settings.flags.missing_masks:
        run = wandb.init(
            project=settings.wandb.project, 
            job_type='data',
            name='missing-mask-stage'
        )
        
        datadir = Path(run.use_artifact('data-splits:latest').download())
        datasets = [np.load(datadir / f'{s}.npz') for s in SPLITS]
        
        masks_splits = wandb.Artifact(
            name='masks-splits', type='data',
            description='Missing data masks for each split dataset'
        )        

        for s, d in zip(SPLITS, datasets):
            if s == 'test':
                continue
            
            for method, func in MI_METHODS:
                n = d['data'].shape[0]
                masks = torch.stack([func() for _ in range(n)])
                
                with masks_splits.new_file(f'{s}/{method}.npz', mode='wb') as f:
                    np.savez(f, masks=masks.numpy())
        
        run.log_artifact(masks_splits)
        run.finish()
    else:
        print('[INFO] Skipping missing data masks stage')


def train_classifier_stage():
    if not settings.flags.original_classifier:
        print('[INFO] Skipping original classifier stage')
        return
    
    model = Classifier().to(settings.device)
    model.configure_optimizers()
    
    run = wandb.init(
        project=settings.wandb.project, 
        job_type='model',
        name='train-orig-cls-stage'
    )
    
    data_splits = run.use_artifact('data-splits:latest')
    train_dataset = np.load(Path(data_splits.download(path_prefix='train.npz')) / 'train.npz')
    train_dataset = MnistDataset(train_dataset['data'], train_dataset['labels'], 'train')
    train_loader = DataLoader(train_dataset, batch_size=settings.cls.train_batch_size, shuffle=True)
    
    test_dataset = np.load(Path(data_splits.download(path_prefix='test.npz')) / 'test.npz')
    test_dataset = MnistDataset(test_dataset['data'], test_dataset['labels'], 'test')
    test_loader = DataLoader(test_dataset, batch_size=settings.cls.test_batch_size, shuffle=False)

    step_cnt = 0
    for epoch in range(settings.cls.num_epochs):
        model.train()
        train_pba = tqdm(train_loader, postfix='Train Original Classifier', leave=False)
        
        for batch in train_pba:
            x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
            y_pred = model(x)
            
            model.opt.zero_grad()
            model.loss_ce(y, y_pred).backward()
            model.opt.step()
            
            if step_cnt % settings.cls.train_log_step == 0 and step_cnt > 0:
                name, result = model.loss_ce.name, model.loss_ce.result()
                run.log({f'train/{name}': result}, step=step_cnt)
            
            step_cnt += 1
        
        model.eval()
        test_pba = tqdm(test_loader, postfix='Test Original Classifier', leave=False)
        
        for batch in test_pba:
            with torch.no_grad():
                x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
                model.loss_ce(y, model(x))
        
        name, result = model.loss_ce.name, model.loss_ce.result()
        run.log({f'test/{name}': result, 'epoch': epoch})
    
    model_artifact = wandb.Artifact(
        name='orig-cls-model', 
        type='model', 
        description=f'Classifier trained on original dataset'
    )
    
    torch.save(model.state_dict(), 'orig-cls-model.pt')
    model_artifact.add_file('orig-cls-model.pt')
    run.log_artifact(model_artifact)
    run.finish()


@torch.no_grad()
def visualize_mi(run, train_dataset, hexagan, step):
    samples = train_dataset.sample()
    x, m = samples['x'].to(DEVICE), samples['m'].to(DEVICE)
    x_hat, _, x_tilde = hexagan.impute(x, m)
    
    caption = 'Top: Non-imputed, Middle: Imputed, Bottom: Original'
    imgs = torch.cat([x_tilde, x_hat, x], dim=0).view(-1, *settings.mnist.input_shape).cpu()
    
    grids = [
        make_grid(torch.stack([x, x_tilde, x_hat], dim=0), nrow=1, pad_value=1)
        for x, x_tilde, x_hat in zip(imgs[:NUM_CLASSES], imgs[NUM_CLASSES:2*NUM_CLASSES], imgs[2*NUM_CLASSES:])
    ]
    grid = make_grid(torch.stack(grids), nrow=len(grids), pad_value=1).permute(1, 2, 0)
    
    run.log({'train/visualizations': wandb.Image(grid.numpy(), caption=caption)}, step=step)


def train_hexagan():
    if not settings.flags.hexagan:
        print('[INFO] Skipping hexagan training stage')
        return

    for method, _ in MI_METHODS:
        run = wandb.init(
            project=settings.wandb.project,
            job_type='model',
            config=settings.hexa.model_dump(),
            name=f'train-hexagan-{method}'
        )
        run.config.update({'method': method})
        
        data_splits = run.use_artifact('data-splits:latest')
        masks_splits = run.use_artifact('masks-splits:latest')
        
        def load(split):
            dataset = np.load(Path(data_splits.download(path_prefix=f'{split}.npz')) / f'{split}.npz')
            masks = np.load(Path(masks_splits.download(path_prefix=f'{split}/{method}.npz')) / f'{split}/{method}.npz')
            return MnistDataset(
                name=split, flatten=True,
                data=dataset['data'], 
                labels=dataset['labels'],
                masks=masks['masks']
            )
        
        train_dataset, mi_dataset = map(load, ['train', 'mi'])
        train_loader = DataLoader(train_dataset, batch_size=settings.hexa.val_batch_size, shuffle=False)
        mi_loader = MissingDataImputationDataLoader(mi_dataset)
        hexagan = HexaGAN().to(DEVICE)
        hexagan.train()
        hexagan.configure_optimizers()
        
        def missing_data_imputation_train_step():
            hexagan.train()
            for _ in range(settings.hexa.n_critic):
                batch = mi_loader.get_disc_batch()
                x, m, y = batch['x'].to(DEVICE), batch['m'].to(DEVICE), batch['y'].to(DEVICE)
                
                x_hat, _, _ = hexagan.impute(x, m)
                d_prob = hexagan.discriminate(x_hat=x_hat, y=y)
                
                hexagan.discriminator_mi_opt.zero_grad()
                hexagan.loss_disc_data_mi(d_prob=d_prob[:, :-1], m=m).backward()
                hexagan.discriminator_mi_opt.step()
            
            batch = mi_loader.get_gen_batch()
            x, m, y= batch['x'].to(DEVICE), batch['m'].to(DEVICE), batch['y'].to(DEVICE)
            
            x_hat, _, _ = hexagan.impute(x, m)
            d_prob = hexagan.discriminate(x_hat, y)
            
            l_g_mi = hexagan.loss_gen_mi(d_prob=d_prob[:, :-1], m=m)
            l_rec = hexagan.loss_recon(x=x, x_hat=x_hat, m=m)
            loss = l_g_mi + settings.hexa.alpha1 * l_rec
            
            hexagan.generator_mi_opt.zero_grad()
            hexagan.encoder_opt.zero_grad()
            loss.backward()
            hexagan.generator_mi_opt.step()
            hexagan.encoder_opt.step()
        
        @torch.no_grad()
        def missing_data_imputation_val_epoch(epoch):
            hexagan.eval()
            for batch in tqdm(train_loader, desc=f'Val `{method}` Hexagan - Epoch: {epoch}', leave=False):
                x, m = batch['x'].to(DEVICE), batch['m'].to(DEVICE)
                x_hat, _, _ = hexagan.impute(x, m)
                hexagan.loss_recon(x, x_hat, m)
        
        step_cnt = 0
        for epoch in range(settings.hexa.num_epochs):
            mi_pba = tqdm(range(len(mi_loader)), desc=f"Train `{method}` Hexagan - Epoch: {epoch}", leave=False)
            for _ in mi_pba:
                missing_data_imputation_train_step()
                
                if step_cnt % settings.hexa.visualize_interval == 0 and step_cnt > 0:
                    visualize_mi(run, train_dataset, hexagan, step_cnt)
                
                if step_cnt % settings.hexa.train_log_interval == 0 and step_cnt > 0:
                    for loss in hexagan.losses():
                        name, result = loss.name, loss.result()
                        run.log({f'train/{name}': result}, step=step_cnt)
                step_cnt += 1
            
            missing_data_imputation_val_epoch(epoch)
            name, results = hexagan.loss_recon.name, hexagan.loss_recon.result()
            run.log({f'val/{name}': results, 'epoch': epoch})
        
        model_artifact = wandb.Artifact(
            name=f'hexagan-{method}', 
            type='model', 
            description=f'Hexagan trained on `{method}` dataset'
        )
        
        torch.save(hexagan.state_dict(), f'hexagan-{method}.pt')
        model_artifact.add_file(f'hexagan-{method}.pt')
        run.log_artifact(model_artifact)
        run.finish()


def impute():
    if not settings.flags.impute:
        print('[INFO] Skipping imputation stage')
        return
    
    run = wandb.init(
        project=settings.wandb.project,
        job_type='data',
        name='impute-stage',
        config=settings.impute.model_dump()
    )
    
    masks_splits = run.use_artifact('masks-splits:latest')
    data_splits = run.use_artifact('data-splits:latest')

    train_dataset = np.load(Path(data_splits.download(path_prefix='train.npz')) / 'train.npz')
    imputed_splits = wandb.Artifact(
        name='imputed-splits',
        type='data',
        description=f'Imputed train dataset for each method'
    )
    
    for method, _ in MI_METHODS:
        masks = np.load(Path(masks_splits.download(path_prefix=f'train/{method}.npz')) / f'train/{method}.npz')
        train_dataset_pt = MnistDataset(
            name='train', flatten=True,
            data=train_dataset['data'], 
            labels=train_dataset['labels'],
            masks=masks['masks']
        )
        train_loader = DataLoader(train_dataset_pt, batch_size=settings.impute.batch_size, shuffle=False)
        
        model_artifact = run.use_artifact(f'hexagan-{method}:latest')
        model_path = Path(model_artifact.download()) / f'hexagan-{method}.pt'
        hexagan = HexaGAN().to(DEVICE)
        hexagan.load_state_dict(torch.load(model_path))
        hexagan.eval()
        
        imputed_data = []
        for batch in tqdm(train_loader, desc=f'Impute `{method}` train dataset', leave=False):
            x, m = batch['x'].to(DEVICE), batch['m'].to(DEVICE)
            x_hat, _, _ = hexagan.impute(x, m)
            imputed_data.append(x_hat)
        
        imputed_data = torch.cat(imputed_data, dim=0) * 255.0
        imputed_data = imputed_data.cpu().view(imputed_data.shape[0], H, W).byte().numpy()
                
        with imputed_splits.new_file(f'train-{method}.npz', mode='wb') as f:
            np.savez(f, data=imputed_data)
    
    run.log_artifact(imputed_splits)
    run.finish()


def train_imputed_classifier():
    if not settings.flags.impute_classifier:
        print('[INFO] Skipping imputed classifier stage')
        return
    
    test_dataset, test_loader = None, None
    for method, _ in MI_METHODS:
        run = wandb.init(
            project=settings.wandb.project,
            job_type='model',
            config=settings.cls.model_dump(),
            name=f'train-imputed-cls-{method}'
        )
        imputed_splits = run.use_artifact('imputed-splits:latest')
        data_splits = run.use_artifact('data-splits:latest')
        
        train_labels = np.load(Path(data_splits.download(path_prefix='train.npz')) / 'train.npz')['labels']
        
        if test_dataset is None and test_loader is None:
            test_dataset = np.load(Path(data_splits.download(path_prefix='test.npz')) / 'test.npz')
            test_dataset = MnistDataset(test_dataset['data'], test_dataset['labels'], 'test')
            test_loader = DataLoader(test_dataset, batch_size=settings.cls.test_batch_size, shuffle=False)
        
        train_dataset = np.load(Path(imputed_splits.download(path_prefix=f'train-{method}.npz')) / f'train-{method}.npz')
        train_dataset = MnistDataset(train_dataset['data'], train_labels, 'train')
        train_loader = DataLoader(train_dataset, batch_size=settings.cls.train_batch_size, shuffle=True)
        
        model = Classifier().to(DEVICE)
        model.configure_optimizers()
        
        step_cnt = 0
        for epoch in range(settings.cls.num_epochs):
            model.train()
            train_pba = tqdm(train_loader, postfix=f'Train `{method}` Classifier', leave=False)
            
            for batch in train_pba:
                x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
                y_pred = model(x)
                
                model.opt.zero_grad()
                model.loss_ce(y, y_pred).backward()
                model.opt.step()
                
                if step_cnt % settings.cls.train_log_step == 0 and step_cnt > 0:
                    name, result = model.loss_ce.name, model.loss_ce.result()
                    run.log({f'train/{name}': result}, step=step_cnt)
                
                step_cnt += 1
            
            model.eval()
            test_pba = tqdm(test_loader, postfix='Test `{method}` Classifier', leave=False)
            
            for batch in test_pba:
                with torch.no_grad():
                    x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
                    model.loss_ce(y, model(x))
            
            name, result = model.loss_ce.name, model.loss_ce.result()
            run.log({f'test/{name}': result, 'epoch': epoch})
        
        model_artifact = wandb.Artifact(
            name=f'{method}-cls-model', 
            type='model', 
            description=f'Classifier trained on `{method}` imputed dataset'
        )
        
        torch.save(model.state_dict(), f'{method}-cls-model.pt')
        model_artifact.add_file(f'{method}-cls-model.pt')
        run.log_artifact(model_artifact)
        run.finish()


def evaluate_classifier():
    if not settings.flags.evaluate_classifier:
        print('[INFO] Skipping classifier evaluation stage')
        return
    
    # evaluate original classifier on test dataset
    table_results = wandb.Table(columns=['model', 'CE', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    run = wandb.init(
        project=settings.wandb.project,
        job_type='model',
        name='evaluation-stage',
        config=settings.cls.model_dump()
    )
    
    data_splits = run.use_artifact('data-splits:latest')
    test_dataset = np.load(Path(data_splits.download(path_prefix='test.npz')) / 'test.npz')
    test_dataset = MnistDataset(test_dataset['data'], test_dataset['labels'], 'test')
    test_loader = DataLoader(test_dataset, batch_size=settings.cls.test_batch_size, shuffle=False)
    test_pba = tqdm(test_loader, postfix='Test Original Classifier', leave=False)
    
    model_artifact = run.use_artifact('orig-cls-model:latest')
    model_path = Path(model_artifact.download()) / 'orig-cls-model.pt'
    model = Classifier().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Initialize metrics
    accuracy_metric = MulticlassAccuracy(average="macro", num_classes=10)
    precision_metric = MulticlassPrecision(average="macro", num_classes=10)
    recall_metric = MulticlassRecall(average="macro", num_classes=10)
    f1_metric = MulticlassF1Score(average="macro", num_classes=10)
    
    for batch in test_pba:
        with torch.no_grad():
            x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
            y_pred = model(x)
            
            # Compute loss
            model.loss_ce(y, y_pred)
            
            y = torch.argmax(y, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            # Update metrics
            accuracy_metric.update(y_pred, y)
            precision_metric.update(y_pred, y)
            recall_metric.update(y_pred, y)
            f1_metric.update(y_pred, y)
    
    # Get final metrics values
    accuracy = accuracy_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1_score = f1_metric.compute().item()
    
    name, result = model.loss_ce.name, model.loss_ce.result()
    table_results.add_data('orig', result, accuracy, precision, recall, f1_score)
    
    # Reset metrics before evaluating imputed classifiers
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    # evaluate imputed classifier on test dataset
    for method, _ in MI_METHODS:
        run = wandb.init(
            project=settings.wandb.project,
            job_type='model',
            name=f'evaluation-{method}-stage',
            config=settings.cls.model_dump()
        )
        model_artifact = run.use_artifact(f'{method}-cls-model:latest')
        model_path = Path(model_artifact.download()) / f'{method}-cls-model.pt'
        model = Classifier().to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        for batch in test_pba:
            with torch.no_grad():
                x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
                y_pred = model(x)
                
                # Compute loss
                model.loss_ce(y, y_pred)
                
                # Update metrics
                y = torch.argmax(y, dim=1)
                y_pred = torch.argmax(y_pred, dim=1)
                accuracy_metric.update(y_pred, y)
                precision_metric.update(y_pred, y)
                recall_metric.update(y_pred, y)
                f1_metric.update(y_pred, y)
        
        # Get final metrics values for the current method
        accuracy = accuracy_metric.compute().item()
        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        f1_score = f1_metric.compute().item()
        
        name, result = model.loss_ce.name, model.loss_ce.result()
        table_results.add_data(f'{method}', result, accuracy, precision, recall, f1_score)
        
        # Reset metrics for the next classifier
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

    run.log({"results": table_results})
    run.finish()


def main():
    split_stage()
    missing_masks_stage()
    train_classifier_stage()
    train_hexagan()
    impute()
    train_imputed_classifier()
    evaluate_classifier()


if __name__ == '__main__':
    main()
