import torch


class BaseLoss:
    def __init__(self, name=""):
        self.name = name 
        self.losses = []

    def __call__(self, *args, **kwargs):
        value = self._calculate_loss(*args, **kwargs)
        self.losses.append(value.item())
        return value

    def result(self, reset: bool = True):
        mean_loss = sum(self.losses) / len(self.losses) if self.losses else None
        if reset:
            self.reset()
            
        return mean_loss

    def reset(self):
        self.losses = []

    def _calculate_loss(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class LossCE(BaseLoss):
    def __init__(self, name='CE'):
        super().__init__(name)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    
    def _calculate_loss(self, y, y_hat):
        return self.criterion(y_hat, y)


class LossDiscDataMI(BaseLoss):
    def __init__(self, name="disc_data_mi"):
        super().__init__(name)
    
    def _calculate_loss(self, d_prob, m):
        return -torch.mean(d_prob * m) + torch.mean((1 - m) * d_prob)


class LossGenMI(BaseLoss):
    def __init__(self, name="loss_gen_mi"):
        super().__init__(name)
    
    def _calculate_loss(self, d_prob, m):
        return -torch.mean(d_prob * (1 - m))


class LossReconstruct(BaseLoss):
    def __init__(self, name="loss_recon"):
        super().__init__(name)
        
    def _calculate_loss(self, x, x_hat, m):
        return ((1 - m) * ((x - x_hat) ** 2)).sum() / ((1 - m).sum() + 1e-8)