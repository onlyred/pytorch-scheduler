import torch

class BasicModel:
    def __init__(self, lr=100):
        self.model  = torch.nn.Linear(2,1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def get_scheduler(self, scheduler, **kwargs):
        return scheduler(self.optimizer, **kwargs)

    def get_optim(self):
        return self.optimizer
