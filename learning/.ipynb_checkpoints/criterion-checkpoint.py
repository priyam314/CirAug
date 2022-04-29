import torch.nn as nn
import torch
from config import ConfigSeq

class Criterion:
    def __init__(self, config: ConfigSeq):
        
        self.config = config
        self.criteria = self.config.criteria.name
        self.lambda_coeff = self.config.criteria.lamda
        self.batch_size = self.get_batch_size()
        self.t = self.config.criteria.temp
    
    def get_batch_size(self):
        if self.config.exec.typ == "train": return self.config.batch.train_size
        elif self.config.exec.typ == "fine-tune": return self.config.batch.val_size
        elif self.config.exec.typ == "test": return self.config.batch.test_size
    
    def __call__(self, p_aug, p_orig):
        if self.criteria == "bce":
            self.bce = nn.BCEWithLogitsLoss()
            return self.bce(p_aug, p_orig)
        elif self.criteria == "ntxent":
            self.ntxent = NTXent(self.t)
            return self.ntxent(p_aug, p_orig)
        elif self.criteria == "cel":
            self.cel = nn.CrossEntropyLoss()
            return self.cel(p_aug, p_orig)
        elif self.criteria == "barlow":
            self.barlow = BarlowTwins(self.lambda_coeff, self.batch_size)
            return self.barlow(p_aug.float(), p_orig.float())
        elif self.criteria == "kldiv":
            lg_sft = nn.functional.log_softmax(p_aug.float())
            sft = nn.functional.softmax(nn.functional.one_hot(p_orig, num_classes=lg_sft.shape[1]).float())
            return nn.functional.kl_div(lg_sft, sft, reduction='batchmean')

class NTXent(nn.Module):
    def __init__(self, temp = 0.05):
        super(NTXent, self).__init__()
        
        self.t = temp
        
    def forward(self, z1, z2):
        
        p_aug_norm = z1 / z1.norm(dim=1)[:, None]
        p_orig_norm = z2 / z2.norm(dim=1)[:, None]
        cat = torch.cat([p_aug_norm, p_orig_norm], axis=0)
        n_samples = len(cat)
        sim = torch.mm(cat, cat.t().contiguous())
        ex = torch.exp(sim/self.t)
        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = ex.masked_select(mask).view(n_samples, -1).sum(dim=-1)
        pos = torch.exp(torch.sum(p_aug_norm*p_orig_norm, dim=-1)/self.t)
        pos = torch.cat([pos, pos], dim=0)
        return -torch.log(pos/neg).mean()

class BarlowTwins(nn.Module):
    def __init__(self, lambda_coeff = 0.01, batch_size=256):
        super(BarlowTwins, self).__init__()
        
        self.batch_size= batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):

        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag
        
        