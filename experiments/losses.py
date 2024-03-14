import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
import torch.optim as optim

class Trades(nn.Module):
    """
    TRADES loss for training adversarially robust models
    based on https://github.com/yaodongyu/TRADES built into class
    """

    def __init__(self, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
        super().__init__()
        self.model = None
        self.optimizer = None
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.distance = distance

    def update(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def __call__(self, x_natural, y, outputs, mixed_targets):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(size_average=False)
        self.model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        logits_natural = self.model(x_natural)
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(self.model(x_adv), dim=1),
                                           F.softmax(logits_natural, dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif self.distance == 'l_2':
            delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=self.epsilon / self.perturb_steps * 2)

            for _ in range(self.perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(self.model(adv), dim=1),
                                               F.softmax(self.model(x_natural), dim=1))
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=self.epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        self.optimizer.zero_grad()
        # calculate robust loss
        with torch.enable_grad():
            loss_natural = F.cross_entropy(outputs, mixed_targets)
            loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(self.model(x_adv), dim=1),
                                                        F.softmax(logits_natural, dim=1))
            loss = loss_natural + self.beta * loss_robust

        self.model.train()
        return loss

class JsdCrossEntropy(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss

    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781

    Hacked together by / Copyright 2020 Ross Wightman
    """
    def __init__(self, num_splits=3, alpha=12, smoothing=0.0):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        self.cross_entropy_loss = CrossEntropyLoss(label_smoothing=smoothing)

    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = torch.split(output, split_size)

        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]

        # Clamp mixture distribution to avoid exploding KL divergence
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
        loss += self.alpha * sum([F.kl_div(
            logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / self.num_splits
        return loss

