import torch
from torchvision.transforms import functional as F
import torchvision
import math
from torch import Tensor
from typing import List, Optional, Tuple
import random


class CustomRandomErasing(torchvision.transforms.RandomErasing):
    def __init__(self, normalize, p, value='random'):
        super(CustomRandomErasing, self).__init__(p=p, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=value,
                                                  inplace=False)
        self.normalize = normalize

    #@staticmethod
    def get_params(self,
            img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[List[float]] = None
    ):

        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                if self.normalize == True:
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                else:
                    v = torch.empty([img_c, h, w], dtype=torch.float32).uniform_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [float(self.value)]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, (list, tuple)):
                value = [float(v) for v in self.value]
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{img.shape[-3]} (number of input channels)"
                )
            print(self.scale, self.ratio, value)
            x, y, h, w, v = self.get_params(self, img, scale=self.scale, ratio=self.ratio, value=value)
            return F.erase(img, x, y, h, w, v, self.inplace)
        return img

class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor, robust_samples: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        #if batch.ndim != 4:
        #    raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        target_rolled = target.roll(1, 0)
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        batches = batch.view(robust_samples + 1, -1, batch.size()[1], batch.size()[2], batch.size()[3])
        for id, b in enumerate(batches):

            # It's faster to roll the batch by one instead of shuffling it to create image pairs
            batch_rolled = batch.roll(1, 0)

            # Implemented as on mixup paper, page 3.
            batch_rolled.mul_(1.0 - lambda_param)
            batch.mul_(lambda_param).add_(batch_rolled)

            batches[id] = b

        batch = batches.view(-1, batch.size()[1], batch.size()[2], batch.size()[3])

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("Please provide a valid positive value for the num_classes.")
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor, robust_samples: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
            robust_samples (int): Integer for the number of concatenated similar images in the batch

        Returns:
            Tensor: Randomly transformed batch.
        """
        #if batch.ndim != 4:
        #    raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = F.get_image_size(batch)

        target_rolled = target.roll(1, 0)
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        batches = batch.view(robust_samples + 1, -1, batch.size()[1], batch.size()[2], batch.size()[3])
        for id, b in enumerate(batches):

            #It's faster to roll the batch by one instead of shuffling it to create image pairs
            batch_rolled = b.roll(1, 0)

            r_x = torch.randint(W, (1,))
            r_y = torch.randint(H, (1,))

            r = 0.5 * math.sqrt(1.0 - lambda_param)
            r_w_half = int(r * W)
            r_h_half = int(r * H)

            x1 = int(torch.clamp(r_x - r_w_half, min=0))
            y1 = int(torch.clamp(r_y - r_h_half, min=0))
            x2 = int(torch.clamp(r_x + r_w_half, max=W))
            y2 = int(torch.clamp(r_y + r_h_half, max=H))

            b[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
            lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))
            batches[id] = b

        batch = batches.view(-1, batch.size()[1], batch.size()[2], batch.size()[3])

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

def mixup_process(inputs, targets, robust_samples, num_classes, mixup_alpha, mixup_p, cutmix_alpha, cutmix_p,
                  random_erase_p, normalize, manifold):

    if manifold==True and mixup_alpha > 0.0:
        mixupcutmix = RandomMixup(num_classes, p=mixup_p, alpha=mixup_alpha)
        inputs, targets = mixupcutmix(inputs, targets, robust_samples)
    elif (cutmix_alpha or cutmix_p) == 0 and (mixup_alpha or mixup_p) == 0 and random_erase_p == 0:
        return inputs, targets
    else:
        total_probability = cutmix_p + mixup_p + random_erase_p
        if total_probability > 1:
            cutmix_p /= total_probability
            mixup_p /= total_probability
            random_erase_p /= total_probability

        random_number = random.uniform(0, 1)

        if random_number < cutmix_p:
            mixupcutmix = RandomCutmix(num_classes, p=1.0, alpha=cutmix_alpha)
            inputs, targets = mixupcutmix(inputs, targets, robust_samples)
        elif random_number < cutmix_p + mixup_p:
            mixupcutmix = RandomMixup(num_classes, p=1.0, alpha=mixup_alpha)
            inputs, targets = mixupcutmix(inputs, targets, robust_samples)
        elif random_number < cutmix_p + mixup_p + random_erase_p:
            mixupcutmix = CustomRandomErasing(normalize, p=1.0, value='random')
            original_images = inputs.size(0) / (robust_samples + 1)
            inputs = mixupcutmix(inputs[original_images:,:,:,:])
        else:
            return inputs, targets

    return inputs, targets