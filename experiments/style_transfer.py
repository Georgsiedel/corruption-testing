import os
import sys
import numpy as np

current_dir = os.path.dirname(__file__)
module_path = os.path.abspath(current_dir)
import time

if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import adaIN.model as adaINmodel
import adaIN.utils as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder_rel_path = 'adaIN/vgg_normalised.pth'
decoder_rel_path = 'adaIN/decoder.pth'
style_feats_rel_path = 'adaIN/style_feats_adain_1000.npy'
encoder_path = os.path.join(current_dir, encoder_rel_path)
decoder_path = os.path.join(current_dir, decoder_rel_path)
style_feats_path = os.path.join(current_dir, style_feats_rel_path)

def load_models():

    vgg = adaINmodel.vgg
    decoder = adaINmodel.decoder
    vgg.load_state_dict(torch.load(encoder_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    decoder.load_state_dict(torch.load(decoder_path))

    vgg.to(device)
    decoder.to(device)

    vgg.eval()
    decoder.eval()
    return vgg, decoder

def load_feat_files():

    style_feats_np = np.load(style_feats_path)
    style_feats_tensor = torch.from_numpy(style_feats_np)
    style_feats_tensor = style_feats_tensor.to(device)
    return style_feats_tensor

class NSTTransform(transforms.Transform):

    """ A class to apply neural style transfer with the help of adaIN to datasets in the training pipeline.

    Parameters:

    style_feats: Style features extracted from the style images using adaIN Encoder
    vgg: AdaIN Encoder
    decoder: AdaIN Decoder
    alpha = Strength of style transfer [between 0 and 1]
    probability = Probability of applying style transfer [between 0 and 1]
    randomize = randomly selected strength of alpha from a given range
    rand_min = Minimum value of alpha if randomized
    rand_max = Maximum value of alpha if randomized

     """
    def __init__(self, style_feats, vgg, decoder, alpha_min=1.0, alpha_max=1.0, probability=0.5):
        super().__init__()
        self.vgg = vgg
        self.decoder = decoder
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.downsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
        self.to_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
        self.style_features = style_feats
        self.num_styles = len(style_feats)
        self.probability = probability
        self.to_pil_img = transforms.ToPILImage()
        #self.bs = 512

    @torch.no_grad()
    def __call__(self, x):

        if torch.rand(1).item() < self.probability: # here

            x = self.to_tensor(x).to(device)
            x = x.unsqueeze(0) # here
            
            x = self.upsample(x)

            idx = torch.randperm(self.num_styles)[0] #[0:int(self.bs*self.probability)]
            #idy = torch.randperm(self.bs)[0:int(self.bs*self.probability)]

            style_image = self.style_features[idx].unsqueeze(0) # here unsqueeze
            
            x = self.style_transfer(self.vgg, self.decoder, x, style_image) #x[idy] x2

            stl_img = self.downsample(x)
            stl_img = stl_img.squeeze(0).cpu() # here

            stl_img = self.norm_style_tensor(stl_img)
            stl_img = self.to_pil_img(stl_img) # here
            return stl_img
        
        else: # here
            return x # here
    
    def norm_style_tensor(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()

        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        scaled_tensor = normalized_tensor * 255
        scaled_tensor = scaled_tensor.byte() # converts dtype to torch.uint8 between 0 and 255 #here
        return scaled_tensor

    @torch.no_grad()
    def style_transfer(self, vgg, decoder, content, style):

        alpha = np.random.uniform(low=self.alpha_min, high=self.alpha_max)

        content_f = vgg(content)
        style_f = style
        feat = utils.adaptive_instance_normalization(content_f, style_f)

        feat = feat * alpha + content_f * (1 - alpha)
        feat = decoder(feat)
        return feat