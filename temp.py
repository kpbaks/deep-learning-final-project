# %%
import torch

from dataset import DrumsDataset

latent = torch.empty(260)
drum_type = 'chat'
drum_type_one_hot = DrumsDataset.onehot_encode_label(drum_type)
latent[256:] = drum_type_one_hot
print(latent)
# latent[-4] = 1.0
# latent[:256] = torch.rand(256)
