#!/usr/bin/env python
import torch
import torchvision
import torchaudio
print(torch.__version__)
print(torchvision.__version__)
print(torchaudio.__version__)
print(torch.cuda.is_available())