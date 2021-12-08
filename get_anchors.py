import torch
import numpy as np

# Model
model = torch.load('runs/train/indoor/exp/weights/best.pt')  # custom model

# Anchors
m = model['model'].model[-1]
anchors = m.anchors.clone().view_as(m.anchors) * m.stride.to(m.anchors.device).view(-1, 1, 1)
print(np.array(anchors.squeeze().cpu()).astype(int).reshape(m.nl, -1))

