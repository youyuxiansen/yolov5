import torch
import numpy as np

# Model
model = torch.load('runs/train/indoor/exp4/weights/best.pt')  # custom model

# Anchors
m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
anchors_for_save = np.array(m.anchors.clone().cpu() * m.stride.to(
	m.anchors.device).view(-1, 1, 1).cpu()).astype(int).reshape(m.nl, -1)  # nl:detect layer
print(anchors_for_save)
