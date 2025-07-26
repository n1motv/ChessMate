# train_resnet.py  ───────────────────────────────────────────────────────────
import os
import torch, torchvision, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

root   = "dataset"
labels = sorted(os.listdir(root))
nb_cls = len(labels)

tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

ds  = datasets.ImageFolder(root, tfm)
dl  = DataLoader(ds, batch_size=32, shuffle=True)
net = torchvision.models.resnet18(weights="DEFAULT")
net.fc = nn.Linear(net.fc.in_features, nb_cls)
opt = torch.optim.AdamW(net.parameters(), lr=0.0001)
loss= nn.CrossEntropyLoss()

for epoch in range(8):
    for x,y in dl:
        opt.zero_grad(); loss(net(x),y).backward(); opt.step()
    print("epoch",epoch,"ok")
torch.save(net.state_dict(), "resnet18_chess.pt")
