import torch
import torchvision
import torchvision.transforms as T
import argparse
import time
from torch.cuda.amp import autocast, GradScaler


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 7"

train_dataset = torchvision.datasets.CIFAR100(root="../data", transform=T.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)

model = torchvision.models.resnet101(pretrained=False).cuda()
model = torch.nn.DataParallel(model)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()
scaler = GradScaler()

for epoch in range(100):
    start = time.time()
    for n, (img, label) in enumerate(train_loader):
        img = img.cuda()
        label = label.cuda()
        label = torch.tensor(torch.arange(0,128)).cuda()

        optimizer.zero_grad()
        with autocast():
            output = model(img)
            loss = criterion(output, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if n % 100 == 0:
            print(f"[{n}/{len(train_loader)}]")
            print(f"time : {time.time() - start}")
    print("epoch")


