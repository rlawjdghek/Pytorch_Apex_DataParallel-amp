import torch
import torchvision
import torchvision.transforms as T
import argparse
import time


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args=parser.parse_args()

train_dataset = torchvision.datasets.CIFAR100(root="./data", download=True, transform=T.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)


model = torchvision.models.resnet101(pretrained=False).cuda()
model = torch.nn.DataParallel(model)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(100):
    start = time.time()
    for n, (img, label) in enumerate(train_loader):
        img = img.cuda()
        label = label.cuda()


        output = model(img)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n % 20 == 0:
            print(f"[{n}/{len(train_loader)}]")
            print(f"time : {time.time() - start}")
    print("epoch")


