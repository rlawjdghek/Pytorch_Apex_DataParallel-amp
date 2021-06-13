import torch
import torchvision
import torchvision.transforms as T
import argparse
import time
from torch.utils.data.distributed import DistributedSampler



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--world_size", default=3, type=int)
args=parser.parse_args()
print("local rank : {}".format(args.local_rank))
torch.distributed.init_process_group(backend="nccl")

train_dataset = torchvision.datasets.CIFAR100(root="./data", download=True, transform=T.ToTensor())
sampler = DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=sampler)

model = torchvision.models.resnet101(pretrained=False).cuda(args.local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(100):
    start = time.time()
    for n, (img, label) in enumerate(train_loader):
        img = img.cuda(args.local_rank)
        label = label.cuda(args.local_rank)


        output = model(img)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n % 20 == 0:
            print(f"[{n}/{len(train_loader)}]")
            print(f"time : {time.time() - start}")
    print("epoch")


