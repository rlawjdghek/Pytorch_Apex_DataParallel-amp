import torch
import torchvision
import torchvision.transforms as T
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import argparse
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args=parser.parse_args()

args.gpu = args.local_rank
torch.cuda.set_device(args.gpu)
torch.distributed.init_process_group(backend="nccl", init_method="env://")

train_dataset = torchvision.datasets.CIFAR100(root="./data", transform=T.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)

model = torchvision.models.resnet101(pretrained=False).cuda(args.gpu)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()

model, optimzier = amp.initialize(model, optimizer, opt_level="O1")
model = DDP(model, delay_allreduce=True)

for epoch in range(100):
    start = time.time()
    for n, (img, label) in enumerate(train_loader):
        img = img.cuda(args.gpu)
        label = label.cuda(args.gpu)


        output = model(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if n % 20 == 0:
            print(f"[{n}/{len(train_loader)}]")
            print(f"time : {time.time() - start}")
    print("epoch")


