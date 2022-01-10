---
title:  "파이토치 DataParallel, DistributedDataParallel, apex, amp 정리"
excerpt: "파이토치 DataParallel, DistributedDataParallel, apex, amp 정리"
categories:
  - Pytorch & Tensorflow & Coding
  
tags:
  - DataParallel
  - pytorch
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-07T00:48:00-05:00
---

ICCV KD rebuttal 때 이미지넷 요청을 받았었는데 시간이 없어서 찾아보고 적용.

먼저 정리하자면, 분산처리에는 2가지 방법이 있고 파이토치에서는 크게 DataParallel과 Distributed.DataParallel이 있다. 하지만 Distributed.DataParallel이는 오류가 뜬다
(어디선가 봤는데 사용하지 않는 파라미터가 있는 경우 오류 뜬다고 함). 따라서 nvidia에서 제공하는 apex를 썼다.

# 모델 저장의 통일성을 위해서 모델을 저장할때 DP나 DDP로 저장할 경우에는 model.module.state_dict로 저장하도록 기억하자.

## 문제
예를 들어 배치가 128이고 resnet101을 훈련시킨다고 하자. 속도의 향상, 또는 VRAM의 부족으로 인해 분산처리를 이용하여 훈련하려고 한다.
서버 컴이 다 돌아가고 있어서 순서 설명은 GPU4개를 이용하는 것으로 설명하였고, 실제 코드는 헷갈림을 방지 (n_world 가 같지 않도록) 고려하기 위해 GPU는 4개중 0,1,3 3개를 이용하는 것으로
구현하였다. 기준 GPU는 0. 
**원래는 cmd창에서**
```bash
CUDA_VISIBLE_DEVICES=0,1,3 python ~~ 
```
**로 실행시키는 것이 더 확실하지만, 본 포스팅에서는 코드에 넣어주었다.**

### DataParallel
1. 배치를 32씩 4개의 GPU에 할당한다. 
2. resnet101 모델 전체를 각 GPU에 복사한다. 
3. 각 GPU에서 32개의 배치를 Forward 한다
4. 나온 output을 기준 GPU3에 모은다.
5. GPU3에서 각 GPU에서 나온 output을 이용하여 각 batch 32개에 대한 loss를 계산
6. loss를 다시 각 GPU에 scatter한다. 
7. 각 GPU가 gradient를 계산하고 
8. GPU3에 모아서 모델 업데이트.
아래 그림과 같다. 
![](/assets/images/2021-07-07-DataParallel/1.JPG)
문제는 이렇게 하면 GPU3이 메모리를 순간적으로 많이 먹게 되어 터질 위험이 있다. 하지만 가장 간편하고 빠른 방법이다. 아래 코드를 참고.

```python
  
import torch
import torchvision
import torchvision.transforms as T
import argparse
import time


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"  # cmd에서 사용하자. 이거는 코드상으로 보여주기용

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args=parser.parse_args()

train_dataset = torchvision.datasets.CIFAR100(root="./data", download=True, transform=T.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)


model = torchvision.models.resnet101(pretrained=False)
model = torch.nn.DataParallel(model).cuda()
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
```

GPU 사용량
![](/assets/images/2021-07-07-DataParallel/2.JPG)
주목할 점은 기준 0번 GPU가 더 많이 VRAM을 잡아 먹는 것과 PID가 모두 같다는 것이다.
<br/>
### Distributed Dataparallel
이 개념이 약간 생소했음. [미결] 이 알고리즘은 각 GPU마다 똑같이 복제된 모델이 어떻게 기준 GPU로 정보를 업데이트 하는지 완전히 이해를 못했다. **하지만 배치사이즈는 동일하게 나누어서 들어가기 때문에 DP와 비슷하다고 볼 수 있다. 
또한 DataLoader에 들어가는 num_workers값도 DP와 달리 프로세스 당 몇개씩 들어가므로 이 연산도 한다. 나는 보통 맨 처음에 argparse로 받는 n_workers는 전체 worker의 갯수로 잡고 코드내에서 나누어준다. 보통은 1개 프로세스당
** 

용어 및 개념 정리

* 여러 GPU를 사용할 때 서버 전체를 쓰면 좋겟지만 그렇지 못할 경우 GPU를 지정해 주어야 한다. 그럴때 쓰는 것이 os.environ["CUDA__VISIBLE_DEVICES"]로 이 
머신(노드)에서 몇 번 GPU를 사용할 것이다 라고 명시.
* local rank : **프로세스를 구분하기 위해 가장 많이 사용하는 변수.** 이 변수는 cmd창에서
```bash
python -m torch.distributed.launch --nproc_per_node=3 main.py
```
를 실행할때 각 프로세스별 순위를 정하는 것이다. 그래서 코드에 명시한 것처럼 local_rank를
출력해보면 0,1,2가 랜덤으로 다르게 출력되는 것을 볼 수 있다. 또한 local rank는 GPU와 무관하게 반드시 0부터 시작되는 것을 기억하자. 즉 아래 코드에서는 GPU를 0,1,3을 
쓰므로 rank가 1,2,0으로 출력이 되었을 때, 첫번째 프로세스에서는 GPU 1을 쓰고 두번째 프로세스에서는 GPU3을 쓰고, 세번째 프로세스에서는 GPU0을 쓴다는 것을 알 수 있다. 
* node : 컴퓨터의 갯수. 아래 코드는 단독 머신이므로 이 개념이 쓰이지는 않았다.
* world_size : 여러 컴퓨터에 같은 GPU갯수가 달려 있다고 가정 할 때, (각 컴퓨터에 달린 GPU 갯수) * (컴퓨터 갯수)라고 할 수 있다. 즉, 모델을 훈련하는 데에 필요한 총 GPU갯수. 없어도 되지만 나는 쓰는 편이다. 
* nproc_per_node : torch.cuda.device_count()와 반드시 일치 해야한다. 이게 가장 애먹었던건데 기본 구글링에서는 그냥 한 서버를 다 쓰는 것을 가정해서 이게 그냥 컴퓨터에 
달린 GPU갯수라고 명시해 놨다. 하지만 몇번 GPU를 쓸것인지를 정확히 명시 하려면 이것을 os.environ["CUDA_VISIBLE_DEVICES"]에 딸린 숫자 갯수와 일치해야 한다.
이건 직접 보고 그때 그때 생각해서 기입하자. 
* master_port : 한 서버에 DDP가 두개 돌아가는 경우가 있다. 이 경우
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 main.py ~~
```
로 돌리면 address in already use에러가 발생한다. 따라서 이 경우는 master_port까지 변경해 주자.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234 main.py ~~
```
각 GPU마다 프로세스가 독립적으로 생성되어 돌아간다고 생각하자.
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torchvision
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser()
#### training ####
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--n_epochs", type=int, default=100)

#### configs ####
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--world_size", type=int)
parser.add_argument("--DDP_backend", type=str, default="nccl")

args = parser.parse_args()
assert args.batch_size > torch.cuda.device_count(), \
    f"batch size ({args.batch_size}) must be greater than # gpus ({torch.cuda.device_count()})"
assert args.n_workers > torch.cuda.device_count(), \
    f"# workers ({args.n_workers}) must be greater than # gpus ({torch.cuda.device_count()})"
args.batch_size = args.batch_size // torch.cuda.device_count()
args.n_workers = args.n_workers // torch.cuda.device_count()

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend=args.DDP_backend, world_size=args.world_size, rank=args.local_rank)

train_dataset = torchvision.datasets.CIFAR100(root="/home/data", download=True, transform=T.ToTensor(), train=True)
valid_dataset = torchvision.datasets.CIFAR100(root="/home/data", download=True, transform=T.ToTensor(), train=False)
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers,
                          sampler=train_sampler)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

model = torchvision.models.resnet18(num_classes=100).cuda(args.local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
criterion_CE = torch.nn.CrossEntropyLoss()
scaler = GradScaler()

for epoch in range(1, args.n_epochs+1):
    for img, label in train_loader:
        img = img.cuda(args.local_rank)
        label = label.cuda(args.local_rank)

        with autocast():
            output = model(img)
            loss = criterion_CE(output, label)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    n_correct = 0
    for img, label in valid_loader:
        img = img.cuda(args.local_rank)
        label = label.cuda(args.local_rank)

        with autocast():
            output = model(img)
            loss = criterion_CE(output, label)
        pred = torch.argmax(output, dim=1)
        n_correct += (pred == label).float().sum()
    if args.local_rank == 0:
        print(f"[Epoch-{epoch}]_[Acc-{n_correct / len(valid_dataset)}]")
```
![](/assets/images/2021-07-07-DataParallel/3.JPG)

결과를 보면 아까와 달리 전부다 같은 양의 메모리를 차지하는 것을 볼 수 있다. 하지만 PID는 모두 다르다.
 즉, 개별적으로 실행되고 있는 코드들이다. 아까보다 훨씬 더 많은 메모리를 차지하는 것을 볼 수 있다. 
<br/>
<br/>
### DistributedDataparallel의 모델 save
프로세스가 여러개이기 때문에 local rank에 따라 다른 모델을 저장 할 수 있다. 하지만 낭비가 너무 심해지기 때문에 보통 마스터 프로세스 (local rank = 0)을 기준으로 모델을 저장하고,
한 epoch가 끝날 때마다 마스터 프로세스에서 저장된 모델을 다른 모델과 동기화 해준다. 개별 프로세스마다 진행 속도가 거의 비슷하지만 다르기 때문에 마스터 프로세스가 모델을 저장하는 것이 
다른 프로세스에서 불러오는 것보다 느리면 이는 다른 프로세스가 epoch를 날리는 것이기 때문에 반드시 마스터 프로세스가 현재 epoch에서 모델을 저장한 다음에 다른 하위 프로세스들이 load를 해야한다.
이를 해주는 것이 dist.barrier()이다. 이 코드는 마스터 프로세스가 도달하기 전까지 다른 프로세스들의 진행을 멈춘다. 즉 순서를 정리 해보면,
1. 마스터 프로세스가 현재 epoch의 모델 저장 (베스트 모델 저장하는 것과 별개로 체크포인트로 저장).
2. 먼저 진행되는 다른 프로세스들이 마스터 프로세스를 넘지 않도록 정지 dist.barrier
3. 마스터 프로세스가 dist.barrier 까지 왔다는 것은 모델이 저장되었다는 것이므로 다른 프로세스들 모델 load
4. 모델을 load 할때 map location으로 gpu:gpu 맞춰주기.
위의 예시 코드의 마지막 부분에 추가된 부분을 주의해서 보자.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
import os.path import join as opj
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torchvision
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser()
#### training ####
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--n_epochs", type=int, default=100)

#### save & load ####
parser.add_argument("--save_root_dir", type=str, default="/home/jeonghokim/dummy")

#### configs ####
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--world_size", type=int)
parser.add_argument("--DDP_backend", type=str, default="nccl")

args = parser.parse_args()
args.save_model_dir = opj(args.save_root_dir, "save_models")
os.makedirs(args.save_model_dir, exist_ok=True)

assert args.batch_size > torch.cuda.device_count(), \
    f"batch size ({args.batch_size}) must be greater than # gpus ({torch.cuda.device_count()})"
assert args.n_workers > torch.cuda.device_count(), \
    f"# workers ({args.n_workers}) must be greater than # gpus ({torch.cuda.device_count()})"
args.batch_size = args.batch_size // torch.cuda.device_count()
args.n_workers = args.n_workers // torch.cuda.device_count()

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend=args.DDP_backend, world_size=args.world_size, rank=args.local_rank)

train_dataset = torchvision.datasets.CIFAR100(root="/home/data", download=True, transform=T.ToTensor(), train=True)
valid_dataset = torchvision.datasets.CIFAR100(root="/home/data", download=True, transform=T.ToTensor(), train=False)
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers,
                          sampler=train_sampler)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

model = torchvision.models.resnet18(num_classes=100).cuda(args.local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
criterion_CE = torch.nn.CrossEntropyLoss()
scaler = GradScaler()
best_acc = 0

for epoch in range(1, args.n_epochs+1):
    for img, label in train_loader:
        img = img.cuda(args.local_rank)
        label = label.cuda(args.local_rank)

        with autocast():
            output = model(img)
            loss = criterion_CE(output, label)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    n_correct = 0
    for img, label in valid_loader:
        img = img.cuda(args.local_rank)
        label = label.cuda(args.local_rank)

        with autocast():
            output = model(img)
            loss = criterion_CE(output, label)
        pred = torch.argmax(output, dim=1)
        n_correct += (pred == label).float().sum()

    if args.local_rank == 0:
        acc = n_correct / len(valid_dataset)
        print(f"[Epoch-{epoch}]_[Acc-{acc}]")
        if best_acc < acc:
            print(f"best accuracy update : {best_acc} -> {acc}")
            best_acc = acc
            save_path = opj(args.save_model_dir, f"[Epoch-{epoch}]_[acc-{best_acc:.4f}]")
            torch.save(model.module.state_dict(), save_path)

    cp_path = opj(args.save_model_dir, "cp.pth")
    if args.local_rank == 0:
        torch.save(model.module.state_dict(), cp_path)
    dist.barrier()  # 메인 프로레스의 모델을 load하기 위해 메인 프로세스가 save하기까지 기다림
    model.module.load_state_dict(torch.load(cp_path, map_location={"cuda:0":f"cuda:{args.local_rank}"}))
    print(f"{args.local_rank} model load")
```
이 코드를 실행시키면 마스터 프로세스 (local rank=0)일 때에만 모델을 저장하고, 각 에폭에서 모델이 저장되기 전까지 dist.barrier()에 의하여 다른 하위 프로세스들은 load_state_dict 이전까지 멈춰있는다. 
즉, print 출력문에서 epoch 다음에 반드시 0 "model save" 이후에 "model load"가 뜬다.
![](/assets/images/2021-07-07-DataParallel/6.JPG)
<br/>
<br/>
### DistributedDataparallel에서 validation
validation 같은 경우에는 loader에서 sampler를 없이한다. 하지만 어차피 마스터 프로세스에서만 validation을 하는 것이 의미가 있으므로 (성능이 안나올 때에는 다 해서 베스트 뽑는 것이 좋다.)
local rank가 0일때만 진행하면 CPU 부하가 적게 걸려 효율적이다. 하지만 마스터 프로세스에서만 validation 함수를 실행시키면 validation을 안하는 하위 프로세스는 barrier에서 기다리고 있고, 마스터 프로세스가
validation을 마친후 모델을 저장하고 barrier로 도착 한뒤에 그 다음부터 코드가 멈춰버린다. 이는 다른 프로세스들과 마스터 프로세스의 연산이 다르게 진행되어 그러는 것으로 보인다. 따라서 
마스터 프로세스만 단독적으로 어떤 추론을 진행하는 경우 model.module(img) 처럼 module을 따로 빼서 진행해 주어야한다. **하지만 GPU에 문제생길수 있으니 그냥 모든 프로세스 다 validation하자. 어차피 CPU차이 거의 없다.** 
<br/>
<br/>
### DistributedDataparallel을 사용 할 경우 반드시 알아야 할 것 (정리)
1. argparser에 local_rank를 추가
2. gpu로 보낼때 반드시 local_rank에 맞춰 보내기
3. DistributedSampler 추가
4. world_size는 별개이지만 GPU총 갯수를 헷갈리지 않게 하기 위해 넣는다.
5. batch_size 갯수 torch.cuda.device_count()로 나누어 주기
6. n_workers 갯수 torch.cuda.device_count()로 나누어 주기
7. 에폭이 끝난 다음에 동기화 하기 위해 checkpoint를 만들고 barrier를 사용하여 save and load 한다. 
8. validation을 모든 프로세스에서 하는 것이 좋으나 피치못할 경우 메인 프로세스만 validation을 하는 경우 다른 프로세스는 barrier함수에 의해 중단되어 있으므로 이럴 경우의 
validation할떄에는 module을 해야 barrier에서 에러가 안난다.
9. logger를 사용할때에는 내 블로그 [링크](https://rlawjdghek.github.io/pytorch%20&%20tensorflow%20&%20coding/utils/)에 있는 것과 같이 local_rank=가 0일때만 저장하는 것이 깔끔하다.
10. 한 서버에 두개의 DDP를 돌릴경우 master_port 매개변수를 활용하자.

### Distributed DataParallel (apex)
예제 코드는 쉬운 코드라 잘 돌아가는 것을 확인할 수 있는데 토치에서 지원하는 Distribute는 모델에서 안 쓰는 매개변수가 있을 경우 오류가 뜬다고 한다. 그래서 추가적으로 알아본 것. 

먼저 https://github.com/NVIDIA/apex 에서 파일들을 다 다운 받는다. 오피셜 코드에서 알려주는 
![](/assets/images/2021-07-07-DataParallel/4.JPG)
이 명령어를 사용해도 되지만 내 경험상 이걸로 설치하면 무조건 apex폴더에 들어가서 import apex를 해야 돌아간다. 
따라서 나는 그냥 압축파일을 다운로드하고 안에 내용물을 내 패키지에 그대로 집어 넣는다. 그다음 3번째 명령어 실행하면 내가 지금 짜는 코드에서 임포트가 된다.

```python
import torch
import torchvision
import torchvision.transforms as T
from apex.parallel import DistributedDataParallel as DDP
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
args.gpu = args.local_rank
torch.cuda.set_device(args.gpu)
torch.distributed.init_process_group(backend="nccl")

train_dataset = torchvision.datasets.CIFAR100(root="./data", transform=T.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=8)


model = torchvision.models.resnet101(pretrained=False).cuda(args.gpu)
model = DDP(model, delay_allreduce=True)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(100):
    start = time.time()
    for n, (img, label) in enumerate(train_loader):
        img = img.cuda(args.gpu)
        label = label.cuda(args.gpu)


        output = model(img)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n % 20 == 0:
            print(f"[{n}/{len(train_loader)}]")
            print(f"time : {time.time() - start}")
    print("epoch")
```
방법은 토치의 DistributeDataParallel과 같다. 무조건 local_rank를 parser에 집어넣고 프로세스 그룹을 initialize한 다음 apex에서 DDP로 묶어준다. 
실행 명령어도 토치와 같다. 
![](/assets/images/2021-07-07-DataParallel/5.JPG)
<strike>하지만 문제는 실제 KD 이미지넷 코드를 돌릴 경우 num_workers=0아니면 only to child process 에러가 뜬다. 아직 방법을 찾지는 못함.</strike> 
(+2021-10-08에 추가한 아래 코드 보고 나중에 해보자. 아래코드는 num_worker가 있어도 된다.)
<br/>
<br/>
### apex + amp
amp는 mixed_precision이라는 것을 사용하여 쓸데없는 계산 량을 줄이고 성능 차이는 거의 안나게 하는 패키지.

아래 코드는 Distribured DataParallel + amp
```python
import torch
import torchvision
import torchvision.transforms as T
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
torch.distributed.init_process_group(backend="nccl")
memory_format = torch.contiguous_format

train_dataset = torchvision.datasets.CIFAR100(root="./data", transform=T.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)


model = torchvision.models.resnet101(pretrained=False).cuda(args.gpu)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()

model, optimzier = amp.initialize(model, optimizer, opt_level="O3")
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)



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
```

아래 코드는 apex + amp
```python
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
torch.distributed.init_process_group(backend="nccl")

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
```
위의 코드에서 opt_level만 바꾸면 된다. 알아본 바로는 O1을 가장 많이 쓴다고 하는데 정확히 무슨 차이가 있는지는 모르겟다. 또한 내 실험 결과에서는 O1이 기존보다 더 느리게 떳다. 
<br/>
<br/>
### 실행 속도 비교
**위의 코드를 돌렸을때 20iter마다 찍힌 실행 시간.**
Dataparallel : 5.6 -> 9.2 -> 12.8 -> 16.3 -> 20.4

nn.DistDataparallel: 0.4 -> 2.5 -> 4.6 -> 6.7 -> 8.7

apex: 0.9 -> 3.3 -> 5.7 -> 8.0 -> 10.3

apexO1: 0.4 -> 3.9 -> 5.7 -> 8.0 -> 10.3

apexO2: 0.4 -> 2.8 -> 5.3 -> 7.8 -> 10.4

**apexO3: 0.3 -> 2.1 -> 3.8 -> 5.5 -> 7.2  apex에서 만든것끼리 쓴것이 제일 빠른듯.**

nn.DistDataparallel O1: 0.4 -> 3.3 -> 6.5 -> 9.6 -> 12.8

nn.DistDataparallel O2: 0.4 -> 2.8 -> 5.5 -> 8.2 -> 10.5

nn.DistDataparallel O3: 0.3 -> 2.2 -> 4.0 -> 6.0 -> 8.0


+ 2021-07-10
<br/>
<br/>
### torch amp
위의 apex의 amp대안으로 파이토치에서 제공하는 amp를 사용해보자. 나중에 시간있으면 이것도 위에처럼 ablation 해보기. 
방법은 아래 코드처럼 scaler만들고, with autocast로 forward부분만 묶어주면 된다.

```python
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
```


+ 2021-10-08 알아낸것.
Representation baseline 환경 구축 하다 알아낸것. arcface에서 MSCeleb을 병렬처리로 훈련하기 위해 커스텀 dataloader를 만들었다. 전체 코드는 아래와 같고 나중에 병렬처리 에러가 뜨면 아래 코드 참고해서 작성하자.

```python
from os.path import join as opj
import numpy as np
import numbers
import threading
import queue as Queue

import mxnet as mx
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

class MXFaceDataset(Dataset):
    def __init__(self, local_rank, data_root_dir="/home/CVPR_data/faces_emore"):
        super().__init__()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.data_root_dir = data_root_dir
        self.local_rank = local_rank
        path_imgrec = opj(self.data_root_dir, "train.rec")
        path_imgidx = opj(self.data_root_dir, "train.idx")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __len__(self):
        return len(self.imgidx)

    def __getitem__(self, idx):
        idx = self.imgidx[idx]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        img = mx.image.imdecode(img).asnumpy()
        img = self.transform(img)
        return img, label


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=8)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")

    train_dataset = MXFaceDataset(local_rank=args.local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=args.local_rank,
        dataset=train_dataset,
        batch_size=4,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    for img, label in train_loader:
        print(img.shape)
        print(label)

```
torch.cuda.set_device가 필수로 들어가야한다. 이거 없으면 nvidia-smi 명령어에서 볼 수 있듯, 0번 GPU에 8개가 추가로 들어간다. 아래 for 문에서 device가 0이 압도적으로 많다. 
