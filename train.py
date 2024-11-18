from torchmetrics import Accuracy
from datasets.Dataset import emg_dataset
from models.TrainCustomModel import TModelHJY
from models.TrainCustomModel import TModelHHJ84
from models.TrainCustomModel import TModel8114
from models.RnnNet2 import Model
import torch.nn as nn
import torch.distributed
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
from utils.AddFunc import check_folder


local_rank = 0
world_size = 0
depth = 8
batch = 64
time_slot = 64
channel = 4
shuffle = True
num_worker = 1
pin = True
prefetch = 2
persistent = True
opt = f'Adam'
learning_rate = 0.0001
amp = False
momentums = 0.9
weight_dc = 0.00005
num_classes = 4

start_epoch = 0
end_epochs = 2000
schdual = 'None'
loss_list = []
accdet_list = []
acccls_list = []
result = './weight_Mandro_HJ_AVGF/'
check_folder(result)


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda', local_rank)
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device(f'mps', local_rank)
    else:
        device = torch.device(f'cpu')
    print(device)
    accuracy = Accuracy(task="multiclass", num_classes=4).to(device)
    # 2. dataloader
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train = emg_dataset(data_dir='./4C_AVGF', window_size=time_slot, channel=channel, step=1)  # 49080
    test = emg_dataset(data_dir='./4C_AVGF', window_size=time_slot, channel=channel, step=1)  # 6135
    trainset = DataLoader(dataset=train, batch_size=batch, shuffle=shuffle, num_workers=num_worker,
                          pin_memory=pin, prefetch_factor=prefetch, persistent_workers=persistent)
    nb = len(list(enumerate(trainset)))
    # hl = list(enumerate(trainset))
    # print(hl[0])
    # breakpoint()
    model = Model(time_slot, depth, num_classes, channel).to(device)

    if opt == f'SGD':
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentums, weight_decay=weight_dc)
    elif opt == f'Adam':
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif opt == f'RMSProp':
        optimizer = RMSprop(model.parameters(), lr=learning_rate)
    else:
        print(f"Opt error !!")
        breakpoint()

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    summary(model,size=(batch,time_slot,4),)
    for epoch in tqdm(range(start_epoch, end_epochs), desc=f'Epoch', disable=False):
        model.train()
        print(f'{"Gpu_mem":10s} {"total":>10s} {"Acc":>10s}')
        pbar = tqdm(enumerate(trainset), total=nb, desc=f'batch', leave=True, disable=False)
        for batch_idx, (features) in pbar:
            targets, data = features["label"].to(device), features["data"].to(device)
            # print(targets)
            # print(data)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                output = model(data)
                # output = output.reshape(-1, output.size(-1))
                # targets = targets.view(-1).long()
                losses = criterion(output, targets)
                # print(torch.argmax(output, dim=2), torch.argmax(targets, dim=2))
                accuracy.update(torch.argmax(output, dim=2), torch.argmax(targets, dim=2))
                # accuracy.update(torch.argmax(output.view(-1, num_classes), dim=1), targets)
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = f'{mem:10s} {losses.mean():10.6g} {accuracy.compute()*100:10.6g}'
            pbar.set_description(s)
            if epoch % 10 == 0:
                torch.save(model.state_dict(), result + str(epoch).zfill(3) + '.pt')
    torch.save(model.state_dict(), result + f'final.pt')
