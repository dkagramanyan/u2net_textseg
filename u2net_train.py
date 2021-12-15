import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

from sklearn.model_selection import train_test_split

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from u2net import U2NET

import time
import datetime

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------- 1. define loss function --------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
bce_loss = nn.BCELoss(size_average=True, reduction='mean').to(device)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v).to(device)
    loss1 = bce_loss(d1, labels_v).to(device)
    loss2 = bce_loss(d2, labels_v).to(device)
    loss3 = bce_loss(d3, labels_v).to(device)
    loss4 = bce_loss(d4, labels_v).to(device)
    loss5 = bce_loss(d5, labels_v).to(device)
    loss6 = bce_loss(d6, labels_v).to(device)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = f'u2net_{datetime.datetime.now().date()}'

images_path = 'data/image/'
masks_path = 'data/semantic_label/'

model_dir = 'saved_models/'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

tra_img_name_list = glob.glob(images_path + '*')
tra_lbl_name_list = glob.glob(masks_path + '*')

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(500),
        #     RandomCrop(288),

        ToTensorLab(flag=0)]
    )
)

# ------- 3. define model --------
# define the net
net = U2NET(3, 1)
net.to(device)

checkpoint_name = 'u2net_2021-12-15epoch_302_train_0.000000_test_0.002015.pth'
folder_name = 'saved_models/'
if checkpoint_name:
    net.load_state_dict(torch.load(folder_name + checkpoint_name, map_location=torch.device('cpu')))

# ------- 4. define optimizer and train --------


save_frq = 1  # save the model every 2000 iterations

epoch_num = 1000
batch_size = 10
test_batch_size = 10
train_num = len(tra_img_name_list)
validation_split = 0.25

train_dataset_size = int(train_num * (1 - validation_split))
test_dataset_size = train_num - train_dataset_size
train_dataset, test_dataset = torch.utils.data.random_split(salobj_dataset, (train_dataset_size, test_dataset_size))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

for epoch in range(0, epoch_num):
    net.train()
    train_loss = 0
    train_step = 0
    test_loss = 0

    for i, train_data in enumerate(train_dataloader):
        train_step += 1
        # train_step_save = + 1
        start_time = time.time()

        train_inputs = train_data['image'].to(device)
        train_labels = train_data['label'].to(device)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(train_inputs)

        _, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, train_labels)

        loss.backward()

        optimizer.step()

        # # print statistics
        train_loss += loss.data.item()

        end_time = time.time()
        eta = (end_time - start_time) * (train_dataset_size - (i + 1) * batch_size) / batch_size
        print(
            f"epoch: {epoch + 1}/{epoch_num} eta:{int(eta)} s batch: {(i + 1) * batch_size}/{train_dataset_size},"
            f" loss: {train_loss / train_step} ")
        # if train_step % save_frq == 0:
    print('testing')
    with torch.no_grad():
        for j, test_data in enumerate(test_dataloader):
            test_inputs = test_data['image'].to(device)
            test_labels = test_data['label'].to(device)

            d0, d1, d2, d3, d4, d5, d6 = net(test_inputs)
            _, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, test_labels)

            # # print statistics
            test_loss += loss.data.item()

    print(
        f"epoch: {epoch + 1}/{epoch_num} loss: {train_loss / train_step} test loss: {test_loss / test_dataset_size} ")

    torch.save(net.state_dict(),
               model_dir + model_name + f"_epoch_{epoch}_train_{train_loss / train_step}_test_{test_loss / test_dataset_size}.pth")
    # train_loss = 0.0
    net.train()  # resume train
#   train_step_save = 0
