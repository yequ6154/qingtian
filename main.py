import torch
import random
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.nn.functional as func
from torch.autograd import Variable
# import torch.autograd as grad
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import math
from birealcapsnet import *
import scipy.ndimage as ndi
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PyTorch bi-real capsulenet' )
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--lamda', type=float, default=0.5, help='learning rate')
parser.add_argument('--m_plus', type=float, default=0.9)
parser.add_argument('--m_minus', type=float, default=0.1)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()

mse_loss = nn.MSELoss(reduction='none')

def margin_loss(x, labels):
        v_c = torch.norm(x, dim=2, keepdim=True)
        tmp1 = func.relu(m_plus - v_c).view(x.shape[0], -1) ** 2
        tmp2 = func.relu(v_c - m_minus).view(x.shape[0], -1) ** 2
        loss_ = labels*tmp1 + lamda*(1-labels)*tmp2
        loss_ = loss_.sum(dim=1)
        return loss_
    
def reconst_loss(recnstrcted, data):
        loss = mse_loss(recnstrcted.view(recnstrcted.shape[0], -1), data.view(recnstrcted.shape[0], -1))
        return 0.4 * loss.sum(dim=1)
    
def loss(x, recnstrcted, data, labels):
        loss_ = margin_loss(x, labels, lamda, m_plus, m_minus) + reconst_loss(recnstrcted, data)
        return loss_.mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


workers = 0   # dataloder线程数

test_flag = True  #测试标志，True时加载保存好的模型进行测试 
log_dir = './fmnist_model.pth'  # 模型保存路径

def transform_matrix_offset_center(matrix, x, y):
    
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform, fill_mode='nearest', fill_value=0.):
    
    x = x.astype('float32')
    transform = transform_matrix_offset_center(transform, x.shape[1], x.shape[2])
    final_affine_matrix = transform[:2, :2]
    final_offset = transform[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
            final_offset, order=0, mode=fill_mode, cval=fill_value) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    return x

class Affine(object):

    def __init__(self, 
                 rotation_range=None, 
                 translation_range=None,
                 shear_range=None, 
                 zoom_range=None, 
                 fill_mode='constant',
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0.):
        
        self.transforms = []
        if translation_range:
            translation_tform = Translation(translation_range, lazy=True)
            self.transforms.append(translation_tform)
        
        if rotation_range:
            rotation_tform = Rotation(rotation_range, lazy=True)
            self.transforms.append(rotation_tform)

        if shear_range:
            shear_tform = Shear(shear_range, lazy=True)
            self.transforms.append(shear_tform) 

        if zoom_range:
            zoom_tform = Translation(zoom_range, lazy=True)
            self.transforms.append(zoom_tform)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value

    def __call__(self, x, y=None):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](x)
        for tform in self.transforms[1:]:
            tform_matrix = np.dot(tform_matrix, tform(x)) 

        x = torch.from_numpy(apply_transform(x.numpy(), tform_matrix,
            fill_mode=self.fill_mode, fill_value=self.fill_value))

        if y:
            y = torch.from_numpy(apply_transform(y.numpy(), tform_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
            return x, y
        else:
            return x

class AffineCompose(object):

    def __init__(self, 
                 transforms, 
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0.):
        
        self.transforms = transforms
        # set transforms to lazy so they only return the tform matrix
        for t in self.transforms:
            t.lazy = True
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value

    def __call__(self, x, y=None):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](x)
        for tform in self.transforms[1:]:
            tform_matrix = np.dot(tform_matrix, tform(x)) 

        x = torch.from_numpy(apply_transform(x.numpy(), tform_matrix,
            fill_mode=self.fill_mode, fill_value=self.fill_value))

        if y:
            y = torch.from_numpy(apply_transform(y.numpy(), tform_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
            return x, y
        else:
            return x


class Rotation(object):

    def __init__(self, 
                 rotation_range, 
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
       
        self.rotation_range = rotation_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        degree = random.uniform(-self.rotation_range, self.rotation_range)
        theta = math.pi / 180 * degree
        rotation_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                                    [math.sin(theta), math.cos(theta), 0],
                                    [0, 0, 1]])
        if self.lazy:
            return rotation_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), rotation_matrix,
                fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), rotation_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed


class Translation(object):

    def __init__(self, 
                 translation_range, 
                 fill_mode='constant',
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
      
        if isinstance(translation_range, float):
            translation_range = (translation_range, translation_range)
        self.height_range = translation_range[0]
        self.width_range = translation_range[1]
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        # height shift
        if self.height_range > 0:
            tx = random.uniform(-self.height_range, self.height_range) * x.size(1)
        else:
            tx = 0
        # width shift
        if self.width_range > 0:
            ty = random.uniform(-self.width_range, self.width_range) * x.size(2)
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.lazy:
            return translation_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), 
                translation_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), translation_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed


class Shear(object):

    def __init__(self, 
                 shear_range, 
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        
        self.shear_range = shear_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        shear = random.uniform(-self.shear_range, self.shear_range)
        shear_matrix = np.array([[1, -math.sin(shear), 0],
                                 [0, math.cos(shear), 0],
                                 [0, 0, 1]])
        if self.lazy:
            return shear_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), 
                shear_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), shear_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed
      

class Zoom(object):

    def __init__(self, 
                 zoom_range, 
                 fill_mode='constant', 
                 fill_value=0, 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        
        if not isinstance(zoom_range, list) and not isinstance(zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zy = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if self.lazy:
            return zoom_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), 
                zoom_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), zoom_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed



class trans(object):
    def __init__(self, 
                 rotation_range=None, 
                 translation_range=None,
                 shear_range=None, 
                 zoom_range=None, 
                 fill_mode='constant',
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0.
                ):
       self.affine = Affine(rotation_range, translation_range, shear_range, zoom_range) 
    
    def __call__(self, data):
        data = transforms.ToTensor()(data)
        return self.affine(data)

train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='./data1/',train=True,download=True,transform=trans(rotation_range=0.1, translation_range=0.1, zoom_range=(0.1, 0.2))),batch_size = args.batch_size,shuffle=True, num_workers=workers)
test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='./data1/',train=False,download=True,transform=transforms.ToTensor()),batch_size = args.batch_size,shuffle=True, num_workers=workers)


model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), 0.001)


def accuracy(indices, labels):
    correct = 0.0
    for i in range(indices.shape[0]):
        if float(indices[i]) == labels[i]:
            correct += 1
    return correct


def test(model, test_loader, loss):
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():    
        for i, (data, label) in enumerate(test_loader):
            data, labels = data.to(device), one_hot(label.to(device))
            optimizer.zero_grad()
            outputs, masked_output, recnstrcted, indices = model(data)
            loss_test = model.loss(outputs, recnstrcted, data, labels, args.lamda, args.m_plus, args.m_minus)
            test_loss += loss_test.data
            indices_cpu, labels_cpu = indices.cpu(), label.cpu()
            correct += accuracy(indices_cpu, labels_cpu)
        test_loss /= (i + 1)
        print("\nAverage Test Loss: ", test_loss, "; Test Accuracy: " , correct/len(test_loader.dataset) * 100.,'\n')


def train(train_loader, model, epoch):    
    #lambda1 = lambda epoch: 0.5**(epoch // 10)
    #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    train_loss = 0
    #for epoch in range(epochs):
    for i, (data, label_) in enumerate(train_loader):
        data, label = data.to(device), label_.to(device)
        labels = one_hot(label)
        optimizer.zero_grad()
        outputs, masked, recnstrcted, indices = model(data, labels)
        loss_val = model.loss(outputs, recnstrcted, data, labels, args.lamda, args.m_plus, args.m_minus)
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val
        loss_mean = train_loss / (i+1)
        print('Train Epoch: {}\t Train nums: {}\t Loss: {:.6f}'.format(epoch, i + 1, loss_mean.item()))
        
def main():
    # 如果test_flag=True,则加载已保存的模型
    #if test_flag:
     #   # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
    #    checkpoint = torch.load(log_dir)
    #    model.load_state_dict(checkpoint['model'])
    #    optimizer.load_state_dict(checkpoint['optimizer'])
    #    start_epoch = checkpoint['epoch']
    #    test(model, test_loader)
    #    return
        
    # 如果有保存的模型，则加载模型，并在其基础上继续训练    
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')
        
        
    for epoch in range(start_epoch+1, args.epochs):
        train(train_loader, model, epoch)
        test(model, test_loader, loss)
        # 保存模型
        state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, log_dir)    
        
    #    if batch_idx%10 == 0:
     #     outputs, masked, recnstrcted, indices = model(data)
      #    loss_val = model.loss(outputs, recnstrcted, data, labels, lamda, m_plus, m_minus)
      #    print("epoch: ", epoch, "batch_idx: ", batch_idx, "loss: ", loss_val, "accuracy: ", accuracy(indices, label_.cpu())/indices.shape[0])
      #test(model, test_loader, loss, batch_size, lamda, m_plus, m_minus)
      #lr_scheduler.step()

if __name__ == '__main__':
    main()



