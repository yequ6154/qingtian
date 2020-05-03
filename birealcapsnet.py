import torch
import random
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

eps = 1e-8
cf = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def squash(s, dim=-1):
    norm = torch.norm(s, dim=dim, keepdim=True)
    return (norm /(1 + norm**2 + eps)) * s
    
def softmax_3d(x, dim):
  return (torch.exp(x) / torch.sum(torch.sum(torch.sum(torch.exp(x), dim=dim[0], keepdim=True), dim=dim[1], keepdim=True), dim=dim[2], keepdim=True))

def one_hot(tensor, num_classes=10):
    return torch.eye(num_classes).to(device).index_select(dim=0, index=tensor.to(device)) # One-hot encode
#     return torch.eye(num_classes).index_select(dim=0, index=tensor).numpy() # One-hot encode
   

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
    
    # 输入卷积的W需要乘一个 scaling_factor a。
    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True) # shape = (out_chn,1,1,1)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)  # W = a*W'
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)  #the clip function 
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y
        
class HardBinaryConv3D(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv3D, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (1, out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
    
    # 输入卷积的W需要乘一个 scaling_factor a。
    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(torch.mean(abs(real_weights),dim=4,keepdim=True),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True) # shape = (out_chn,1,1,1)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)  # W = a*W'
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)  #the clip function 
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv3d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

class ConvertToCaps(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
      # channels first
      return torch.unsqueeze(inputs, 2)

class FlattenCaps(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        # inputs.shape = (batch, channels, dimensions, height, width)
        batch, channels, dimensions, height, width = inputs.shape
        inputs = inputs.permute(0, 3, 4, 1, 2).contiguous()
        output_shape = (batch, channels * height * width, dimensions)
        return inputs.view(*output_shape)

class CapsToScalars(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        # inputs.shape = (batch, num_capsules, dimensions)
        return torch.norm(inputs, dim=2)

      

class Conv2DCaps(nn.Module):
    def __init__(self, h, w, ch_i, n_i, ch_j, n_j, kernel_size=3, stride=1, r_num=1):
        super().__init__()
        self.ch_i = ch_i
        self.n_i = n_i
        self.ch_j = ch_j
        self.n_j = n_j
        self.kernel_size = kernel_size
        self.stride = stride
        self.r_num = r_num        
        in_channels = self.ch_i * self.n_i
        out_channels = self.ch_j * self.n_j
        #self.pad = 1
        
#         self.w = nn.Parameter(torch.randn(ch_j, n_j, ch_i, n_i, kernel_size, kernel_size) * 0.01)
        
#         self.w_reshaped = self.w.view(ch_j*n_j, ch_i*n_i, kernel_size, kernel_size)
        
        #self.conv1 = nn.Conv2d(in_channels=in_channels,
        #                       out_channels=out_channels,
        #                       kernel_size=self.kernel_size,
         #                      stride=self.stride,
          #                     padding=self.pad).cuda()
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(in_channels, out_channels, stride=stride)
        
        
    def forward(self, inputs):
        # check if happened properly
        # inputs.shape: (batch, channels, dimension, hight, width)
        residual = inputs
        self.batch, self.ch_i, self.n_i, self.h_i, self.w_i = inputs.shape
        in_size = self.h_i
        x = inputs.view(self.batch, self.ch_i * self.n_i, self.h_i, self.w_i)
        
        #x = self.conv1(x)
        x = self.binary_activation(x)
        x = self.binary_conv(x)   
        width = x.shape[2]
        x = x.view(inputs.shape[0], self.ch_j, self.n_j, width, width)
        x = squash(x,dim=2)
        x += residual
        return x # squash(x).shape: (batch, channels, dimension, ht, wdth)

class Conv2DCapsdownsample(nn.Module):
    def __init__(self, h, w, ch_i, n_i, ch_j, n_j, kernel_size=3, stride=1, r_num=1):
        super().__init__()
        self.ch_i = ch_i
        self.n_i = n_i
        self.ch_j = ch_j
        self.n_j = n_j
        self.kernel_size = kernel_size
        self.stride = stride
        self.r_num = r_num        
        in_channels = self.ch_i * self.n_i
        out_channels = self.ch_j * self.n_j
        #self.pad = 1
        
#         self.w = nn.Parameter(torch.randn(ch_j, n_j, ch_i, n_i, kernel_size, kernel_size) * 0.01)
        
#         self.w_reshaped = self.w.view(ch_j*n_j, ch_i*n_i, kernel_size, kernel_size)
        
        #self.conv1 = nn.Conv2d(in_channels=in_channels,
        #                       out_channels=out_channels,
        #                       kernel_size=self.kernel_size,
         #                      stride=self.stride,
          #                     padding=self.pad).cuda()
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(in_channels, out_channels, stride=stride)
        
        
    def forward(self, inputs):
        # check if happened properly
        # inputs.shape: (batch, channels, dimension, hight, width)    
        self.batch, self.ch_i, self.n_i, self.h_i, self.w_i = inputs.shape
        in_size = self.h_i
        x = inputs.view(self.batch, self.ch_i * self.n_i, self.h_i, self.w_i)
        
        #x = self.conv1(x)
        x = self.binary_activation(x)
        x = self.binary_conv(x)   
        width = x.shape[2]
        x = x.view(inputs.shape[0], self.ch_j, self.n_j, width, width)
        x = squash(x,dim=2)
        return x # squash(x).shape: (batch, channels, dimension, ht, wdth)

class ConvCapsLayer3D(nn.Module):
  def __init__(self, ch_i, n_i, ch_j=32, n_j=4, kernel_size=3, r_num=3):
    
    super().__init__()
    self.ch_i = ch_i
    self.n_i = n_i
    self.ch_j = ch_j
    self.n_j = n_j
    self.kernel_size = kernel_size
    self.r_num = r_num
    in_channels = 1
    out_channels = self.ch_j * self.n_j
    stride = (n_i, 1, 1)
    pad = (0, 1, 1)
    
#     self.w = nn.Parameter(torch.randn(ch_j*n_j, 1, n_i, 3, 3))
    
    
    #self.conv1 = nn.Conv3d(in_channels=in_channels,
     #                      out_channels=out_channels,
      #                     kernel_size=self.kernel_size,
       #                    stride=stride,
        #                   padding=pad).to(device)
    
    self.binary_activation = BinaryActivation()
    self.binary_conv = HardBinaryConv3D(in_channels, out_channels, stride=stride)
    
  def forward(self, inputs):
    # x.shape = (batch, channels, dimension, height, width)
    self.batch, self.ch_i, self.n_i, self.h_i, self.w_i = inputs.shape
    in_size = self.h_i
    out_size = self.h_i

    x = inputs.view(self.batch, self.ch_i * self.n_i, self.h_i, self.w_i)
    x = x.unsqueeze(1)
    #x = self.conv1(x)
    x = self.binary_activation(x)
    x = self.binary_conv(x) 
    
    self.width = x.shape[-1]
    
    x = x.permute(0,2,1,3,4)
    x = x.view(self.batch, self.ch_i, self.ch_j, self.n_j, self.width, self.width)
    x = x.permute(0, 4, 5, 3, 2, 1).contiguous()
    self.B = x.new(x.shape[0], self.width, self.width, 1, self.ch_j, self.ch_i).zero_()
    x = self.update_routing(x, self.r_num)
    return x
  
  def update_routing(self, x, itr=3):
    # x.shape = (batch, width, width, n_j, ch_j, ch_i)    
    for i in range(itr):
      # softmax of self.B along (1,2,4)
      tmp = self.B.permute(0,5,3,1,2,4).contiguous().reshape(x.shape[0],self.ch_i,1,self.width*self.width*self.ch_j)
      #k = softmax_3d(self.B, (1,2,4))   # (batch, width, width, 1, ch_j, ch_i)
      #k = func.softmax(self.B, dim=4)
      k = func.softmax(tmp,dim=-1)
      k = k.reshape(x.shape[0],self.ch_i,1,self.width,self.width,self.ch_j).permute(0,3,4,2,5,1).contiguous()
      S_tmp = k * x
      S = torch.sum(S_tmp, dim=-1, keepdim=True)
      S_hat = squash(S)
      
      if i < (itr-1):
        agrements = (S_hat * x).sum(dim=3, keepdim=True)   # sum over n_j dimension
        self.B = self.B + agrements
      
    S_hat = S_hat.squeeze(-1)
    #batch, h_j, w_j, n_j, ch_j  = S_hat.shape
    return S_hat.permute(0, 4, 3, 1, 2).contiguous()



class Mask_CID(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, x, target=None):
        # x.shape = (batch, classes, dim)
        # one-hot required
        if target is None:
            classes = torch.norm(x, dim=2)
            max_len_indices = classes.max(dim=1)[1].squeeze()
        else:
            max_len_indices = target.max(dim=1)[1]
        
#         print("max_len_indices: ", max_len_indices)
        increasing = torch.arange(start=0, end=x.shape[0]).to(device)
        m = torch.stack([increasing, max_len_indices], dim=1)
        
        masked = torch.zeros((x.shape[0], 1) + x.shape[2:])
        for i in increasing:
            masked[i] = x[m[i][0], m[i][1], :].unsqueeze(0)

        return masked.squeeze(-1), max_len_indices  # dim: (batch, 1, capsule_dim)


# In[11]:


print("class CapsuleLayer")

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules=10, num_routes=640, in_channels=8, out_channels=16, routing_iters=3): 
        # in_channels: input_dim;   out_channels: output_dim.
        super().__init__()
        
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.routing_iters = routing_iters
        
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels) * 0.01)
        self.bias = nn.Parameter(torch.rand(1, 1, num_capsules, out_channels) * 0.01)
    
    def forward(self, x):
        # x: [batch_size, 32, 16] -> [batch_size, 32, 1, 16]
        #                          -> [batch_size, 32, 1, 16, 1]
#         print("CapsuleLayer_x.shape: ", x.shape)
        x = x.unsqueeze(2).unsqueeze(dim=4)
        
        u_hat = torch.matmul(self.W, x).squeeze()  # u_hat -> [batch_size, 32, 10, 32]
        
        #   b_ij = torch.zeros((batch_size, self.num_routes, self.num_capsules, 1))
        b_ij = x.new(x.shape[0], self.num_routes, self.num_capsules, 1).zero_()
        
        for itr in range(self.routing_iters):
            c_ij = func.softmax(b_ij, dim=2)
            s_j  = (c_ij * u_hat).sum(dim=1, keepdim=True) + self.bias
            v_j  = squash(s_j, dim=-1)
            
            if itr < self.routing_iters-1:
                a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
                b_ij = b_ij + a_ij
        v_j = v_j.squeeze() #.unsqueeze(-1)
        
        return v_j   # dim: (batch, num_capsules, out_channels or dim_capsules)




class Decoder_mnist(nn.Module):
    def __init__(self, caps_size=16, num_caps=1, img_size=28, img_channels=1):
        super().__init__()
        
        self.num_caps = num_caps
        self.img_channels = img_channels
        self.img_size = img_size

        self.dense = torch.nn.Linear(caps_size*num_caps, 7*7*16).to(device)
        self.relu = nn.ReLU(inplace=True)
                                     
        
        self.reconst_layers1 = nn.Sequential(nn.BatchNorm2d(num_features=16, momentum=0.8),
                                            
                                            nn.ConvTranspose2d(in_channels=16, out_channels=64, 
                                                               kernel_size=3, stride=1, padding=1
                                                              )
                                            )
        
        self.reconst_layers2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, 
                                                  kernel_size=3, stride=2, padding=1
                                                 )
                               
        self.reconst_layers3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, 
                                                  kernel_size=3, stride=2, padding=1
                                                 )
                                            
        self.reconst_layers4 = nn.ConvTranspose2d(in_channels=16, out_channels=1, 
                                                  kernel_size=3, stride=1, padding=1
                                                 )
                                            
        self.reconst_layers5 = nn.ReLU()
                                           
    
    
    def forward(self, x):
        # x.shape = (batch, 1, capsule_dim(=32 for MNIST))
        batch = x.shape[0]
        
        x = x.type(torch.FloatTensor)

        x = x.to(device)
        
        x = self.dense(x)
        x = self.relu(x)
        x = x.reshape(-1, 16, 7, 7)
        
        x = self.reconst_layers1(x)
        
        x = self.reconst_layers2(x)
        
        # padding
        p2d = (1, 0, 1, 0)
        x = func.pad(x, p2d, "constant", 0)
        x = self.reconst_layers3(x)

        # padding
        p2d = (1, 0, 1, 0)
        x = func.pad(x, p2d, "constant", 0)
        x = self.reconst_layers4(x)      
        
        x = self.reconst_layers5(x)
        x = x.reshape(-1, 1, self.img_size, self.img_size)
        return x  # dim: (batch, 1, imsize, imsize)



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=128,
                                kernel_size=3, stride=1, padding=1)
        self.batchNorm = torch.nn.BatchNorm2d(num_features=128, eps=1e-08, momentum=0.99)
        self.toCaps = ConvertToCaps()
        
        self.conv2dCaps1_nj_4_strd_2 = Conv2DCapsdownsample(h=28, w=28, ch_i=128, n_i=1, ch_j=32, n_j=4, kernel_size=3, stride=2, r_num=1)
        self.conv2dCaps1_nj_4_strd_1_1 = Conv2DCaps(h=14, w=14, ch_i=32, n_i=4, ch_j=32, n_j=4, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps1_nj_4_strd_1_2 = Conv2DCaps(h=14, w=14, ch_i=32, n_i=4, ch_j=32, n_j=4, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps1_nj_4_strd_1_3 = Conv2DCaps(h=14, w=14, ch_i=32, n_i=4, ch_j=32, n_j=4, kernel_size=3, stride=1, r_num=1)
        
        self.conv2dCaps2_nj_8_strd_2 = Conv2DCapsdownsample(h=14, w=14, ch_i=32, n_i=4, ch_j=32, n_j=8, kernel_size=3, stride=2, r_num=1)
        self.conv2dCaps2_nj_8_strd_1_1 = Conv2DCaps(h=7, w=7, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps2_nj_8_strd_1_2 = Conv2DCaps(h=7, w=7, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps2_nj_8_strd_1_3 = Conv2DCaps(h=7, w=7, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        
        self.conv2dCaps3_nj_8_strd_2 = Conv2DCapsdownsample(h=7, w=7, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=2, r_num=1)
        self.conv2dCaps3_nj_8_strd_1_1 = Conv2DCaps(h=4, w=4, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps3_nj_8_strd_1_2 = Conv2DCaps(h=4, w=4, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps3_nj_8_strd_1_3 = Conv2DCaps(h=4, w=4, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        
        self.conv2dCaps4_nj_8_strd_2 = Conv2DCapsdownsample(h=4, w=4, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=2, r_num=1)
        self.conv3dCaps4_nj_8 = Conv2DCaps(h=2, w=2, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps4_nj_8_strd_1_1 = Conv2DCaps(h=2, w=2, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps4_nj_8_strd_1_2 = Conv2DCaps(h=2, w=2, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
                
        self.decoder = Decoder_mnist(caps_size=16, num_caps=1, img_size=28, img_channels=1)
        self.flatCaps = FlattenCaps()
        self.digCaps = CapsuleLayer(num_capsules=10, num_routes=640, in_channels=8, out_channels=16, routing_iters=3)
        self.capsToScalars = CapsToScalars()
        self.mask = Mask_CID()
        self.mse_loss = nn.MSELoss(reduction="none")
    
    def forward(self, x, target=None):
        x = self.conv2d(x)
        x = self.batchNorm(x)
        x = self.toCaps(x)
        
        x = self.conv2dCaps1_nj_4_strd_2(x)
        x_skip = self.conv2dCaps1_nj_4_strd_1_1(x)
        x = self.conv2dCaps1_nj_4_strd_1_2(x)
        x = self.conv2dCaps1_nj_4_strd_1_3(x)
        x = x + x_skip
        
        x = self.conv2dCaps2_nj_8_strd_2(x)
        x_skip = self.conv2dCaps2_nj_8_strd_1_1(x)
        x = self.conv2dCaps2_nj_8_strd_1_2(x)
        x = self.conv2dCaps2_nj_8_strd_1_3(x)
        x = x + x_skip
        
        x = self.conv2dCaps3_nj_8_strd_2(x)
        x_skip = self.conv2dCaps3_nj_8_strd_1_1(x)
        x = self.conv2dCaps3_nj_8_strd_1_2(x)
        x = self.conv2dCaps3_nj_8_strd_1_3(x)
        x = x + x_skip
        x1 = x
        
        x = self.conv2dCaps4_nj_8_strd_2(x)
        x_skip = self.conv3dCaps4_nj_8(x)
        x = self.conv2dCaps4_nj_8_strd_1_1(x)
        x = self.conv2dCaps4_nj_8_strd_1_2(x)
        x = x + x_skip
        x2 = x
        
        xa = self.flatCaps(x1)
        xb = self.flatCaps(x2)
        x = torch.cat((xa, xb), dim=-2)
        dig_caps = self.digCaps(x)
        
        x = self.capsToScalars(dig_caps)
        masked, indices = self.mask(dig_caps, target)
        decoded = self.decoder(masked)

        return dig_caps, masked, decoded, indices
    
    def margin_loss(self, x, labels, lamda, m_plus, m_minus):
        v_c = torch.norm(x, dim=2, keepdim=True)
        tmp1 = func.relu(m_plus - v_c).view(x.shape[0], -1) ** 2
        tmp2 = func.relu(v_c - m_minus).view(x.shape[0], -1) ** 2
        loss = labels*tmp1 + lamda*(1-labels)*tmp2
        loss = loss.sum(dim=1)
        return loss
    
    def reconst_loss(self, recnstrcted, data):
        loss = self.mse_loss(recnstrcted.view(recnstrcted.shape[0], -1), data.view(recnstrcted.shape[0], -1))
        return 0.4 * loss.sum(dim=1)
    
    def loss(self, x, recnstrcted, data, labels, lamda=0.5, m_plus=0.9, m_minus=0.1):
        loss = self.margin_loss(x, labels, lamda, m_plus, m_minus) + self.reconst_loss(recnstrcted, data)
        return loss.mean()

