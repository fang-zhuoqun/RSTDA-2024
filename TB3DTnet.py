import random
import numpy as np
import torch
# import cv2
import math
import torch.nn.functional as F
from torch import nn
# from skimage import transform
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class OurFE(nn.Module):
    def __init__(self, channel, dim):
        super(OurFE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(3 * channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out = self.out_conv(torch.cat((out1, out2, out3), dim=1))
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            DEPTHWISECONV(dim, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=dim, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x):
        b, d, c = x.shape
        w = int(math.sqrt(d))
        x1 = rearrange(x, 'b (w h) c -> b c w h', w=w, h=w)
        x1 = self.net(x1)
        x1 = rearrange(x1, 'b c w h -> b (w h) c')
        x = x + x1
        return x


class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, padding=0, stride=1, is_fe=False):
        super(DEPTHWISECONV, self).__init__()
        self.is_fe = is_fe
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        if self.is_fe:
            return out
        out = self.point_conv(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0., num_patches=10):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.spatial_norm = nn.BatchNorm2d(heads)
        self.spatial_conv = nn.Conv2d(heads, heads, kernel_size=3, padding=1)

        self.spectral_norm = nn.BatchNorm2d(1)
        self.spectral_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.to_qkv_spec = nn.Linear(num_patches, num_patches*3, bias=False)
        self.attend_spec = nn.Softmax(dim=-1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        # attn = self.spatial_conv(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        output = self.to_out(out)

        x = x.transpose(-2, -1)
        qkv_spec = self.to_qkv_spec(x).chunk(3, dim=-1)
        q_spec, k_spec, v_spec = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=1), qkv_spec)
        dots_spec = torch.matmul(q_spec, k_spec.transpose(-1, -2)) * self.scale
        attn = self.attend_spec(dots_spec)  # .squeeze(dim=1)
        attn = self.spectral_conv(attn).squeeze(dim=1)

        return torch.matmul(output, attn)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., num_patches=25):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.index = 0
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim)),
            ]))

    def forward(self, x):
        # output = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            # output.append(x)

        # return x, output
        return x


class SubNet(nn.Module):
    def __init__(self, patch_size, num_patches, dim, emb_dropout, depth, heads, dim_head, mlp_dim, dropout):
        super(SubNet, self).__init__()
        self.to_patch_embedding = nn.Sequential(
            DEPTHWISECONV(in_ch=dim, out_ch=dim, kernel_size = patch_size, stride = patch_size, padding=0, is_fe=True),
            Rearrange('b c w h -> b (h w) c '),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches+1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, dropout=dropout, num_patches=num_patches)


def get_num_patches(ps, ks):
    return int((ps - ks)/ks)+1


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super(ViT, self).__init__()
        # self._out_features = dim
        self.ournet = OurFE(channels, dim)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=dim, kernel_size=1)
        self.net = nn.Sequential()
        self.mlp_head = nn.ModuleList()
        for ps in patch_size:
            num_patches = get_num_patches(image_size, ps) ** 2
            patch_dim = dim * num_patches
            sub_net = SubNet(ps, num_patches, dim, emb_dropout, depth, heads, dim_head, mlp_dim, dropout)
            self.net.append(sub_net)
            self.mlp_head.append(nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, num_classes)
            ))

        self.weight = torch.ones(len(patch_size))

    def forward(self, img):
        if len(img.shape) == 5: img = img.squeeze()
        img = self.ournet(img)
        img = self.pool(img)
        img = self.conv4(img)

        all_branch = []
        for sub_branch in self.net:
            spatial = sub_branch.to_patch_embedding(img)
            b, n, c = spatial.shape
            spatial = spatial + sub_branch.pos_embedding[:, :n]
            spatial = sub_branch.dropout(spatial)
            _, outputs = sub_branch.transformer(spatial)
            res = outputs[-1]
            all_branch.append(res)

        self.weight = F.softmax(self.weight, 0)
        res = 0
        out2 = []
        for i, mlp_head in enumerate(self.mlp_head):
            out1 = all_branch[i].flatten(start_dim=1)
            # cls1 = mlp_head(out1)
            # res = res + cls1 * self.weight[i]
        # return res
            out2.append(out1)
        out = torch.cat((out2[0], out2[1]), dim=1)
        return out

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return 3400  # 900+2500
        # return 2500

class TB3DTnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=7, reduced_channels=30, num_tokens=4, dim=64, depth=1,
                 heads=8, mlp_dim=8,
                 dropout=0.1, emb_dropout=0.1,
                 kernel3d_size=3, kernel3d_depth=3, conv3d_kernels=8, kernel2d_size=3, conv2d_kernels=64,
                 is_pca=False):
        super(TB3DTnet, self).__init__()
        self.L = num_tokens
        self.cT = dim  # 512
        self.conv3d_features = nn.Sequential(
            # nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.Conv3d(1, out_channels=conv3d_kernels, kernel_size=(kernel3d_depth, kernel3d_size, kernel3d_size)),
            nn.BatchNorm3d(conv3d_kernels),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            # nn.Conv2d(in_channels=8*28, out_channels=64, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=conv3d_kernels * (reduced_channels - kernel3d_depth + 1),
                      out_channels=conv2d_kernels,
                      kernel_size=(kernel2d_size, kernel2d_size)),
            nn.BatchNorm2d(conv2d_kernels),
            nn.ReLU(),
        )

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, conv2d_kernels),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        # self.token_wV = nn.Parameter(torch.empty(1, conv2d_kernels, self.cT),
        #                              requires_grad=True)  # Tokenization parameters
        # torch.nn.init.xavier_normal_(self.token_wV)

        # self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens, dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head=64, dropout=dropout, num_patches=num_tokens)

        self.to_cls_token = nn.Identity()

        # self.nn1 = nn.Linear(dim, num_classes)
        # self.nn1 = nn.Linear(dim, 128) # add
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        # Add dim reduced block instead of PCA
        self.is_pca = is_pca
        if not self.is_pca:
            self.channel_reduce = nn.Conv2d(in_channels, reduced_channels, kernel_size=1,
                                            padding=0)  # input channels for Pavia Dataset
        self.n_outputs = self.cT

    def output_num(self):
        return self.cT  # input channels for Pavia Dataset

    def get_embedding(self, x):
        return self.forward(x)

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self.cT

    def forward(self, x, mask=None):
        # Add dim reduced block instead of PCA
        if not self.is_pca:
            x = self.channel_reduce(x)  # input channels for Pavia Dataset
        x = x.unsqueeze(1)

        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        # VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        # T = torch.einsum('bij,bjk->bik', A, VV)

        T = torch.einsum('bij,bjk->bik', A, x)  # better in Pavia and Houston.

        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, T), dim=1)
        x = T
        x += self.pos_embedding
        x = self.dropout(x)
        # x = self.transformer(x, mask)  # main game
        x = self.transformer(x)  # main game
        # x = self.to_cls_token(x[:, 0])
        x = x.mean(dim=1)
        # x = self.nn1(x)

        return x

# Plan B: Two Branches.

class SpectralSE(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False) #4-->16
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SSSE(nn.Module):

    def __init__(self, input_channels, patch_size, dim=192):
        super(SSSE, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='replicate',
                               bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='replicate',
                               bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, dim, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(dim)
        self.activation4 = nn.ReLU()
        # self.SpectralSE = SpectralSE(dim, dim, self.sz)
        self.SpectralSE = ChannelAttention(dim)#self.inter_size
        # self.SpectralSE = SpectralSE_R(128, 128, self.sz)
        # self.SpectralSE = SpectralSE_S(128, 128, self.sz)

        # # Convolution layer for spatial information
        # self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        # self.bn5 = nn.BatchNorm3d(24)
        # self.activation5 = nn.ReLU()
        #
        # # Residual block 2
        # self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
        #                        bias=True)
        # self.bn6 = nn.BatchNorm3d(24)
        # self.activation6 = nn.ReLU()
        # self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
        #                        bias=True)
        # self.bn7 = nn.BatchNorm3d(24)
        # self.activation7 = nn.ReLU()
        # self.SpatialSE = SpatialSE(24, 1)
        # self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # # Finish
        #
        # # Combination shape
        # self.inter_size = 128 + 24
        #
        # # Residual block 3
        # self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
        #                        padding_mode='replicate', bias=True)
        # self.bn9 = nn.BatchNorm3d(self.inter_size)
        # self.activation9 = nn.ReLU()
        # self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
        #                         padding_mode='replicate', bias=True)
        # self.bn10 = nn.BatchNorm3d(self.inter_size)
        # self.activation10 = nn.ReLU()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # # Fully connected Layer
        # self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, bounds=None):
        # Convolution layer 1
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
        e1 = self.SpectralSE(x1)
        x1 = torch.mul(e1, x1)

        # x2 = self.conv5(x)
        # x2 = self.activation5(self.bn5(x2))
        #
        # # Residual layer 2
        # residual = x2
        # residual = self.conv8(residual)
        # x2 = self.conv6(x2)
        # x2 = self.activation6(self.bn6(x2))
        # x2 = self.conv7(x2)
        # x2 = residual + x2
        #
        # x2 = self.activation7(self.bn7(x2))
        # x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))
        # e2 = self.SpatialSE(x2)
        # x2 = torch.mul(e2,x2)
        #
        # # concat spatial and spectral information
        # x = torch.cat((x1, x2), 1)

        x = self.avgpool(x1)
        x = x.reshape((x.size(0), -1))

        # Fully connected layer
        # x = self.fc1(x)

        return x

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        # return 152
        return 192

class TB3DTnet2(nn.Module):
    def __init__(self, in_channels=1, num_classes=7, reduced_channels=30, num_tokens=4, dim=64, depth=1,
                 heads=8, mlp_dim=8,
                 dropout=0.1, emb_dropout=0.1,
                 kernel3d_size=3, kernel3d_depth=3, conv3d_kernels=8, kernel2d_size=3, conv2d_kernels=64,
                 is_pca=False, patch_size=7):
        super(TB3DTnet2, self).__init__()
        self.L = num_tokens
        self.cT = dim  # 512
        self.dim = dim
        self.conv3d_features = nn.Sequential(
            # nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.Conv3d(1, out_channels=conv3d_kernels, kernel_size=(kernel3d_depth, kernel3d_size, kernel3d_size)),
            nn.BatchNorm3d(conv3d_kernels),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            # nn.Conv2d(in_channels=8*28, out_channels=64, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=conv3d_kernels * (reduced_channels - kernel3d_depth + 1),
                      out_channels=conv2d_kernels,
                      kernel_size=(kernel2d_size, kernel2d_size)),
            # nn.BatchNorm2d(conv2d_kernels),
            # nn.ReLU(),
        )

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, conv2d_kernels),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        # self.token_wV = nn.Parameter(torch.empty(1, conv2d_kernels, self.cT),
        #                              requires_grad=True)  # Tokenization parameters
        # torch.nn.init.xavier_normal_(self.token_wV)

        # self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens, self.dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.transformer = Transformer(self.dim, depth, heads, dim_head=64, dropout=dropout, num_patches=num_tokens)

        self.to_cls_token = nn.Identity()

        # self.nn1 = nn.Linear(dim, num_classes)
        # self.nn1 = nn.Linear(dim, 128) # add
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        # Add dim reduced block instead of PCA
        self.is_pca = is_pca
        if not self.is_pca:
            self.channel_reduce = nn.Conv2d(in_channels, reduced_channels, kernel_size=1,
                                            padding=0)  # input channels for Pavia Dataset
        self.n_outputs = self.cT

        # self.SpecStream = SSSE(in_channels, patch_size, self.dim)
        self.ourFE = OurFE(in_channels)

    def output_num(self):
        return self.cT  # input channels for Pavia Dataset

    def get_embedding(self, x):
        return self.forward(x)

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self.cT

    def forward(self, x, mask=None):
        # Add dim reduced block instead of PCA
        # if not self.is_pca:
        #     x1 = self.channel_reduce(x)  # input channels for Pavia Dataset
        # else:
        #     x1 = x
        x1 = self.ourFE(x)
        x1 = x1.unsqueeze(1)

        x1 = self.conv3d_features(x1)
        x1 = rearrange(x1, 'b c h w y -> b (c h) w y')
        x1 = self.conv2d_features(x1)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')

        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x1, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        # VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        # T = torch.einsum('bij,bjk->bik', A, VV)

        x1 = torch.einsum('bij,bjk->bik', A, x1)  # better in Pavia and Houston.

        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, T), dim=1)
        x1 += self.pos_embedding
        x1 = self.dropout(x1)
        # x = self.transformer(x, mask)  # main game
        x1 = self.transformer(x1)  # main game
        # x = self.to_cls_token(x[:, 0])
        x1 = x1.mean(dim=1)

        # # Spectral Stream.
        # x2 = self.SpecStream(x)
        # x = torch.cat((x1, x2), dim=1)

        return x1


class OurFE(nn.Module):
    def __init__(self, channel):
        super(OurFE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(3 * channel, channel, kernel_size=1)
            # nn.BatchNorm2d(channel),
            # nn.ReLU()
        )
        # self.conv3d_features = nn.Sequential(
        #     # nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
        #     nn.Conv3d(1, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(),
        # )
        # self.r_channels = nn.Conv2d(32 * (channel-2), out_channels=channel, kernel_size=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out = self.out_conv(torch.cat((out1, out2, out3), dim=1))

        # out = out.unsqueeze(1)
        # out = self.conv3d_features(out)
        # out = rearrange(out, 'b c h w y -> b (c h) w y')
        # # out = out.reshape(out.size(0), out.size(1), out.size(3), out.size(4))
        # out = self.r_channels(out)
        return out

