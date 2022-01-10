import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO
        input_size = 3
        kernel_sizes = [3,3,5,3,3,5,3,3]
        stride_sizes = [1,1,2,1,1,2,1,1]
        padding_sizes = [1,1,2,1,1,2,1,1]
        out_channel_sizes = [8,8,16,16,16,32,32,32]
        in_channel_sizes = [input_size]+out_channel_sizes[:-1]
        
        layers_conv = []
        layers_bn = []
        
        self.num_layers = len(kernel_sizes)
        
        for i in range(self.num_layers):
            layers_conv.append(nn.Conv2d(in_channels=in_channel_sizes[i],
                                          out_channels=out_channel_sizes[i],
                                          kernel_size=kernel_sizes[i], 
                                          padding=padding_sizes[i],
                                          stride=stride_sizes[i]))
            
            layers_bn.append(nn.BatchNorm2d(out_channel_sizes[i]))
        
        self.layers_conv = nn.ModuleList(layers_conv)
        self.batch_norm = nn.ModuleList(layers_bn)
        self.relu = nn.ReLU(True)
        
        self.final_conv = nn.Conv2d(in_channels=out_channel_sizes[-1],
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
            
        

    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        y = x.clone()
        
        for i in range(self.num_layers):
            y = self.layers_conv[i](y)
            y = self.batch_norm[i](y)
            y = self.relu(y)
            
        y = self.final_conv(y)
        
        return y
        

class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO
        input_size = G
        
        kernel_convrelu = [3,3,3]
        padding_convrelu = [1,1,1]
        stride_convrelu = [1,2,2]
        out_channel_convrelu = [8,16,32]
        in_channel_convrelu = [input_size]+out_channel_convrelu[:-1]
        
        kernel_convtr = [3,3]
        padding_inp_convtr = [1,1]
        stride_convtr = [2,2]
        padding_op_convtr = [1,1]
        out_channel_convtr = [16,8]
        in_channel_convtr = [out_channel_convrelu[-1]]+out_channel_convtr[:-1]
        
        self.num_convrelu = len(kernel_convrelu)
        self.num_convtr = len(kernel_convtr)
        
        layers_convrelu = []
        layers_convtr = []
        
        for i in range(self.num_convrelu):
            layers_convrelu.append(nn.Conv2d(in_channels = in_channel_convrelu[i],
                                             out_channels = out_channel_convrelu[i],
                                             kernel_size = kernel_convrelu[i],
                                             padding=padding_convrelu[i],
                                             stride=stride_convrelu[i]))
            
        for i in range(self.num_convtr):
            layers_convtr.append(nn.ConvTranspose2d(in_channels = in_channel_convtr[i],
                                                    out_channels = out_channel_convtr[i],
                                                    kernel_size = kernel_convtr[i],
                                                    stride = stride_convtr[i],
                                                    padding = padding_inp_convtr[i],
                                                    output_padding = padding_op_convtr[i]))
        self.relu = nn.ReLU(True)
        self.layers_conv_relu = nn.ModuleList(layers_convrelu)
        self.layers_conv_tr = nn.ModuleList(layers_convtr)
        self.conv_final = nn.Conv2d(in_channels=8, 
                                    out_channels=1, 
                                    kernel_size=3,
                                    padding=1,
                                    stride=1)

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        B,G,D,H,W = x.size()
        temp = torch.transpose(x,1,2)
        x_new = torch.reshape(temp, (B*D,G,H,W))
        c = [x_new]
        for i in range(self.num_convrelu):
            y = self.layers_conv_relu[i](c[-1])
            y = self.relu(y)
            c.append(y)
            del y
        
        c_3 = self.layers_conv_tr[0](c[-1])
        c.append(c_3)
        c_4 = self.layers_conv_tr[1](c[-1]+c[2])
        S_bar = self.conv_final(c_4+c[1])        
        S_bar = torch.reshape(S_bar,(B,D,1,H,W))
        S_bar = S_bar.squeeze(2)
        return S_bar

def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # TODO
        hom_2d = torch.cat((x[:,None],y[:,None],torch.ones(x.size(0),1,device=src_fea.device)),dim=1)
        del y, x
        hom_2d = hom_2d.type(torch.float)
        proj = proj.type(torch.float)
        
        pts_3d = torch.mul(hom_2d[:,:,None,None],depth_values[None,None,:,:]) #[H*W,3,B,D]
        pts_3d = torch.transpose(pts_3d,0,1)
        
        del hom_2d
        pts_3d.to(device=src_fea.device)
        pts_3d = torch.cat((pts_3d,torch.ones(1,H*W,B,D))) #[4,H*W,B,D]
        pts_3d = torch.permute(pts_3d,(2,0,1,3)) #[B,4,H*W,D]
        #proj is [B,4,4]
        
        m = torch.mul(proj[:,:,:,None,None],pts_3d[:,None,:,:,:]) #[B,4,4,H*W,D]
        
        pixels_in_src = torch.sum(m,dim=2) #[B,4,H*W,D]
        del pts_3d, m
        
        pixels_in_src = torch.permute(pixels_in_src,(1,2,0,3)) #[4,H*W,B,D]
        
        pixels_in_src = pixels_in_src[:-2]/pixels_in_src[-2]
        
        #Normalise to take values in [-1,1] as required by grid_sample
        pixels_in_src[0] = ((2/W)*pixels_in_src[0])-1
        pixels_in_src[1] = ((2/H)*pixels_in_src[1])-1
        
        pixels_in_src = torch.permute(pixels_in_src,(2,3,1,0)) #Now has size [B,D,H*W,2]
        
        
    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO
    warped_src_fea = F.grid_sample(src_fea, pixels_in_src) #Shape is [B,C,D,H*W]
    warped_src_fea = torch.reshape(warped_src_fea, (B,C,D,H,W))
    
    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    B,C,D,H,W = warped_src_fea.size()
    
    ref_g = torch.split(ref_fea,split_size_or_sections=G,dim=1)
    src_g = torch.split(warped_src_fea,split_size_or_sections=G,dim=1)
    
    ref_g = torch.stack(ref_g)
    src_g = torch.stack(src_g)
    
    S = torch.sum(ref_g[:,:,:,None,:,:]*src_g,dim=0)
    S = S/(C/G)
    
    return S
    

def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    d = torch.sum(torch.mul(p,depth_values[:,:,None,None]),dim=1)
    return d

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    
    l1_diff = torch.abs(depth_gt-depth_est)
    l1_loss = torch.mul(l1_diff,mask)
    
    # TODO 
    loss_mvs = torch.sum(l1_loss)/torch.sum(mask)
    
    return loss_mvs