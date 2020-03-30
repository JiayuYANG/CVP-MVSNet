# CVP-MVSNet
# By: Jiayu Yang
# Date: 2019-08-05

# Note: This file use part of the code from the following projects.
#       Thanks for the authors for the great code.
#       MVSNet: https://github.com/YoYo000/MVSNet
#       MVSNet_pytorch: https://github.com/xy-guo/MVSNet_pytorch

import torch
import torch.nn as nn
from models.modules import *
# Debug:
# import pdb,time
# import matplotlib.pyplot as plt
# from verifications import *

# Feature pyramid
class FeaturePyramid(nn.Module):
    def __init__(self):
        super(FeaturePyramid, self).__init__()
        self.conv0aa = conv(3, 64, kernel_size=3, stride=1)
        self.conv0ba = conv(64,64, kernel_size=3, stride=1)
        self.conv0bb = conv(64,64, kernel_size=3, stride=1)
        self.conv0bc = conv(64,32, kernel_size=3, stride=1)
        self.conv0bd = conv(32,32, kernel_size=3, stride=1)
        self.conv0be = conv(32,32, kernel_size=3, stride=1)
        self.conv0bf = conv(32,16, kernel_size=3, stride=1)
        self.conv0bg = conv(16,16, kernel_size=3, stride=1)
        self.conv0bh = conv(16,16, kernel_size=3, stride=1)

    def forward(self, img, scales=5):
        fp = []
        f = self.conv0aa(img)
        f = self.conv0bh(self.conv0bg(self.conv0bf(self.conv0be(self.conv0bd(self.conv0bc(self.conv0bb(self.conv0ba(f))))))))
        fp.append(f)
        for scale in range(scales-1):
            img = nn.functional.interpolate(img,scale_factor=0.5,mode='bilinear',align_corners=None).detach()
            f = self.conv0aa(img)
            f = self.conv0bh(self.conv0bg(self.conv0bf(self.conv0be(self.conv0bd(self.conv0bc(self.conv0bb(self.conv0ba(f))))))))
            fp.append(f)

        return fp

class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()

        self.conv0 = ConvBnReLU3D(16, 16, kernel_size=3, pad=1)
        self.conv0a = ConvBnReLU3D(16, 16, kernel_size=3, pad=1)

        self.conv1 = ConvBnReLU3D(16, 32,stride=2, kernel_size=3, pad=1)
        self.conv2 = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        self.conv2a = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        self.conv3 = ConvBnReLU3D(32, 64, kernel_size=3, pad=1)
        self.conv4 = ConvBnReLU3D(64, 64, kernel_size=3, pad=1)
        self.conv4a = ConvBnReLU3D(64, 64, kernel_size=3, pad=1)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=0, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.prob0 = nn.Conv3d(16, 1, 3, stride=1, padding=1)

    def forward(self, x):

        conv0 = self.conv0a(self.conv0(x))
        conv2 = self.conv2a(self.conv2(self.conv1(conv0)))
        conv4 = self.conv4a(self.conv4(self.conv3(conv2)))

        conv5 = conv2+self.conv5(conv4)

        conv6 = conv0+self.conv6(conv5)
        prob = self.prob0(conv6).squeeze(1)

        return prob

class network(nn.Module):
    def __init__(self, args):
        super(network, self).__init__()
        self.featurePyramid = FeaturePyramid()
        self.cost_reg_refine = CostRegNet()
        self.args = args

    def forward(self, ref_img, src_imgs, ref_in, src_in, ref_ex, src_ex, \
                    depth_min, depth_max):

        ## Initialize output list for loss
        depth_est_list = []
        output = {}

        ## Feature extraction
        ref_feature_pyramid = self.featurePyramid(ref_img,self.args.nscale)

        src_feature_pyramids = []
        for i in range(self.args.nsrc):
            src_feature_pyramids.append(self.featurePyramid(src_imgs[:,i,:,:,:],self.args.nscale))

        # Pre-conditioning corresponding multi-scale intrinsics for the feature:
        ref_in_multiscales = conditionIntrinsics(ref_in,ref_img.shape,[feature.shape for feature in ref_feature_pyramid])
        src_in_multiscales = []
        for i in range(self.args.nsrc):
            src_in_multiscales.append(conditionIntrinsics(src_in[:,i],ref_img.shape, [feature.shape for feature in src_feature_pyramids[i]]))
        src_in_multiscales = torch.stack(src_in_multiscales).permute(1,0,2,3,4)

        ## Estimate initial coarse depth map
        depth_hypos = calSweepingDepthHypo(ref_in_multiscales[:,-1],src_in_multiscales[:,0,-1],ref_ex,src_ex,depth_min, depth_max)

        ref_volume = ref_feature_pyramid[-1].unsqueeze(2).repeat(1, 1, len(depth_hypos[0]), 1, 1)

        volume_sum = ref_volume
        volume_sq_sum = ref_volume.pow_(2)
        if self.args.mode == "test":
            del ref_volume
        for src_idx in range(self.args.nsrc):
            # warpped features
            warped_volume = homo_warping(src_feature_pyramids[src_idx][-1], ref_in_multiscales[:,-1], src_in_multiscales[:,src_idx,-1,:,:], ref_ex, src_ex[:,src_idx], depth_hypos)


            if self.args.mode == "train":
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            elif self.args.mode == "test":
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
                del warped_volume
            else: 
                print("Wrong!")
                pdb.set_trace()
                
        # Aggregate multiple feature volumes by variance
        cost_volume = volume_sq_sum.div_(self.args.nsrc+1).sub_(volume_sum.div_(self.args.nsrc+1).pow_(2))
        if self.args.mode == "test":
            del volume_sum
            del volume_sq_sum

        # Regularize cost volume
        cost_reg = self.cost_reg_refine(cost_volume)

        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_hypos)
        depth_est_list.append(depth)

        ## Upsample depth map and refine along feature pyramid
        for level in range(self.args.nscale-2,-1,-1):

            # Upsample
            depth_up = nn.functional.interpolate(depth[None,:],size=None,scale_factor=2,mode='bicubic',align_corners=None)
            depth_up = depth_up.squeeze(0)
            # Generate depth hypothesis
            depth_hypos = calDepthHypo(self.args,depth_up,ref_in_multiscales[:,level,:,:], src_in_multiscales[:,:,level,:,:],ref_ex,src_ex,depth_min, depth_max,level)

            cost_volume = proj_cost(self.args,ref_feature_pyramid[level],src_feature_pyramids,level,ref_in_multiscales[:,level,:,:], src_in_multiscales[:,:,level,:,:], ref_ex, src_ex[:,:],depth_hypos)

            cost_reg2 = self.cost_reg_refine(cost_volume)
            if self.args.mode == "test":
                del cost_volume
            
            prob_volume = F.softmax(cost_reg2, dim=1)
            if self.args.mode == "test":
                del cost_reg2

            # Depth regression
            depth = depth_regression_refine(prob_volume, depth_hypos)

            depth_est_list.append(depth)

        # Photometric confidence
        with torch.no_grad():
            num_depth = prob_volume.shape[1]
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            prob_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        if self.args.mode == "test":
            del prob_volume

        ## Return
        depth_est_list.reverse() # Reverse the list so that depth_est_list[0] is the largest scale.
        output["depth_est_list"] = depth_est_list
        output["prob_confidence"] = prob_confidence

        return output

def sL1_loss(depth_est, depth_gt, mask):
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

def MSE_loss(depth_est, depth_gt, mask):
    return F.mse_loss(depth_est[mask], depth_gt[mask], size_average=True)



