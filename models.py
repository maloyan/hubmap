import pretrainedmodels
import torch
import torch.nn.functional as F
from torch import nn

from config import config
import timm

def conv3x3(in_channel, out_channel):  # not change resolusion
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    )


def conv1x1(in_channel, out_channel):  # not change resolution
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    )


def init_weight(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # nn.init.xavier_uniform_(m.weight, gain=1)
        # nn.init.xavier_normal_(m.weight, gain=1)
        # nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Batch") != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Embedding") != -1:
        nn.init.orthogonal_(m.weight, gain=1)


class cSEBlock(nn.Module):
    def __init__(self, c, feat):
        super().__init__()
        self.attention_fc = nn.Linear(feat, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros((1, c, 1), requires_grad=True))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs):
        batch, c, h, w = inputs.size()
        x = inputs.view(batch, c, -1)
        x = self.attention_fc(x) + self.bias
        x = x.view(batch, c, 1, 1)
        x = self.sigmoid(x)
        x = self.dropout(x)
        return inputs * x


class sSEBlock(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.attention_fc = nn.Linear(c, 1, bias=False).apply(init_weight)
        self.bias = nn.Parameter(torch.zeros((1, h, w, 1), requires_grad=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        batch, c, h, w = inputs.size()
        x = torch.transpose(inputs, 1, 2)  # (*,c,h,w)->(*,h,c,w)
        x = torch.transpose(x, 2, 3)  # (*,h,c,w)->(*,h,w,c)
        x = self.attention_fc(x) + self.bias
        x = torch.transpose(x, 2, 3)  # (*,h,w,1)->(*,h,1,w)
        x = torch.transpose(x, 1, 2)  # (*,h,1,w)->(*,1,h,w)
        x = self.sigmoid(x)
        return inputs * x


class scSEBlock(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.cSE = cSEBlock(c, h * w)
        self.sSE = sSEBlock(c, h, w)

    def forward(self, inputs):
        x1 = self.cSE(inputs)
        x2 = self.sSE(inputs)
        return x1 + x2


# class SpatialAttention2d(nn.Module):
#     def __init__(self, in_channel):
#         super().__init__()
#         self.squeeze = conv1x1(in_channel,1).apply(init_weight)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, inputs):
#         x = self.squeeze(inputs)
#         x = self.sigmoid(x)
#         return inputs * x


# class GAB(nn.Module):
#     def __init__(self, in_channel, reduction=4):
#         super().__init__()
#         self.global_avgpool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = conv1x1(in_channel, in_channel//reduction).apply(init_weight)
#         self.conv2 = conv1x1(in_channel//reduction, in_channel).apply(init_weight)
#         self.relu  = nn.ReLU(True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, inputs):
#         x = self.global_avgpool(inputs)
#         x = self.relu(self.conv1(x))
#         x = self.sigmoid(self.conv2(x))
#         return inputs * x


# class scSEBlock2(nn.Module):
#     def __init__(self, in_channel, reduction=4):
#         super().__init__()
#         self.cSE = GAB(in_channel, reduction)
#         self.sSE = SpatialAttention2d(in_channel)

#     def forward(self, inputs):
#         x1 = self.cSE(inputs)
#         x2 = self.sSE(inputs)
#         return x1+x2


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta = nn.utils.spectral_norm(conv1x1(channels, channels // 8)).apply(
            init_weight
        )
        self.phi = nn.utils.spectral_norm(conv1x1(channels, channels // 8)).apply(
            init_weight
        )
        self.g = nn.utils.spectral_norm(conv1x1(channels, channels // 2)).apply(
            init_weight
        )
        self.o = nn.utils.spectral_norm(conv1x1(channels // 2, channels)).apply(
            init_weight
        )
        self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, inputs):
        batch, c, h, w = inputs.size()
        theta = self.theta(inputs)  # ->(*,c/8,h,w)
        phi = F.max_pool2d(self.phi(inputs), [2, 2])  # ->(*,c/8,h/2,w/2)
        g = F.max_pool2d(self.g(inputs), [2, 2])  # ->(*,c/2,h/2,w/2)

        theta = theta.view(batch, self.channels // 8, -1)  # ->(*,c/8,h*w)
        phi = phi.view(batch, self.channels // 8, -1)  # ->(*,c/8,h*w/4)
        g = g.view(batch, self.channels // 2, -1)  # ->(*,c/2,h*w/4)

        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)  # ->(*,h*w,h*w/4)
        o = self.o(
            torch.bmm(g, beta.transpose(1, 2)).view(batch, self.channels // 2, h, w)
        )  # ->(*,c,h,w)
        return self.gamma * o + inputs


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            conv1x1(in_channel, in_channel // reduction).apply(init_weight),
            nn.ReLU(True),
            conv1x1(in_channel // reduction, in_channel).apply(init_weight),
        )

    def forward(self, inputs):
        x1 = self.global_maxpool(inputs)
        x2 = self.global_avgpool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x = torch.sigmoid(x1 + x2)
        return x


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = conv3x3(2, 1).apply(init_weight)

    def forward(self, inputs):
        x1, _ = torch.max(inputs, dim=1, keepdim=True)
        x2 = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3x3(x)
        x = torch.sigmoid(x)
        return x


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, inputs):
        x = inputs * self.channel_attention(inputs)
        x = x * self.spatial_attention(x)
        return x


class CenterBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = conv3x3(in_channel, out_channel).apply(init_weight)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module(
                "upsample", nn.Upsample(scale_factor=2, mode="nearest")
            )
        self.conv3x3_1 = conv3x3(in_channel, in_channel).apply(init_weight)
        self.bn2 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.conv3x3_2 = conv3x3(in_channel, out_channel).apply(init_weight)
        self.cbam = CBAM(out_channel, reduction=16)
        self.conv1x1 = conv1x1(in_channel, out_channel).apply(init_weight)

    def forward(self, inputs):
        x = F.relu(self.bn1(inputs))
        x = self.upsample(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(F.relu(self.bn2(x)))
        x = self.cbam(x)
        x += self.conv1x1(self.upsample(inputs))  # shortcut
        return x


# U-Net ResNet34 + CBAM + hypercolumns + deepsupervision
class UNET_RESNET34(nn.Module):
    def __init__(self, resolution, deepsupervision, clfhead, load_weights=True):
        super().__init__()
        h, w = resolution
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead

        # encoder
        model_name = "resnet34"  # 26M
        resnet34 = timm.create_model(model_name, pretrained=True)

        self.conv1 = resnet34.conv1  # (*,3,h,w)->(*,64,h/2,w/2)
        self.bn1 = resnet34.bn1
        self.maxpool = resnet34.maxpool  # ->(*,64,h/4,w/4)
        self.layer1 = resnet34.layer1  # ->(*,64,h/4,w/4)
        self.layer2 = resnet34.layer2  # ->(*,128,h/8,w/8)
        self.layer3 = resnet34.layer3  # ->(*,256,h/16,w/16)
        self.layer4 = resnet34.layer4  # ->(*,512,h/32,w/32)

        # center
        self.center = CenterBlock(512, 512)  # ->(*,512,h/32,w/32)

        # decoder
        self.decoder4 = DecodeBlock(512 + 512, 64, upsample=True)  # ->(*,64,h/16,w/16)
        self.decoder3 = DecodeBlock(64 + 256, 64, upsample=True)  # ->(*,64,h/8,w/8)
        self.decoder2 = DecodeBlock(64 + 128, 64, upsample=True)  # ->(*,64,h/4,w/4)
        self.decoder1 = DecodeBlock(64 + 64, 64, upsample=True)  # ->(*,64,h/2,w/2)
        self.decoder0 = DecodeBlock(64, 64, upsample=True)  # ->(*,64,h,w)

        # upsample
        self.upsample4 = nn.Upsample(
            scale_factor=16, mode="bilinear", align_corners=True
        )
        self.upsample3 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.upsample2 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        if self.deepsupervision:
            # deep supervision
            self.deep4 = conv1x1(64, 1).apply(init_weight)
            self.deep3 = conv1x1(64, 1).apply(init_weight)
            self.deep2 = conv1x1(64, 1).apply(init_weight)
            self.deep1 = conv1x1(64, 1).apply(init_weight)

        # final conv
        self.final_conv = nn.Sequential(
            conv3x3(320, 64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64, 1).apply(init_weight),
        )
        if self.clfhead:
            # clf head
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.clf = nn.Sequential(
                nn.BatchNorm1d(512).apply(init_weight),
                nn.Linear(512, 512).apply(init_weight),
                nn.ELU(True),
                nn.BatchNorm1d(512).apply(init_weight),
                nn.Linear(512, 1).apply(init_weight),
            )

    def forward(self, inputs):
        # encoder
        x0 = F.relu(self.bn1(self.conv1(inputs)))  # ->(*,64,h/2,w/2)
        x0 = self.maxpool(x0)  # ->(*,64,h/4,w/4)
        x1 = self.layer1(x0)  # ->(*,64,h/4,w/4)
        x2 = self.layer2(x1)  # ->(*,128,h/8,w/8)
        x3 = self.layer3(x2)  # ->(*,256,h/16,w/16)
        x4 = self.layer4(x3)  # ->(*,512,h/32,w/32)

        if self.clfhead:
            # clf head
            logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1))  # ->(*,1)
            if config["clf_threshold"] is not None:
                if (torch.sigmoid(logits_clf) > config["clf_threshold"]).sum().item() == 0:
                    bs, _, h, w = inputs.shape
                    logits = torch.zeros((bs, 1, h, w))
                    if self.clfhead:
                        if self.deepsupervision:
                            return logits, _, _
                        else:
                            return logits, _
                    else:
                        if self.deepsupervision:
                            return logits, _
                        else:
                            return logits

        # center
        y5 = self.center(x4)  # ->(*,512,h/32,w/32)

        # decoder
        y4 = self.decoder4(torch.cat([x4, y5], dim=1))  # ->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3, y4], dim=1))  # ->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2, y3], dim=1))  # ->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1, y2], dim=1))  # ->(*,64,h/2,w/2)
        y0 = self.decoder0(y1)  # ->(*,64,h,w)

        # hypercolumns
        y4 = self.upsample4(y4)  # ->(*,64,h,w)
        y3 = self.upsample3(y3)  # ->(*,64,h,w)
        y2 = self.upsample2(y2)  # ->(*,64,h,w)
        y1 = self.upsample1(y1)  # ->(*,64,h,w)
        hypercol = torch.cat([y0, y1, y2, y3, y4], dim=1)

        # final conv
        logits = self.final_conv(hypercol)  # ->(*,1,h,w)

        res = [logits]
        if self.deepsupervision:
            s4 = self.deep4(y4)
            s3 = self.deep3(y3)
            s2 = self.deep2(y2)
            s1 = self.deep1(y1)
            logits_deeps = [s4, s3, s2, s1]
            res.append(logits_deeps)

        if self.clfhead:
            # clf head
            logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1))  # ->(*,1)
            res.append(logits_clf)

        return res

# U-Net SeResNext50 + CBAM + hypercolumns + deepsupervision
class UNET_SERESNEXT50(nn.Module):
    def __init__(self, resolution, deepsupervision, clfhead, load_weights=True):
        super().__init__()
        h, w = resolution
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead

        # encoder
        model_name = "se_resnext50_32x4d"  # 26M
        seresnext50 = pretrainedmodels.__dict__[model_name](pretrained=None)
        if load_weights:
            seresnext50.load_state_dict(
                torch.load(f"../../../pretrainedmodels_weight/{model_name}.pth")
            )

        self.encoder0 = nn.Sequential(
            seresnext50.layer0.conv1,  # (*,3,h,w)->(*,64,h/2,w/2)
            seresnext50.layer0.bn1,
            seresnext50.layer0.relu1,
        )
        self.encoder1 = nn.Sequential(
            seresnext50.layer0.pool,  # ->(*,64,h/4,w/4)
            seresnext50.layer1,  # ->(*,256,h/4,w/4)
        )
        self.encoder2 = seresnext50.layer2  # ->(*,512,h/8,w/8)
        self.encoder3 = seresnext50.layer3  # ->(*,1024,h/16,w/16)
        self.encoder4 = seresnext50.layer4  # ->(*,2048,h/32,w/32)

        # center
        self.center = CenterBlock(2048, 512)  # ->(*,512,h/32,w/32) 10,16

        # decoder
        self.decoder4 = DecodeBlock(
            512 + 2048, 64, upsample=True
        )  # ->(*,64,h/16,w/16) 20,32
        self.decoder3 = DecodeBlock(
            64 + 1024, 64, upsample=True
        )  # ->(*,64,h/8,w/8) 40,64
        self.decoder2 = DecodeBlock(
            64 + 512, 64, upsample=True
        )  # ->(*,64,h/4,w/4) 80,128
        self.decoder1 = DecodeBlock(
            64 + 256, 64, upsample=True
        )  # ->(*,64,h/2,w/2) 160,256
        self.decoder0 = DecodeBlock(64, 64, upsample=True)  # ->(*,64,h,w) 320,512

        # upsample
        self.upsample4 = nn.Upsample(
            scale_factor=16, mode="bilinear", align_corners=True
        )
        self.upsample3 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.upsample2 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        # deep supervision
        self.deep4 = conv1x1(64, 1).apply(init_weight)
        self.deep3 = conv1x1(64, 1).apply(init_weight)
        self.deep2 = conv1x1(64, 1).apply(init_weight)
        self.deep1 = conv1x1(64, 1).apply(init_weight)

        # final conv
        self.final_conv = nn.Sequential(
            conv3x3(320, 64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64, 1).apply(init_weight),
        )

        # clf head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(
            nn.BatchNorm1d(2048).apply(init_weight),
            nn.Linear(2048, 512).apply(init_weight),
            nn.ELU(True),
            nn.BatchNorm1d(512).apply(init_weight),
            nn.Linear(512, 1).apply(init_weight),
        )

    def forward(self, inputs):
        # encoder
        x0 = self.encoder0(inputs)  # ->(*,64,h/2,w/2) 160,256
        x1 = self.encoder1(x0)  # ->(*,256,h/4,w/4)
        x2 = self.encoder2(x1)  # ->(*,512,h/8,w/8)
        x3 = self.encoder3(x2)  # ->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3)  # ->(*,2048,h/32,w/32)

        # clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1))  # ->(*,1)
        if config["clf_threshold"] is not None:
            if (torch.sigmoid(logits_clf) > config["clf_threshold"]).sum().item() == 0:
                bs, _, h, w = inputs.shape
                logits = torch.zeros((bs, 1, h, w))
                if self.clfhead:
                    if self.deepsupervision:
                        return logits, _, _
                    else:
                        return logits, _
                else:
                    if self.deepsupervision:
                        return logits, _
                    else:
                        return logits

        # center
        y5 = self.center(x4)  # ->(*,320,h/32,w/32)

        # decoder
        y4 = self.decoder4(torch.cat([x4, y5], dim=1))  # ->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3, y4], dim=1))  # ->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2, y3], dim=1))  # ->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1, y2], dim=1))  # ->(*,64,h/2,w/2) 160,256
        y0 = self.decoder0(y1)  # ->(*,64,h,w) 320,512

        # hypercolumns
        y4 = self.upsample4(y4)  # ->(*,64,h,w)
        y3 = self.upsample3(y3)  # ->(*,64,h,w)
        y2 = self.upsample2(y2)  # ->(*,64,h,w)
        y1 = self.upsample1(y1)  # ->(*,64,h,w)
        hypercol = torch.cat([y0, y1, y2, y3, y4], dim=1)

        # final conv
        logits = self.final_conv(hypercol)  # ->(*,4,h,w)

        # clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1))  # ->(*,1)

        if self.clfhead:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4, s3, s2, s1]
                return logits, logits_deeps, logits_clf
            else:
                return logits, logits_clf
        else:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4, s3, s2, s1]
                return logits, logits_deeps
            else:
                return logits


# U-Net SeResNext101 + CBAM + hypercolumns + deepsupervision
class UNET_SERESNEXT101(nn.Module):
    def __init__(self, resolution, deepsupervision, clfhead, load_weights=True):
        super().__init__()
        h, w = resolution
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead

        # encoder
        #model_name = "se_resnext101_32x4d"
        model_name = "seresnext101_32x4d"

        seresnext101 = timm.create_model(model_name, pretrained=True) #pretrainedmodels.__dict__[model_name](pretrained=None)
        # if load_weights:
        #     seresnext101.load_state_dict(
        #         torch.load(f"../../../pretrainedmodels_weight/{model_name}.pth")
        #     )

        self.encoder0 = nn.Sequential(
            seresnext101.conv1,  # (*,3,h,w)->(*,64,h/2,w/2)
            seresnext101.bn1,
            seresnext101.act1,
        )
        self.encoder1 = nn.Sequential(
            seresnext101.maxpool,  # ->(*,64,h/4,w/4)
            seresnext101.layer1,  # ->(*,256,h/4,w/4)
        )
        self.encoder2 = seresnext101.layer2  # ->(*,512,h/8,w/8)
        self.encoder3 = seresnext101.layer3  # ->(*,1024,h/16,w/16)
        self.encoder4 = seresnext101.layer4  # ->(*,2048,h/32,w/32)

        # center
        self.center = CenterBlock(2048, 512)  # ->(*,512,h/32,w/32)

        # decoder
        self.decoder4 = DecodeBlock(512 + 2048, 64, upsample=True)  # ->(*,64,h/16,w/16)
        self.decoder3 = DecodeBlock(64 + 1024, 64, upsample=True)  # ->(*,64,h/8,w/8)
        self.decoder2 = DecodeBlock(64 + 512, 64, upsample=True)  # ->(*,64,h/4,w/4)
        self.decoder1 = DecodeBlock(64 + 256, 64, upsample=True)  # ->(*,64,h/2,w/2)
        self.decoder0 = DecodeBlock(64, 64, upsample=True)  # ->(*,64,h,w)

        # upsample
        self.upsample4 = nn.Upsample(
            scale_factor=16, mode="bilinear", align_corners=True
        )
        self.upsample3 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.upsample2 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        if self.deepsupervision:
            # deep supervision
            self.deep4 = conv1x1(64, 1).apply(init_weight)
            self.deep3 = conv1x1(64, 1).apply(init_weight)
            self.deep2 = conv1x1(64, 1).apply(init_weight)
            self.deep1 = conv1x1(64, 1).apply(init_weight)

        # final conv
        self.final_conv = nn.Sequential(
            conv3x3(320, 64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64, 1).apply(init_weight),
        )

        # clf head
        if self.clfhead:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.clf = nn.Sequential(
                nn.BatchNorm1d(2048).apply(init_weight),
                nn.Linear(2048, 512).apply(init_weight),
                nn.ELU(True),
                nn.BatchNorm1d(512).apply(init_weight),
                nn.Linear(512, 1).apply(init_weight),
            )

    def forward(self, inputs):
        # encoder
        x0 = self.encoder0(inputs)  # ->(*,64,h/2,w/2)
        x1 = self.encoder1(x0)  # ->(*,256,h/4,w/4)
        x2 = self.encoder2(x1)  # ->(*,512,h/8,w/8)
        x3 = self.encoder3(x2)  # ->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3)  # ->(*,2048,h/32,w/32)

        if self.clfhead:
            # clf head
            logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1))  # ->(*,1)
            if config["clf_threshold"] is not None:
                if (torch.sigmoid(logits_clf) > config["clf_threshold"]).sum().item() == 0:
                    bs, _, h, w = inputs.shape
                    logits = torch.zeros((bs, 1, h, w))
                    if self.clfhead:
                        if self.deepsupervision:
                            return logits, _, _
                        else:
                            return logits, _
                    else:
                        if self.deepsupervision:
                            return logits, _
                        else:
                            return logits

        # center
        y5 = self.center(x4)  # ->(*,320,h/32,w/32)

        # decoder
        y4 = self.decoder4(torch.cat([x4, y5], dim=1))  # ->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3, y4], dim=1))  # ->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2, y3], dim=1))  # ->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1, y2], dim=1))  # ->(*,64,h/2,w/2)
        y0 = self.decoder0(y1)  # ->(*,64,h,w)

        # hypercolumns
        y4 = self.upsample4(y4)  # ->(*,64,h,w)
        y3 = self.upsample3(y3)  # ->(*,64,h,w)
        y2 = self.upsample2(y2)  # ->(*,64,h,w)
        y1 = self.upsample1(y1)  # ->(*,64,h,w)
        hypercol = torch.cat([y0, y1, y2, y3, y4], dim=1)

        # final conv
        logits = self.final_conv(hypercol)  # ->(*,1,h,w)

        res = [logits]
        if self.deepsupervision:
            s4 = self.deep4(y4)
            s3 = self.deep3(y3)
            s2 = self.deep2(y2)
            s1 = self.deep1(y1)
            logits_deeps = [s4, s3, s2, s1]
            res.append(logits_deeps)

        if self.clfhead:
            # clf head
            logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1))  # ->(*,1)
            res.append(logits_clf)

        return res

def build_model(resolution, deepsupervision, clfhead, load_weights):
    model_name = config["model_name"]
    if model_name == "unet_resnet34":
        model = UNET_RESNET34(resolution, deepsupervision, clfhead, load_weights)
    elif model_name == "seresnext50":
        model = UNET_SERESNEXT50(resolution, deepsupervision, clfhead, load_weights)
    elif model_name == "seresnext101":
        model = UNET_SERESNEXT101(resolution, deepsupervision, clfhead, load_weights)
    return model
