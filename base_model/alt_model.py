import timm
from glob import glob
from tqdm import tqdm
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn

class TimmSegModel(nn.Module):
    def __init__(self, backbone, segtype='unet', pretrained=True, attention_type = None, out_dim=10, in_chan=3, reshape=True):#att_type: scse
        super(TimmSegModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=in_chan,
            features_only=True,
            drop_rate=0.1,
            pretrained=pretrained
        )
        n_blocks = 4
        a = torch.rand(1, in_chan, 512,512)
        g = self.encoder(a)
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]

        self.segmentation_head = nn.Conv2d(decoder_channels[n_blocks-1], out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        if segtype == 'unet':
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
                attention_type = attention_type
            )
        elif segtype == 'unetpp':
            self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
                attention_type = attention_type
            )
        elif segtype == 'link':
            self.decoder = smp.decoders.linknet.decoder.LinknetDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                prefinal_channels=decoder_channels[n_blocks-1],
                n_blocks=n_blocks,
            )
        elif segtype == 'fpn':
            self.decoder = smp.decoders.fpn.decoder.FPNDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                segmentation_channels=decoder_channels[n_blocks-1],
            )
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(decoder_channels[n_blocks-1], 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(16, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            )
        elif segtype == 'pan':
            self.decoder= smp.decoders.pan.decoder.PANDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=32,
            )
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(decoder_channels[n_blocks-1], 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(16, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            )

    def forward(self,x):
        n_blocks = 4
        global_features = [0] + self.encoder(x)[:n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features
    

class TimmSegModel_v2(nn.Module):
    def __init__(self, backbone, segtype='unet', pretrained=True, attention_type =None, out_dim=10, in_chans=3, reshape=True):#att_type: scse
        super(TimmSegModel_v2, self).__init__()

        self.encoder, encoder_chs = get_encoder(
            backbone=backbone,
            pretrained=pretrained,
            in_chans=in_chans,
        )
        self.decoder_channels = [512,256,128,64,32]
        n_blocks = 4

        self.segmentation_head = nn.Conv2d(self.decoder_channels[n_blocks-1], out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        if segtype == 'unet':
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                decoder_channels=self.decoder_channels[:n_blocks],
                n_blocks=n_blocks,
                attention_type = attention_type
            )
        elif segtype == 'unetpp':
            self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                decoder_channels=self.decoder_channels[:n_blocks],
                n_blocks=n_blocks,
                attention_type = attention_type
            )
        elif segtype == 'deeplabv3':
            self.decoder = smp.decoders.deeplabv3.decoder.DeepLabV3PlusDecoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                out_channels=256,
                encoder_depth=n_blocks
            )
        elif segtype == "segformer":
            self.decoder = smp.decoders.segformer.decoder.SegformerDecoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                segmentation_channels=256,
                encoder_depth=n_blocks
            )
        fet = [0] + self.encoder(torch.randn(2, 3, 512, 512))[:4]
        dec = self.decoder(*fet)
        seg = self.segmentation_head(dec)
        if seg.shape[2:] != (512, 512):
            scaler_factor = 512 // seg.shape[2]
            print(f"output shape is {seg.shape}, scaler factor is {scaler_factor}")
            self.segmentation_head= nn.Sequential(
                nn.Upsample(scale_factor=scaler_factor, mode='bilinear'),
                nn.Conv2d(self.decoder_channels[n_blocks-1], self.decoder_channels[n_blocks-1]//2, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.decoder_channels[n_blocks-1]//2, out_dim, kernel_size=1)
            )
    #@torch.autocast("cuda", dtype=torch.float16)
    def forward(self,x):
        global_features = [0] + self.encoder(x)[:4]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features

class TimmSegModel_v3(nn.Module):
    def __init__(self, backbone, segtype='deeplabv3', pretrained=True, out_dim=10, in_chans=3, reshape=True):#att_type: scse
        super(TimmSegModel_v3, self).__init__()

        self.encoder, encoder_chs = get_encoder(
            backbone=backbone,
            pretrained=pretrained,
            in_chans=in_chans,
        )
        encoder_chs = encoder_chs[1:]
        self.decoder_channels = [512,256,128,64,32]
        n_blocks = 4

        self.segmentation_head = nn.Conv2d(256, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        if segtype == 'deeplabv3':
            self.decoder = smp.decoders.deeplabv3.decoder.DeepLabV3PlusDecoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                out_channels=256,
                encoder_depth=n_blocks,
                atrous_rates=(12, 24, 36),
                output_stride=16,
                aspp_separable=True,
                aspp_dropout=0.5
            )
        elif segtype == "segformer":
            self.decoder = smp.decoders.segformer.decoder.SegformerDecoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                segmentation_channels=256,
                encoder_depth=n_blocks
            )
        elif segtype == 'unetpp':
            self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                decoder_channels=self.decoder_channels[:n_blocks],
                n_blocks=n_blocks,
                attention_type = None
            )
        fet = self.encoder(torch.randn(2, 3, 512, 512))
        dec = self.decoder(*fet)
        seg = self.segmentation_head(dec)
        if seg.shape[2:] != (512, 512):
            scaler_factor = 512 // seg.shape[2]
            print(f"output shape is {seg.shape}, scaler factor is {scaler_factor}")
            self.segmentation_head= nn.Sequential(
                nn.Upsample(scale_factor=scaler_factor, mode='bilinear'),
                nn.Conv2d(256, 64, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, out_dim, kernel_size=1)
            )
    #@torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self,x):
        global_features = self.encoder(x)
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features

class TimmSegModel_swin(nn.Module):
    def __init__(self, backbone, segtype='unet', pretrained=True, attention_type =None, out_dim=10, in_chans=3, reshape=True):#att_type: scse
        super(TimmSegModel_swin, self).__init__()

        self.encoder, encoder_chs = get_encoder(
            backbone=backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            image_size=384
        )
        self.decoder_channels = [512,256,128,64,32]
        n_blocks = 4

        self.segmentation_head = nn.Conv2d(self.decoder_channels[n_blocks-1], out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        if segtype == 'unet':
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                decoder_channels=self.decoder_channels[:n_blocks],
                n_blocks=n_blocks,
                attention_type = attention_type
            )
        elif segtype == 'unetpp':
            self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                decoder_channels=self.decoder_channels[:n_blocks],
                n_blocks=n_blocks,
                attention_type = attention_type
            )
        elif segtype == 'deeplabv3':
            self.decoder = smp.decoders.deeplabv3.decoder.DeepLabV3PlusDecoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                out_channels=256,
                encoder_depth=n_blocks
            )
        elif segtype == "segformer":
            self.decoder = smp.decoders.segformer.decoder.SegformerDecoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                segmentation_channels=256,
                encoder_depth=n_blocks
            )
        fet = self.encoder(torch.randn(2, 3, 384, 384))[:4]
        fet = [0] + [x.permute(0, 3, 1, 2) for x in fet]
        dec = self.decoder(*fet)
        seg = self.segmentation_head(dec)
        if seg.shape[2:] != (384, 384):
            scaler_factor = 384 // seg.shape[2]
            print(f"output shape is {seg.shape}, scaler factor is {scaler_factor}")
            self.segmentation_head= nn.Sequential(
                nn.Upsample(scale_factor=scaler_factor, mode='bilinear'),
                nn.Conv2d(self.decoder_channels[n_blocks-1], self.decoder_channels[n_blocks-1]//2, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.decoder_channels[n_blocks-1]//2, out_dim, kernel_size=1)
            )
    #@torch.autocast("cuda", dtype=torch.float16)
    def forward(self,x):
        global_features = self.encoder(x)[:4]
        global_features = [0] + [x.permute(0, 3, 1, 2) for x in global_features]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features

def get_encoder(
    backbone: str,
    pretrained: bool = True,
    in_chans: int = 3,
    image_size: int = 512):
    """
    Create a TIMM model as an encoder with 'features_only' mode enabled.
    Dynamically compute the output channel sizes by passing a dummy input.
    
    Args:
        backbone (str): Name of the TIMM backbone (e.g., 'resnet34', 'efficientnet_b0', etc.)
        pretrained (bool): If True, loads pretrained weights if available.
        in_chans (int): Number of input channels to the network.
        out_indices (tuple): Stages of the encoder from which feature maps are extracted.

    Returns:
        encoder (nn.Module): The TIMM encoder in features_only mode.
        encoder_channels (list): The number of channels in each extracted feature map.
    """
    # 1. Create the encoder in "features_only" mode
    encoder = timm.create_model(
        backbone,
        pretrained=pretrained,
        in_chans=in_chans,
        features_only=True,
    )

    # 2. Pass a dummy input to get feature shapes
    dummy_input = torch.zeros((2, in_chans, image_size, image_size))
    with torch.no_grad():
        features = encoder(dummy_input)  # list of feature maps at out_indices
    
    # 3. Build the list of channels (add in_chans as the first encoder layer, though this depends on SMP usage)
    # If your UNetDecoder expects [encoder_chs[0], encoder_chs[1], ..., encoder_chs[n]], 
    # ensure the lengths match your out_indices + 1 for the input channel size (if needed).
    channels = [f.shape[1] for f in features]
    if channels[0] > channels[1]:
        channels = [f.shape[3] for f in features]
    encoder_channels = [1] + channels

    return encoder, encoder_channels

def main():
    model = TimmSegModel_v3("maxvit_tiny_tf_512.in1k", segtype='deeplabv3', attention_type=None)
    print(model(torch.randn(2, 3, 512, 512)).shape)
if __name__ == "__main__":
    main()