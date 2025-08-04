import timm
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_encoder(
    backbone: str,
    pretrained: bool = True,
    in_chans: int = 3,):
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
        features_only=True,
        pretrained=pretrained,
        in_chans=in_chans,
    )

    # 2. Pass a dummy input to get feature shapes
    dummy_input = torch.zeros((2, in_chans, 512, 512))
    with torch.no_grad():
        features = encoder(dummy_input)  # list of feature maps at out_indices
    
    # 3. Build the list of channels (add in_chans as the first encoder layer, though this depends on SMP usage)
    # If your UNetDecoder expects [encoder_chs[0], encoder_chs[1], ..., encoder_chs[n]], 
    # ensure the lengths match your out_indices + 1 for the input channel size (if needed).
    encoder_channels = [1] + [f.shape[1] for f in features]

    return encoder, encoder_channels

class UNET(nn.Module):
    """
    TIMM エンコーダ + SMP の UnetDecoder を使った柔軟な U-Net 実装例。
    """
    def __init__(
        self,
        backbone="tf_efficientnet_b7.ns_jft_in1k",
        pretrained=True,
        in_chans=3,
        out_channels=10,
    ):
        """
        Args:
            backbone (str): TIMM で利用可能なモデル名
            pretrained (bool): エンコーダを学習済みにするかどうか
            in_chans (int): 入力画像のチャンネル数
            out_channels (int): 出力チャネル数 (セグメンテーションのクラス数など)
        """
        super().__init__()
        
        self.encoder, encoder_chs = get_encoder(
            backbone=backbone,
            pretrained=pretrained,
            in_chans=in_chans,
        )
        self.decoder_channels = [256, 128, 64, 32, 16]
        n_blocks = 4
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_chs[:n_blocks+1],
            decoder_channels=self.decoder_channels[:n_blocks],
            n_blocks=n_blocks,
            use_batchnorm=True,
            attention_type="scse"
        )
        self.segmentation_head = nn.Conv2d(
            self.decoder_channels[n_blocks],
            out_channels,
            kernel_size=1
        )
        fet = self.encoder(torch.randn(2, 3, 512, 512))
        dec = self.decoder(*fet)
        seg = self.segmentation_head(dec)
        if seg.shape[2:] != (512, 512):
            scaler_factor = 512 // seg.shape[2]
            self.segmentation_head = nn.Sequential(
                nn.Upsample(scale_factor=scaler_factor, mode='bilinear'),
                nn.Conv2d(self.decoder_channels[n_blocks], self.decoder_channels[n_blocks]//2, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.decoder_channels[n_blocks]//2, out_channels, kernel_size=1)
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, in_chans, H, W)

        Returns:
            torch.Tensor: (B, out_channels, H, W)
        """
        # (a) エンコーダで特徴マップを抽出
        features = [0] + self.encoder(x)
        # (b) デコーダでアップサンプリング + Skip Connection
        decoder_output = self.decoder(*features)
        # (c) 最後に Conv1x1 でクラス数 (out_channels) に変換
        logits = self.segmentation_head(decoder_output)
        return logits


class TimmSegModel_v2(nn.Module):
    def __init__(self, backbone, segtype='unet', pretrained=False, attention_type = "scse", out_dim=10, in_chans=3, reshape=True):#att_type: scse
        super(TimmSegModel_v2, self).__init__()

        self.encoder, encoder_chs = get_encoder(
            backbone=backbone,
            pretrained=pretrained,
            in_chans=in_chans,
        )
        self.decoder_channels = [256, 128, 64, 32, 16]
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
            self.decoder = smp.decoders.deeplabv3.decoder.DeepLabV3Decoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                decoder_channels=self.decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )
        elif segtype == "segformer":
            self.decoder = smp.decoders.segformer.decoder.SegFormerDecoder(
                encoder_channels=encoder_chs[:n_blocks+1],
                decoder_channels=self.decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )
            '''elif segtype == 'link':
                self.decoder = smp.decoders.linknet.decoder.LinknetDecoder(
                    encoder_channels=encoder_chs[:n_blocks+1],
                    prefinal_channels=self.decoder_channels[n_blocks-1],
                    n_blocks=n_blocks,
                )'''
            '''elif segtype == 'fpn':
                self.decoder = smp.decoders.fpn.decoder.FPNDecoder(
                    encoder_channels=encoder_chs[:n_blocks+1],
                    segmentation_channels=self.decoder_channels[n_blocks-1],
                )
                self.segmentation_head = nn.Sequential(
                    nn.Conv2d(self.decoder_channels[n_blocks-1], 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(16, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                )'''
            '''elif segtype == 'pan':
                self.decoder= smp.decoders.pan.decoder.PANDecoder(
                    encoder_channels=encoder_chs[:n_blocks+1],
                    decoder_channels=32,
                )
                self.segmentation_head = nn.Sequential(
                    nn.Conv2d(self.decoder_channels[n_blocks-1], 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(16, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                )'''
        fet = self.encoder(torch.randn(2, 3, 512, 512))
        dec = self.decoder(*fet)
        seg = self.segmentation_head(dec)
        if seg.shape[2:] != (512, 512):
            scaler_factor = 512 // seg.shape[2]
            self.segmentation_head = nn.Sequential(
                nn.Upsample(scale_factor=scaler_factor, mode='bilinear'),
                nn.Conv2d(self.decoder_channels[n_blocks], self.decoder_channels[n_blocks]//2, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.decoder_channels[n_blocks]//2, out_dim, kernel_size=1)
            )
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self,x):
        global_features = [0] + self.encoder(x)
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features