import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import time
import copy
#from sam2.build_sam import build_sam2_video_predictor
class VideoSegModel(nn.Module):
    def __init__(self, encoder, decoder, seg_head, device="cuda", sam2_predictor=None, batch_size=1, index = [i for i in range(60)], image_size = (1024, 1024)):
        super(VideoSegModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seg_head = seg_head
        # メモリインデックスを指定
        self.index = index
        self.memory_length = max(self.index) + 1
        
        self.batch_size = batch_size
        self.frame_count = 0  # フレームカウンターを追加
        
        # sam2_predictor まわりは同じ
        predictor = sam2_predictor.to(device)
        self.memory_encoder = predictor.memory_encoder
        self.memory_attention = predictor.memory_attention
        self.norm = nn.InstanceNorm1d(num_features=256)

        self.is_first_frame = torch.Tensor([True]*self.batch_size).to(device)

        # 初回 forward 時に使う初期メモリを inference_mode 下で作成
        with torch.inference_mode():
            # テンソル形状は元コードに合わせて適宜
            features = self.encoder(torch.zeros(batch_size, 3, image_size[0], image_size[1], device=device))[3]
            n_inference = features.shape[1] // 256
            # print(features.shape)  # デバッグ出力をコメントアウト
            if n_inference == 0:
                n_inference = 1
                features = torch.cat([features, features[:, 256-features.shape[1]:]], dim=1)
            curr = features
            # print(curr.shape)  # デバッグ出力をコメントアウト
            curr = F.interpolate(curr, size=(64, 64), mode='bilinear', align_corners=False)
            curr = curr[:, :256*n_inference, :, :].view(n_inference, 256, 64, 64)
            k = self.memory_encoder(
                curr, # shape: (B=n_inference, C=256, H=64, W=64)
                torch.zeros(n_inference, 1, 16, 16, device=device)
            )
            v_feat = k["vision_features"].flatten(2).permute(0, 2, 1)
            v_pos = k["vision_pos_enc"][0].flatten(2).permute(0, 2, 1)
            # vision_features, vision_pos_enc を flatten/transpose して結合
            # shape: (1, 64, 64, 64) 等の場合は flatten(2)->(1,64,4096) -> permute(0,2,1)->(1,4096,64) のように。
            # (B, HW, C)
            self.init_memory = torch.stack([v_feat for _ in range(self.memory_length)], dim=0) # (memory_length, n_inference, 4096, 64)
            self.init_memory_pos = torch.stack([v_pos for _ in range(self.memory_length)], dim=0) # (memory_length, n_inference, 4096, 64)

            # 画像エンコーダから最終の position encoding も取り出しておく
            self.curr_enc = predictor.image_encoder(
                torch.randn(1, 3, 1024, 1024, device=device)
            )["vision_pos_enc"][-1]
            self.curr_enc = self.curr_enc.view(1, 256, 64*64).permute(2,0,1)
            #self.curr_enc = torch.stack([self.curr_enc]*n_inference, dim=0)
            # 実メモリとして登録（学習しないので buffer）
            self.register_buffer('memory', self.init_memory)       # (1, 60*4096, 64) の想定 
            self.register_buffer('memory_pos', self.init_memory_pos)
        self.freeze_sam2()

    def init_weights(self, video_start):
        """再初期化用, video_startは初期化するbatchがTrueになってるtensor"""
        if video_start.any():
            # (memory_length, n_inference, 4096, 64)
            self.memory = self.init_memory
            # (memory_length, n_inference, 4096, 64)
            self.memory_pos = self.init_memory_pos
        self.is_first_frame.fill_(True)
        self.frame_count = 0

    #@torch.inference_mode()
    def _memory_attention(self, curr):
        """
        curr[-1] が最終特徴 (B, C, H, W)
        64x64 にリサイズ → flatten → transpose で (seq_len, B, embed_dim) に整形して self.memory_attention()。
        """

        #curr = copy.deepcopy(original_curr)
        in_curr = curr[-1]
        incurr_size = in_curr.shape[2:]
        in_curr = F.interpolate(in_curr, size=(64, 64), mode='bilinear', align_corners=False)

        # 例えば C=256 でなければパディングするロジック
        n0_flag = False
        C_in = in_curr.shape[1]
        n_inference = C_in // 256
        if n_inference == 0:
            n_inference = 1
            in_curr = torch.cat([in_curr, in_curr[:, :256-C_in]], dim=1)
            n0_flag = True
        elif C_in % 256 != 0:
            in_curr, residue = torch.split(in_curr, [256*n_inference, C_in-256*n_inference], dim=1)
            in_curr = in_curr.view(n_inference, 256, 64, 64)
        else:
            in_curr = in_curr.view(n_inference, 256, 64, 64)
        #print(in_curr.shape)

        if self.frame_count >= 1:  # 2フレーム目以降
            in_curr = in_curr.view(n_inference, 256, 64*64).permute(2, 0, 1)
            selected_indices = []
            
            # self.indexで指定されたフレーム数前のメモリを取得
            for idx in self.index:
                if self.frame_count >= idx:  # 参照可能なフレームのみ
                    selected_indices.append(idx)
            
             # Memory : shape: memory_length, n_inference, 4096, 64
            if selected_indices:
                # 選択されたインデックスのメモリを結合（既に時系列順）
                memory_flat = torch.cat([
                    self.memory[idx_slice].permute(1,0,2)
                    for idx_slice in selected_indices
                ], dim=0)
                memory_pos_flat = torch.cat([
                    self.memory_pos[idx_slice].permute(1,0,2)
                    for idx_slice in selected_indices
                ], dim=0)

            else:
                # 初期フレームの場合は現在のメモリをそのまま使用
                memory_flat = self.memory#.view(4096*self.memory_length, 1, 64)
                memory_pos_flat = self.memory_pos#.view(4096*self.memory_length, 1, 64)
                # posは1batchでok
            #memory_flat = torch.stack([memory_flat]*n_inference, dim=0) # batch数合わせる
            #print("self.memory.shape, self.memory_pos.shape", self.memory.shape, self.memory_pos.shape)
            #print("in_curr.shape, curr_pos.shape, memory_flat.shape, memory_pos_flat.shape", in_curr.shape, self.curr_enc.shape, memory_flat.shape, memory_pos_flat.shape)
            in_curr = self.memory_attention(
                curr=in_curr, # 4096, n_inference, 256 Ok
                memory=memory_flat, # 2, 11
                curr_pos=self.curr_enc, # 4096, 1, 256 ok
                memory_pos=memory_pos_flat # 2, 11, 4096, 64 not ok
            )
                # normalize
                #curr_mean = _curr_flat.mean(dim=0, keepdim=True)
                #curr_std = _curr_flat.std(dim=0, keepdim=True)
                #_tmp_curr = (_tmp_curr - _tmp_curr.mean(dim=0, keepdim=True)) / (_tmp_curr.std(dim=0, keepdim=True) + 1e-6)
                #_tmp_curr = _tmp_curr * curr_std + curr_mean
            # ============ 元の空間形状へ戻す ============
            # 現状: tmp_curr は (4096, n_inferece, 256) 等 → 軸を入替えて (n_infrecem, 256, 4096) → view で (B, dim, H, W)
            #print(in_curr.shape) # 4096 11 256
            in_curr = in_curr.permute(1, 2, 0).reshape(n_inference, 256, 64, 64)#.reshape(1, -1, 64, 64)     # 4096, n_inference, 256 -> (n_inference, 256, 64, 64)
            if n0_flag:
                out_curr = in_curr
                curr[-1] = F.interpolate(in_curr[:, :C_in], size=incurr_size, mode='bilinear', align_corners=False)

            elif C_in % 256 != 0:
                #out_curr = torch.cat([in_curr, residue], dim=1)
                out_curr = in_curr
                curr[-1] = F.interpolate(torch.cat([in_curr.reshape(1, -1, 64, 64), residue], dim=1), size=incurr_size, mode='bilinear', align_corners=False)
            else:
                out_curr = in_curr
                curr[-1] = F.interpolate(out_curr.reshape(1, -1, 64, 64), size=incurr_size, mode='bilinear', align_corners=False)
        else:
            out_curr = in_curr
        #print(out_curr.shape)
        return curr, out_curr # feature_map, feature_map[-1]: shape: n_inference, 256,64,64

    @torch.inference_mode()
    def _memory_bank_update(self, pix_feat, mask):
        """
        - mask: softmax後の1チャンネルを反転(1 - mask) 6→ 下流メソッドへ
        - メモリ (self.memory) に新しいフレームの埋め込みをキューの要領で追加
        """
        
        mask = mask.softmax(dim=1)[:, 0:1]
        mask = F.interpolate(1 - mask, size=(16, 16), mode='bilinear')
        n_inference = pix_feat.shape[1]
        #pix_feat = pix_feat.permute(1,2,0).view(n_inference, 256, 64, 64)
        #print("pix_feat.shape, mask.shape", pix_feat.shape, mask.shape)
        k = self.memory_encoder(pix_feat, mask)
        v_feat = k["vision_features"].flatten(2).permute(0, 2, 1) # n_inference, 4096, 64
        v_pos = k["vision_pos_enc"][0].flatten(2).permute(0, 2, 1)
        
        #print(self.memory.shape, v_feat.shape)
        # (memory_length, n_inference, 4096, 64)
        if self.is_first_frame.any():
            # 初期フレームの場合は全スロットに同じ特徴をコピー
            for i in range(len(self.memory)):
                #torch.Size([10, 11, 4096, 64]) torch.Size([10, 11, 4096, 64])
                self.memory[i, :, :, :] = v_feat
                self.memory_pos[i, :, :, :] = v_pos
            self.is_first_frame.fill_(False)
        else:
            # 既存のメモリを1スロット分後ろにシフト
            
            self.memory[1:, :] = self.memory[:-1, :].clone()
            self.memory_pos[1:,:] = self.memory_pos[-1:, :].clone()
            
            # 最新のフレームを先頭に配置
            self.memory[0] = v_feat
            self.memory_pos[0] = v_pos
            
        self.frame_count += 1

    def freeze_sam2(self):
        for param in self.memory_attention.parameters():
            param.requires_grad = False
        for param in self.memory_encoder.parameters():
            param.requires_grad = False

    def forward(self, img):
        """
        - encoder: List of feature maps [f1, f2, ..., f4]
        - curr[-1] が最終特徴と想定
        - decoder + seg_head でマスク生成 → memory_bank_update
        """
        # 例: pix_feat = [0] + self.encoder(img)[:4]
        pix_feat = [0] + self.encoder(img)[:4]
        
        if self.frame_count != 0:
        # Attention で特徴を更新
            pix_feat, curr = self._memory_attention(pix_feat)
        else:
            _, curr = self._memory_attention(pix_feat)

        # デコーダでセグマスク
        mask = self.decoder(*pix_feat)
        mask = self.seg_head(mask)

        # 更新した特徴をメモリに登録
        self._memory_bank_update(curr, mask)
        return mask

def adapt_input_conv(new_channels, weight):
    """
    入力チャネル数を変更するために重みを調整
    Args:
        in_channels (int): 新しい入力チャネル数
        weight (torch.Tensor): 元の重み (shape: [out_channels, in_channels, kernel_h, kernel_w])
    
    Returns:
        torch.Tensor: 調整された重み
    """
    out_channels, old_in_channels, kernel_h, kernel_w = weight.shape
    if new_channels == old_in_channels:
        return weight
    if new_channels < old_in_channels:
        # チャネル数を減らす場合はスライス
        return weight[:, :new_channels, :, :]
    else:
        # チャネル数を増やす場合はパディング
        # 新しいチャネルは既存のチャネルの重みを繰り返して初期化
        new_weight = torch.zeros((out_channels, new_channels, kernel_h, kernel_w), 
                               dtype=weight.dtype, device=weight.device)
        new_weight[:, :old_in_channels, :, :] = weight
        # 残りのチャネルを既存の重みで繰り返し初期化
        repeat_times = (new_channels - old_in_channels + old_in_channels - 1) // old_in_channels
        repeated_weights = weight.repeat(1, repeat_times + 1, 1, 1)
        new_weight[:, old_in_channels:, :, :] = repeated_weights[:, old_in_channels:new_channels, :, :]
        return new_weight

def adapt_conv(old_conv, new_channel):
    """
    入力チャネル数を変更するために Conv2d レイヤーを調整
    Args:
        conv (nn.Conv2d): 元の Conv2d レイヤー
        new_channels (int): 新しい入力チャネル数
    
    Returns:
        nn.Conv2d: 調整された Conv2d レイヤー
    """
    # 新しい Conv2d レイヤーを作成
    new_conv = nn.Conv2d(
                new_channel, 
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
    # 重みを設定
    new_conv.weight.data = adapt_input_conv(new_channel, old_conv.weight)
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias
    return new_conv

class TMAM(nn.Module):
    def __init__(self, encoder, decoder, seg_head, device="cuda", sam2_predictor=None, batch_size=1, index = [i for i in range(60)], depth=5,  image_size = (1024, 1024)):
        super(TMAM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.depth = depth
        self.seg_head = seg_head
        # メモリインデックスを指定
        self.index = index
        self.memory_length = max(self.index) + 1
        
        self.batch_size = batch_size
        self.frame_count = 0  # フレームカウンターを追加
        
        # sam2_predictor まわりは同じ
        predictor = sam2_predictor.to(device)
        self.memory_encoder = predictor.memory_encoder
        self.memory_attention = predictor.memory_attention
        self.norm = nn.InstanceNorm1d(num_features=256)

        self.is_first_frame = torch.Tensor([True]*self.batch_size).to(device)

        # 初回 forward 時に使う初期メモリを inference_mode 下で作成
        with torch.inference_mode():
            # テンソル形状は元コードに合わせて適宜
            features = self.encoder(torch.zeros(batch_size, 3, image_size[0], image_size[1], device=device))[self.depth]
            n_inference = features.shape[1] // 256
            # print(features.shape)  # デバッグ出力をコメントアウト
            if n_inference == 0:
                n_inference = 1
                features = torch.cat([features, features[:, 256-features.shape[1]:]], dim=1)
            curr = features
            # print(curr.shape)  # デバッグ出力をコメントアウト
            curr = F.interpolate(curr, size=(64, 64), mode='bilinear', align_corners=False)
            curr = curr[:, :256*n_inference, :, :].view(n_inference, 256, 64, 64)
            k = self.memory_encoder(
                curr, # shape: (B=n_inference, C=256, H=64, W=64)
                torch.zeros(n_inference, 1, 16, 16, device=device)
            )
            v_feat = k["vision_features"].flatten(2).permute(0, 2, 1)
            v_pos = k["vision_pos_enc"][0].flatten(2).permute(0, 2, 1)
            # vision_features, vision_pos_enc を flatten/transpose して結合
            # shape: (1, 64, 64, 64) 等の場合は flatten(2)->(1,64,4096) -> permute(0,2,1)->(1,4096,64) のように。
            # (B, HW, C)
            self.init_memory = torch.stack([v_feat for _ in range(self.memory_length)], dim=0) # (memory_length, n_inference, 4096, 64)
            self.init_memory_pos = torch.stack([v_pos for _ in range(self.memory_length)], dim=0) # (memory_length, n_inference, 4096, 64)

            # 画像エンコーダから最終の position encoding も取り出しておく
            self.curr_enc = predictor.image_encoder(
                torch.randn(1, 3, 1024, 1024, device=device)
            )["vision_pos_enc"][-1]
            self.curr_enc = self.curr_enc.view(1, 256, 64*64).permute(2,0,1)
            #self.curr_enc = torch.stack([self.curr_enc]*n_inference, dim=0)
            # 実メモリとして登録（学習しないので buffer）
            self.register_buffer('memory', self.init_memory)       # (1, 60*4096, 64) の想定 
            self.register_buffer('memory_pos', self.init_memory_pos)
        self.freeze_sam2()

    def init_weights(self, video_start):
        """再初期化用, video_startは初期化するbatchがTrueになってるtensor"""
        if video_start.any():
            # (memory_length, n_inference, 4096, 64)
            self.memory = self.init_memory
            # (memory_length, n_inference, 4096, 64)
            self.memory_pos = self.init_memory_pos
        self.is_first_frame.fill_(True)
        self.frame_count = 0

    #@torch.inference_mode()
    def _memory_attention(self, curr):
        """
        curr[-1] が最終特徴 (B, C, H, W)
        64x64 にリサイズ → flatten → transpose で (seq_len, B, embed_dim) に整形して self.memory_attention()。
        """

        #curr = copy.deepcopy(original_curr)
        in_curr = curr[-1]
        incurr_size = in_curr.shape[2:]
        in_curr = F.interpolate(in_curr, size=(64, 64), mode='bilinear', align_corners=False)

        # 例えば C=256 でなければパディングするロジック
        n0_flag = False
        C_in = in_curr.shape[1]
        n_inference = C_in // 256
        if n_inference == 0:
            n_inference = 1
            in_curr = torch.cat([in_curr, in_curr[:, :256-C_in]], dim=1)
            n0_flag = True
        elif C_in % 256 != 0:
            in_curr, residue = torch.split(in_curr, [256*n_inference, C_in-256*n_inference], dim=1)
            in_curr = in_curr.view(n_inference, 256, 64, 64)
        else:
            in_curr = in_curr.view(n_inference, 256, 64, 64)
        #print(in_curr.shape)

        if self.frame_count >= 1:  # 2フレーム目以降
            in_curr = in_curr.view(n_inference, 256, 64*64).permute(2, 0, 1)
            selected_indices = []
            
            # self.indexで指定されたフレーム数前のメモリを取得
            for idx in self.index:
                if self.frame_count >= idx:  # 参照可能なフレームのみ
                    selected_indices.append(idx)
            
             # Memory : shape: memory_length, n_inference, 4096, 64
            if selected_indices:
                # 選択されたインデックスのメモリを結合（既に時系列順）
                memory_flat = torch.cat([
                    self.memory[idx_slice].permute(1,0,2)
                    for idx_slice in selected_indices
                ], dim=0)
                memory_pos_flat = torch.cat([
                    self.memory_pos[idx_slice].permute(1,0,2)
                    for idx_slice in selected_indices
                ], dim=0)

            else:
                # 初期フレームの場合は現在のメモリをそのまま使用
                memory_flat = self.memory#.view(4096*self.memory_length, 1, 64)
                memory_pos_flat = self.memory_pos#.view(4096*self.memory_length, 1, 64)
                # posは1batchでok
            #memory_flat = torch.stack([memory_flat]*n_inference, dim=0) # batch数合わせる
            #print("self.memory.shape, self.memory_pos.shape", self.memory.shape, self.memory_pos.shape)
            #print("in_curr.shape, curr_pos.shape, memory_flat.shape, memory_pos_flat.shape", in_curr.shape, self.curr_enc.shape, memory_flat.shape, memory_pos_flat.shape)
            in_curr = self.memory_attention(
                curr=in_curr, # 4096, n_inference, 256 Ok
                memory=memory_flat, # 2, 11
                curr_pos=self.curr_enc, # 4096, 1, 256 ok
                memory_pos=memory_pos_flat # 2, 11, 4096, 64 not ok
            )
                # normalize
                #curr_mean = _curr_flat.mean(dim=0, keepdim=True)
                #curr_std = _curr_flat.std(dim=0, keepdim=True)
                #_tmp_curr = (_tmp_curr - _tmp_curr.mean(dim=0, keepdim=True)) / (_tmp_curr.std(dim=0, keepdim=True) + 1e-6)
                #_tmp_curr = _tmp_curr * curr_std + curr_mean
            # ============ 元の空間形状へ戻す ============
            # 現状: tmp_curr は (4096, n_inferece, 256) 等 → 軸を入替えて (n_infrecem, 256, 4096) → view で (B, dim, H, W)
            #print(in_curr.shape) # 4096 11 256
            in_curr = in_curr.permute(1, 2, 0).reshape(n_inference, 256, 64, 64)#.reshape(1, -1, 64, 64)     # 4096, n_inference, 256 -> (n_inference, 256, 64, 64)
            if n0_flag:
                out_curr = in_curr
                curr[-1] = F.interpolate(in_curr[:, :C_in], size=incurr_size, mode='bilinear', align_corners=False)

            elif C_in % 256 != 0:
                #out_curr = torch.cat([in_curr, residue], dim=1)
                out_curr = in_curr
                curr[-1] = F.interpolate(torch.cat([in_curr.reshape(1, -1, 64, 64), residue], dim=1), size=incurr_size, mode='bilinear', align_corners=False)
            else:
                out_curr = in_curr
                curr[-1] = F.interpolate(out_curr.reshape(1, -1, 64, 64), size=incurr_size, mode='bilinear', align_corners=False)
        else:
            out_curr = in_curr
        #print(out_curr.shape)
        return curr, out_curr # feature_map, feature_map[-1]: shape: n_inference, 256,64,64

    @torch.inference_mode()
    def _memory_bank_update(self, pix_feat, mask):
        """
        - mask: softmax後の1チャンネルを反転(1 - mask) 6→ 下流メソッドへ
        - メモリ (self.memory) に新しいフレームの埋め込みをキューの要領で追加
        """
        
        mask = mask.softmax(dim=1)[:, 0:1]
        mask = F.interpolate(1 - mask, size=(16, 16), mode='bilinear')
        n_inference = pix_feat.shape[1]
        #pix_feat = pix_feat.permute(1,2,0).view(n_inference, 256, 64, 64)
        #print("pix_feat.shape, mask.shape", pix_feat.shape, mask.shape)
        k = self.memory_encoder(pix_feat, mask)
        v_feat = k["vision_features"].flatten(2).permute(0, 2, 1) # n_inference, 4096, 64
        v_pos = k["vision_pos_enc"][0].flatten(2).permute(0, 2, 1)
        
        #print(self.memory.shape, v_feat.shape)
        # (memory_length, n_inference, 4096, 64)
        if self.is_first_frame.any():
            # 初期フレームの場合は全スロットに同じ特徴をコピー
            for i in range(len(self.memory)):
                #torch.Size([10, 11, 4096, 64]) torch.Size([10, 11, 4096, 64])
                self.memory[i, :, :, :] = v_feat
                self.memory_pos[i, :, :, :] = v_pos
            self.is_first_frame.fill_(False)
        else:
            # 既存のメモリを1スロット分後ろにシフト
            
            self.memory[1:, :] = self.memory[:-1, :].clone()
            self.memory_pos[1:,:] = self.memory_pos[-1:, :].clone()
            
            # 最新のフレームを先頭に配置
            self.memory[0] = v_feat
            self.memory_pos[0] = v_pos
            
        self.frame_count += 1
        torch.cuda.empty_cache()

    def freeze_sam2(self):
        for param in self.memory_attention.parameters():
            param.requires_grad = False
        for param in self.memory_encoder.parameters():
            param.requires_grad = False

    def forward(self, img):
        """
        - encoder: List of feature maps [f1, f2, ..., f4]
        - curr[-1] が最終特徴と想定
        - decoder + seg_head でマスク生成 → memory_bank_update
        """
        # 例: pix_feat = [0] + self.encoder(img)[:4]
        if self.depth == -1:
            pix_feat = self.encoder(img)
        else:
            pix_feat = self.encoder(img)[:self.depth]
        # print([_.shape for _ in pix_feat])  # デバッグ出力をコメントアウト
        if self.frame_count != 0:
        # Attention で特徴を更新
            pix_feat, curr = self._memory_attention(pix_feat)
        else:
            _, curr = self._memory_attention(pix_feat)

        # デコーダでセグマスク
        # print([_.shape for _ in pix_feat])  # デバッグ出力をコメントアウト
        mask = self.decoder(*pix_feat)
        mask = self.seg_head(mask)

        # 更新した特徴をメモリに登録
        self._memory_bank_update(curr, mask)
        return mask
