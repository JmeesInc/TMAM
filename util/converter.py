import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import copy
#from sam2.build_sam import build_sam2_video_predictor
class VideoSegModel(nn.Module):
    def __init__(self, encoder, decoder, seg_head, device="cuda", sam2_predictor=None, batch_size=1, index = [i for i in range(60)]):
        super(VideoSegModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seg_head = seg_head
        # メモリインデックスを指定
        self.index = index
        self.memory_length = max(self.index)+1 
        
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
            a = self.memory_encoder(
                torch.zeros(batch_size, 256, 64, 64, device=device),
                torch.zeros(batch_size, 1, 16, 16, device=device)
            )
            # vision_features, vision_pos_enc を flatten/transpose して結合
            # shape: (1, 64, 64, 64) 等の場合は flatten(2)->(1,64,4096) -> permute(0,2,1)->(1,4096,64) のように。
            # 以下は元コードのままという想定:
            self.init_memory = torch.cat(
                [a["vision_features"].flatten(2).permute(0,2,1) for _ in range(self.memory_length)], dim=1
            )
            self.init_memory_pos = torch.cat(
                [a["vision_pos_enc"][0].flatten(2).permute(0,2,1) for _ in range(self.memory_length)], dim=1
            )

            # 画像エンコーダから最終の position encoding も取り出しておく
            self.curr_enc = predictor.image_encoder(
                torch.randn(batch_size, 3, 1024, 1024, device=device)
            )["vision_pos_enc"][-1]

            # 実メモリとして登録（学習しないので buffer）
            self.register_buffer('memory', self.init_memory)       # (1, 60*4096, 64) の想定 
            self.register_buffer('memory_pos', self.init_memory_pos)
        self.freeze_sam2()

    def init_weights(self, video_start):
        """再初期化用, video_startは初期化するbatchがTrueになってるtensor"""
        self.memory = torch.where(
            video_start.unsqueeze(-1).unsqueeze(-1),
            self.init_memory,
            self.memory
        )
        self.memory_pos = torch.where(
            video_start.unsqueeze(-1).unsqueeze(-1), 
            self.init_memory_pos,
            self.memory_pos
        )
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
        in_curr = F.interpolate(in_curr, size=(64, 64), mode='bilinear', align_corners=False)

        # 例えば C=256 でなければパディングするロジック
        C_in = in_curr.shape[1]
        n_inference = C_in // 256
        if n_inference == 0:
            pad_size = 256 - C_in
            in_curr = torch.cat(
                [in_curr, in_curr[:,0:pad_size,:,:]], # to do inCurr recursive
                dim=1
            )
            n_inference = 1

        # ============ 変換: flatten + transpose ============
        # shape: (B, C, H, W) -> (B, C, H*W) -> (H*W, B, C)
        B, C, H, W = in_curr.shape
        curr_flat = in_curr.view(B, C, H*W).permute(2, 0, 1)  # (4096, B=1, C=256など)

        # 同様に pos_enc も flatten + transpose
        # shape: self.curr_enc も (B, C, H, W) と想定
        Bp, Cp, Hp, Wp = self.curr_enc.shape
        curr_pos = self.curr_enc.view(Bp, Cp, Hp*Wp).permute(2, 0, 1)

        #####ここが間違えている
        if self.frame_count >= 1:  # 2フレーム目以降
            selected_indices = []
            
            # self.indexで指定されたフレーム数前のメモリを取得
            for idx in self.index:
                if self.frame_count >= idx:  # 参照可能なフレームのみ
                    selected_indices.append(slice(idx * 4096, (idx + 1) * 4096))
            
            if selected_indices:
                # 選択されたインデックスのメモリを結合（既に時系列順）
                memory_flat = torch.cat([
                    self.memory.transpose(0, 1)[idx_slice] 
                    for idx_slice in selected_indices
                ], dim=0)
                memory_pos_flat = torch.cat([
                    self.memory_pos.transpose(0, 1)[idx_slice]
                    for idx_slice in selected_indices
                ], dim=0)
            else:
                # 初期フレームの場合は現在のメモリをそのまま使用
                memory_flat = self.memory.transpose(0, 1)
                memory_pos_flat = self.memory_pos.transpose(0, 1)
            tmp_curr = []
            for i in range(n_inference):
                _curr_flat = curr_flat[i*4096:(i+1)*4096]
                _curr_pos = curr_pos[i*4096:(i+1)*4096]
                _tmp_curr = self.memory_attention(
                    curr=_curr_flat,
                    memory=memory_flat,
                    curr_pos=_curr_pos,
                    memory_pos=memory_pos_flat
                )
                # normalize
                #curr_mean = _curr_flat.mean(dim=0, keepdim=True)
                #curr_std = _curr_flat.std(dim=0, keepdim=True)
                #_tmp_curr = (_tmp_curr - _tmp_curr.mean(dim=0, keepdim=True)) / (_tmp_curr.std(dim=0, keepdim=True) + 1e-6)
                #_tmp_curr = _tmp_curr * curr_std + curr_mean
                tmp_curr.append(_tmp_curr)
            # ============ 元の空間形状へ戻す ============
            # 現状: tmp_curr は (4096, 1, 64) 等 → 軸を入替えて (B, dim, seq_len) → view で (B, dim, H, W)
            tmp_curr = torch.cat(tmp_curr, dim=0)
            tmp_curr_bdc = tmp_curr.transpose(0, 1).transpose(1, 2)  # (B, dim, seq_len)
            in_curr = tmp_curr_bdc.view(B, -1, H, W)                 # (B, ?, 64, 64)

            # パディング分戻す（C_in != 256 の時）
            if n_inference == 1 and C_in != 256:
                in_curr = in_curr[:, :C_in]

            # 元の解像度に戻す
            curr[-1] = F.interpolate(
                in_curr, size=curr[-1].shape[-2:], mode='bilinear', align_corners=False
            )

            # 返り値として必要な (1, 256, 64, 64) 相当を作るなら以下
            # tmp_curr_bdc は (B, dim=64, seq_len=4096) → 例えば 64*64=4096 ならチャンネル256と対応が合わない？ 
            # 元コード同様に "256,64,64" に再変換するなら:
            out_curr = tmp_curr_bdc.view(B, -1, 64, 64)[:, :256, :, :]  # 必要に応じて取り出し
        else:
            out_curr = in_curr.view(B, -1, 64, 64)[:, :256, :, :]
            
        return curr, out_curr

    @torch.inference_mode()
    def _memory_bank_update(self, pix_feat, mask):
        """
        - mask: softmax後の1チャンネルを反転(1 - mask) → 下流メソッドへ
        - メモリ (self.memory) に新しいフレームの埋め込みをキューの要領で追加
        """
        mask = mask.softmax(dim=1)[:, 0:1]
        mask = F.interpolate(1 - mask, size=(16, 16), mode='bilinear')

        k = self.memory_encoder(pix_feat, mask)
        v_feat = k["vision_features"].flatten(2).permute(0, 2, 1)
        v_pos = k["vision_pos_enc"][0].flatten(2).permute(0, 2, 1)

        if self.is_first_frame.any():
            # 初期フレームの場合は全スロットに同じ特徴をコピー
            for i in range(self.memory_length):
                self.memory[:, i*4096:(i+1)*4096, :] = v_feat
                self.memory_pos[:, i*4096:(i+1)*4096, :] = v_pos
            self.is_first_frame.fill_(False)
        else:
            # 既存のメモリを1スロット分後ろにシフト
            self.memory[:, 4096:] = self.memory[:, :-4096].clone()
            self.memory_pos[:, 4096:] = self.memory_pos[:, :-4096].clone()
            
            # 最新のフレームを先頭に配置
            self.memory[:, :4096] = v_feat
            self.memory_pos[:, :4096] = v_pos
            
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