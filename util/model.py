import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import time
import copy

class TMAM(nn.Module):
    def __init__(self, encoder, decoder, seg_head, device="cuda", sam2_predictor=None, batch_size=1, index = [i for i in range(60)], depth=5,  image_size = (1024, 1024)):
        super(TMAM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.depth = depth
        self.seg_head = seg_head
        # Specify memory indices
        self.index = index
        self.memory_length = max(self.index) + 1
        
        self.batch_size = batch_size
        self.frame_count = 0  # Add frame counter
        
        # sam2_predictor related setup
        predictor = sam2_predictor.to(device)
        self.memory_encoder = predictor.memory_encoder
        self.memory_attention = predictor.memory_attention
        self.norm = nn.InstanceNorm1d(num_features=256)

        self.is_first_frame = torch.Tensor([True]*self.batch_size).to(device)

        # Create initial memory for first forward pass under inference_mode
        with torch.inference_mode():
            # Tensor shapes adjusted according to original code
            features = self.encoder(torch.zeros(batch_size, 3, image_size[0], image_size[1], device=device))[self.depth]
            n_inference = features.shape[1] // 256
            # print(features.shape)  # Comment out debug output
            if n_inference == 0:
                n_inference = 1
                features = torch.cat([features, features[:, 256-features.shape[1]:]], dim=1)
            curr = features
            # print(curr.shape)  # Comment out debug output
            curr = F.interpolate(curr, size=(64, 64), mode='bilinear', align_corners=False)
            curr = curr[:, :256*n_inference, :, :].view(n_inference, 256, 64, 64)
            k = self.memory_encoder(
                curr, # shape: (B=n_inference, C=256, H=64, W=64)
                torch.zeros(n_inference, 1, 16, 16, device=device)
            )
            v_feat = k["vision_features"].flatten(2).permute(0, 2, 1)
            v_pos = k["vision_pos_enc"][0].flatten(2).permute(0, 2, 1)
            # Flatten/transpose vision_features, vision_pos_enc and combine
            # For shape like (1, 64, 64, 64): flatten(2)->(1,64,4096) -> permute(0,2,1)->(1,4096,64)
            # (B, HW, C)
            self.init_memory = torch.stack([v_feat for _ in range(self.memory_length)], dim=0) # (memory_length, n_inference, 4096, 64)
            self.init_memory_pos = torch.stack([v_pos for _ in range(self.memory_length)], dim=0) # (memory_length, n_inference, 4096, 64)

            # Extract final position encoding from image encoder
            self.curr_enc = predictor.image_encoder(
                torch.randn(1, 3, 1024, 1024, device=device)
            )["vision_pos_enc"][-1]
            self.curr_enc = self.curr_enc.view(1, 256, 64*64).permute(2,0,1)
            #self.curr_enc = torch.stack([self.curr_enc]*n_inference, dim=0)
            # Register as actual memory (not learned, so buffer)
            self.register_buffer('memory', self.init_memory)       # Expected shape: (1, 60*4096, 64) 
            self.register_buffer('memory_pos', self.init_memory_pos)
        self.freeze_sam2()

    def init_weights(self, video_start):
        """Reinitialization method, video_start is tensor with True for batches to initialize"""
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
        curr[-1] is final feature (B, C, H, W)
        Resize to 64x64 → flatten → transpose to (seq_len, B, embed_dim) format for self.memory_attention().
        """

        #curr = copy.deepcopy(original_curr)
        in_curr = curr[self.depth]
        incurr_size = in_curr.shape[2:]
        in_curr = F.interpolate(in_curr, size=(64, 64), mode='bilinear', align_corners=False)

        # Padding logic if C!=256
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

        if self.frame_count >= 1:  # From 2nd frame onwards
            in_curr = in_curr.view(n_inference, 256, 64*64).permute(2, 0, 1)
            selected_indices = []
            
            # Get memory from frames specified by self.index
            for idx in self.index:
                if self.frame_count >= idx:  # Only referenceable frames
                    selected_indices.append(idx)
            
             # Memory : shape: memory_length, n_inference, 4096, 64
            if selected_indices:
                # Combine memory from selected indices (already in chronological order)
                memory_flat = torch.cat([
                    self.memory[idx_slice].permute(1,0,2)
                    for idx_slice in selected_indices
                ], dim=0)
                memory_pos_flat = torch.cat([
                    self.memory_pos[idx_slice].permute(1,0,2)
                    for idx_slice in selected_indices
                ], dim=0)

            else:
                # For initial frame, use current memory as is
                memory_flat = self.memory#.view(4096*self.memory_length, 1, 64)
                memory_pos_flat = self.memory_pos#.view(4096*self.memory_length, 1, 64)
                # pos is ok with 1 batch
            #memory_flat = torch.stack([memory_flat]*n_inference, dim=0) # match batch size
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
            # ============ Restore original spatial shape ============
            # Current: tmp_curr is (4096, n_inferece, 256) etc → transpose axes to (n_infrecem, 256, 4096) → view to (B, dim, H, W)
            #print(in_curr.shape) # 4096 11 256
            in_curr = in_curr.permute(1, 2, 0).reshape(n_inference, 256, 64, 64)#.reshape(1, -1, 64, 64)     # 4096, n_inference, 256 -> (n_inference, 256, 64, 64)
            if n0_flag:
                out_curr = in_curr
                curr[self.depth] = F.interpolate(in_curr[:, :C_in], size=incurr_size, mode='bilinear', align_corners=False)

            elif C_in % 256 != 0:
                #out_curr = torch.cat([in_curr, residue], dim=1)
                out_curr = in_curr
                curr[self.depth] = F.interpolate(torch.cat([in_curr.reshape(1, -1, 64, 64), residue], dim=1), size=incurr_size, mode='bilinear', align_corners=False)
            else:
                out_curr = in_curr
                curr[self.depth] = F.interpolate(out_curr.reshape(1, -1, 64, 64), size=incurr_size, mode='bilinear', align_corners=False)
        else:
            out_curr = in_curr
        #print(out_curr.shape)
        return curr, out_curr # feature_map, feature_map[-1]: shape: n_inference, 256,64,64

    @torch.inference_mode()
    def _memory_bank_update(self, pix_feat, mask):
        """
        - mask: invert 1 channel after softmax (1 - mask) → pass to downstream method
        - Add new frame embedding to memory (self.memory) in queue-like manner
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
            # For initial frame, copy same features to all slots
            for i in range(len(self.memory)):
                #torch.Size([10, 11, 4096, 64]) torch.Size([10, 11, 4096, 64])
                self.memory[i, :, :, :] = v_feat
                self.memory_pos[i, :, :, :] = v_pos
            self.is_first_frame.fill_(False)
        else:
            # Shift existing memory back by 1 slot
            
            self.memory[1:, :] = self.memory[:-1, :].clone()
            self.memory_pos[1:,:] = self.memory_pos[:-1, :].clone()
            
            # Place latest frame at the front
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
        pix_feat = self.encoder(img)
        mask = self.decoder(pix_feat)
        if self.frame_count != 0:
            pix_feat, curr = self._memory_attention(pix_feat)
        else:
            _, curr = self._memory_attention(pix_feat)
        
        mask = self.decoder(pix_feat)
        mask = self.seg_head(mask)
        self._memory_bank_update(curr, mask)
        return mask

def replace_batch_norm_with_group_norm(model, num_groups=32):
    """Convert BatchNorm2d to GroupNorm to support batch size 1"""
    import torch.nn as nn
    
    def get_valid_num_groups(num_channels, target_groups):
        """Adjust group count so that channel count is divisible by group count"""
        if num_channels < target_groups:
            return 1
        # Find divisors of channel count
        for i in range(min(target_groups, num_channels), 0, -1):
            if num_channels % i == 0:
                return i
        return 1  # Default is 1
    
    def replace_in_module(module):
        """Recursively replace BatchNorm2d with GroupNorm in module"""
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                # Calculate valid group count
                valid_groups = get_valid_num_groups(child.num_features, num_groups)
                
                # Convert to GroupNorm
                new_module = nn.GroupNorm(
                    num_groups=valid_groups,
                    num_channels=child.num_features,
                    eps=child.eps,
                    affine=child.affine
                )
                # Copy weights
                if child.affine:
                    new_module.weight.data = child.weight.data.clone()
                    new_module.bias.data = child.bias.data.clone()
                
                # Replace module
                setattr(module, name, new_module)
                print(f"Replaced BatchNorm2d with GroupNorm: {name} (channels: {child.num_features}, groups: {valid_groups})")
            else:
                # Recursively process child modules
                replace_in_module(child)
    
    replace_in_module(model)
    return model