"""Mask-selected foreground-token encoder for SelEx asymmetric-view training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricForegroundEncoder(nn.Module):
    """Shared ViT with a full-token global path and a foreground-token path."""

    def __init__(self, backbone, max_foreground_tokens=128, min_foreground_tokens=1):
        super().__init__()
        if max_foreground_tokens < 1 or min_foreground_tokens < 1:
            raise ValueError("Foreground token limits must be positive.")
        self.backbone = backbone
        self.max_foreground_tokens = max_foreground_tokens
        self.min_foreground_tokens = min_foreground_tokens
        self.foreground_cls_token = nn.Parameter(backbone.cls_token.detach().clone())

    def _align_mask(self, mask, token_count, device):
        mask = mask.to(device=device, dtype=torch.float32)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() != 4:
            raise ValueError(f"Expected [B,H,W] or [B,1,H,W] mask, got {tuple(mask.shape)}.")
        side = int(token_count ** 0.5)
        if side * side != token_count:
            raise ValueError("Foreground token count must form a square grid.")
        mask = F.interpolate(mask, size=(side, side), mode="nearest")
        return mask.squeeze(1).flatten(1) > 0.5

    def _pack_tokens(self, patch_tokens, selected):
        selected = selected.clone()
        selected[selected.sum(dim=1) < self.min_foreground_tokens] = True
        lengths = selected.sum(dim=1).clamp(max=self.max_foreground_tokens)
        max_length = int(lengths.max().item())
        packed = patch_tokens.new_zeros(patch_tokens.shape[0], max_length, patch_tokens.shape[-1])
        valid = torch.zeros(patch_tokens.shape[0], max_length, dtype=torch.bool, device=patch_tokens.device)
        for index in range(patch_tokens.shape[0]):
            tokens = patch_tokens[index][selected[index]]
            if tokens.shape[0] > self.max_foreground_tokens:
                positions = torch.linspace(0, tokens.shape[0] - 1, self.max_foreground_tokens, device=tokens.device)
                tokens = tokens.index_select(0, positions.round().long())
            packed[index, :tokens.shape[0]] = tokens
            valid[index, :tokens.shape[0]] = True
        return packed, valid

    @staticmethod
    def _masked_attention(attention, x, valid):
        batch, count, dim = x.shape
        head_dim = dim // attention.num_heads
        qkv = attention.qkv(x).reshape(batch, count, 3, attention.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = (query @ key.transpose(-2, -1)) * attention.scale
        key_mask = valid[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1).masked_fill(~key_mask, 0.0)
        weights = weights * valid[:, None, :, None].to(weights.dtype)
        weights = attention.attn_drop(weights)
        output = (weights @ value).transpose(1, 2).reshape(batch, count, dim)
        output = attention.proj_drop(attention.proj(output))
        return output * valid.unsqueeze(-1).to(output.dtype)

    def _masked_block(self, block, x, valid):
        x = x + block.drop_path(self._masked_attention(block.attn, block.norm1(x), valid))
        x = x * valid.unsqueeze(-1).to(x.dtype)
        mlp = block.mlp(block.norm2(x)) * valid.unsqueeze(-1).to(x.dtype)
        return (x + block.drop_path(mlp)) * valid.unsqueeze(-1).to(x.dtype)

    def global_features(self, images):
        return self.backbone(images)

    def foreground_features(self, images, mask):
        prepared = self.backbone.prepare_tokens(images)
        patch_tokens = prepared[:, 1:]
        selected = self._align_mask(mask, patch_tokens.shape[1], images.device)
        packed, valid = self._pack_tokens(patch_tokens, selected)
        cls = self.foreground_cls_token + self.backbone.pos_embed[:, :1]
        x = torch.cat([cls.expand(images.shape[0], -1, -1), packed], dim=1)
        x = self.backbone.pos_drop(x)
        valid = torch.cat([torch.ones(images.shape[0], 1, dtype=torch.bool, device=images.device), valid], dim=1)
        for block in self.backbone.blocks:
            x = self._masked_block(block, x, valid)
        return self.backbone.norm(x)[:, 0]

    def forward(self, global_images, foreground_images=None, foreground_mask=None):
        if foreground_images is None:
            return self.global_features(global_images)
        return torch.cat(
            [self.global_features(global_images), self.foreground_features(foreground_images, foreground_mask)],
            dim=0,
        )
