"""NumPy forward pass of the OpenCLIP ViT-B/32 image tower.

The learned parameters are supplied as float32 arrays and are never updated.
Only the arithmetic of the forward pass is reproduced here, with NumPy
reshape/transpose, matrix multiplication, reduction and broadcasting.

Data path and shapes (B = batch size):

    [B,3,224,224]  prepared image batch
        patchify    -> [B,49,3072]      49 patch rows, each flattened C,H,W
        embed       -> [B,50,768]       CLS row prepended, learned positions added
        ln_pre      -> [B,50,768]       pre-Norm applied once before block 0
        12 blocks   -> [B,50,768]       attention + MLP, two residuals each
        CLS select  -> [B,768]          tokens[:,0,:]
        ln_post     -> [B,768]
        projection  -> [B,512]          768 -> 512, stored [in,out]
        unit vector -> [B,512]          one normalised row per image

Parameter conventions that matter for this checkpoint:
  * visual.conv1.weight holds [768,3,32,32]; reshaping it to [768,3072] keeps the
    out-channel as the row, so a patch row is projected with patch_weight.T.
  * Every attention/MLP linear layer stores [out,in]; the transpose is applied
    at the matmul, never to the stored array.
  * visual.proj is already [in,out] = [768,512] and must NOT be transposed.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    """Multi-head self-attention over the token axis of one Transformer block."""

    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']
        self.qkv_bias = weights[prefix + '.in_proj_bias']
        self.out_weight = weights[prefix + '.out_proj.weight']
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads
    def split_heads(self, x):
        """[B,N,D] -> [B,H,N,D/H].

        The feature axis D is split into H heads of D/H channels. The head axis
        is created last and then moved in front of the token axis, so token
        order inside every head is unchanged: token n of head h always holds
        features h*D/H .. (h+1)*D/H of token n.
        """
        B, N, D = x.shape
        x = x.reshape(B, N, self.heads, self.head_dim)

        return x.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        qkv = x @ self.qkv_weight.T + self.qkv_bias

        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])
        key_rows = k.transpose(0, 1, 3, 2)
        scores = q @ key_rows

        scores = scores * np.float32(1.0 / np.sqrt(self.head_dim))

        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)
        context = coefficients @ v
        merged = context.transpose(0, 2, 1, 3).reshape(x.shape)

        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    """Pre-Norm block: attention, MLP and one residual addition each."""

    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        update = self.attention(self.norm1(x), causal=causal)

        residual = x + update
        u = self.norm2(residual)
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    """The complete image tower used by the final project."""

    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']
    def patchify(self, images):
        """[B,3,224,224] -> [B,49,3072].

        The 224x224 image is cut into a 7x7 grid of 32x32 patches. Patch rows
        run left to right and then top to bottom. Inside one patch the values
        are flattened in C,H,W order, so a patch contributes 3*32*32 = 3072
        numbers. The two spatial axes are split and reordered instead of
        flattened directly, because a direct reshape of [B,3,224,224] would put
        C in front of the patch index and scramble the row order.
        """
        B, C, H, W = images.shape
        patch_size = 32
        grid_h, grid_w = H // patch_size, W // patch_size
        x = images.reshape(B, C, grid_h, patch_size, grid_w, patch_size)
        x = x.transpose(0, 2, 4, 1, 3, 5)
        return x.reshape(B, grid_h * grid_w, C * patch_size * patch_size)

    def embed(self, images):
        """[B,3,224,224] -> [B,50,768]: patch tokens, CLS row and positions."""
        patches = self.patchify(images)

        tokens = patches @ self.patch_weight.T
        cls = np.tile(self.cls, (images.shape[0], 1, 1))


        x = np.concatenate([cls, tokens], axis=1)
        return x + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        cls_tokens = tokens[:, 0, :]
        normalized = self.post_norm(cls_tokens)
        output = normalized @ self.projection

        return unit_vector(output)
