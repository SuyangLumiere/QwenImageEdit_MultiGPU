import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from accelerate import dispatch_model
from diffusers import QwenImageTransformer2DModel
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers, logger
from diffusers.utils.torch_utils import maybe_allow_in_graph

# dispatch flux transformer---------------------------------------------------------------------------------------------------------
class MultiGPUTransformer():
    """multi GPU 包装器，通过 device_map 自动分配（pipeline/model parallel by block）"""
    def __init__(self, transformer):
        self.transformer = transformer
        self.num_gpus = max(torch.cuda.device_count(), 1)
        self.total_blocks = len(transformer.transformer_blocks)
        # 均匀切块
        self.split_points = [i*(self.total_blocks // self.num_gpus) for i in range(1, self.num_gpus)]

    @property
    def device_map(self):
        device_map = {}
        res = 0
        # 非 transformer_blocks 子模块放到 cuda:0
        for name, _ in self.transformer.named_children():
            if name != "transformer_blocks":
                device_map[name] = "cuda:0"
        # 按块切分 transformer_blocks：从 cuda:1 开始依次映射
        for item, splt in enumerate(self.split_points):
            temp = {f"transformer_blocks.{i}": f"cuda:{item+1}" for i in range(res, splt)}
            res = splt
            device_map.update(temp)

        temp = {f"transformer_blocks.{i}": f"cuda:{self.num_gpus-1}" for i in range(res, self.total_blocks)}
        device_map.update(temp)

        return device_map

    def auto_split(self):
        # accelerate dispatch
        try:
            model = dispatch_model(self.transformer, device_map=self.device_map)
            print("Successfully applied device_map using accelerate")
        except Exception as e:
            print(f"Error with accelerate dispatch: {e}")
            model = self.transformer
            pass
        return model

# wrapped transformer blocks---------------------------------------------------------------------------------------------------------
@maybe_allow_in_graph
class DoubleTransformerBlock(QwenImageTransformerBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        ref_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ref_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}

        if ref_states is not None and ref_rotary_emb is not None:
            ref_mod_params = self.img_mod(temb)
            txt_mod_params = self.txt_mod(temb)

            ref_mod1, _ = ref_mod_params.chunk(2, dim=-1)
            txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

            ref_normed = self.img_norm1(ref_states)
            ref_modulated, _ = self._modulate(ref_normed, ref_mod1)

            txt_normed = self.txt_norm1(encoder_hidden_states)
            txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

            _, txt_attn_output = self.attn(
            hidden_states=ref_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=ref_rotary_emb,
            **joint_attention_kwargs,
        )
            encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
            txt_normed2 = self.txt_norm2(encoder_hidden_states)
            txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
            txt_mlp_output = self.txt_mlp(txt_modulated2)
            encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

# wrapped transformer---------------------------------------------------------------------------------------------------------
class doubleStringTransformer(QwenImageTransformer2DModel):
    def __init__(self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        *args, **kwargs):

        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
            *args, **kwargs)

        self.transformer_blocks = nn.ModuleList([
            DoubleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,) for _ in range(self.num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        latent_seq_lens: int = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )


        if latent_seq_lens:
            ref_states = hidden_states[:, latent_seq_lens : ]
            ref_states = self.img_in(ref_states)

            hidden_states = self.img_in(hidden_states)
            image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

            ref_shapes = [shape[1:] for shape in img_shapes]
            ref_rotary_emb = self.pos_embed(ref_shapes, txt_seq_lens, device=hidden_states.device)
        else:
            ref_states = None
            ref_rotary_emb = None
            hidden_states = self.img_in(hidden_states)
            image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )


        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    ref_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                    ref_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    ref_states = ref_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    ref_rotary_emb=ref_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)


        return Transformer2DModelOutput(sample=output)

