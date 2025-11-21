import torch
import numpy as np
from PIL import Image
import math
from diffusers import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_shift, calculate_dimensions


# load image
def get_image(path):
    img = Image.open(path).convert("RGB")
    pass

    return img

# calculate dimension for easy divised by 32
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height

class VanillaPipeline(QwenImageEditPlusPipeline):

    @torch.inference_mode()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        target_area: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 1024,
    ):


        # 1. Encode input image
        calculated_width, calculated_height = calculate_dimensions(target_area, image.size[0] / image.size[1])
        contorl_image = self.image_processor.resize(image, calculated_height, calculated_width)

        device = torch.device("cuda:0")

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                    image=[contorl_image],
                    prompt=[prompt],
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=max_sequence_length, # modified here to save time
        )
        if negative_prompt:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=[contorl_image],
                prompt=[negative_prompt],
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )

        # 2. Prepare latent variables
        contorl_image = self.image_processor.preprocess(contorl_image, calculated_height, calculated_width).unsqueeze(2)
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            contorl_image,
            num_images_per_prompt,
            num_channels_latents,
            calculated_height,
            calculated_width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        print(latents.shape)

        # 3. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            contorl_image.device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 4. Denoising loop
        transformer_device = next(self.transformer.parameters()).device
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # 将latents移到transformer设备
                latent_model_input = latents.to(transformer_device)
                #img_shapes = [(1, latents.shape[3]// 2, latents.shape[4]//2), (1, image_latents.shape[3]//2, image_latents.shape[4]//2)]
                img_shapes = [
            [
                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
            ]
        ]
                if image_latents is not None:
                    latent_model_input = torch.cat([latents.to(transformer_device), image_latents.to(transformer_device)], dim=1)
                latent_model_input = latent_model_input.to(self.transformer.dtype)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(self.transformer.dtype).to(transformer_device)
                prompt_embeds = prompt_embeds.to(self.transformer.dtype)

                with self.transformer.cache_context("cond"):
                    
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]
                    noise_pred = noise_pred.to(device)
                if negative_prompt:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_prompt_embeds_mask.sum(dim=1).tolist(),
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    neg_noise_pred = neg_noise_pred.to(device)
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)


                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, calculated_height, calculated_width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
