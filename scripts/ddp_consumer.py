import argparse
import os
import json

import torch
from tqdm.auto import tqdm

from accelerate import Accelerator,DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import QwenImageTransformer2DModel

from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from preprocess_dataset import loader, path_done_well
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

# from diffusers.loaders import AttnProcsLayers
from diffusers import QwenImageEditPlusPipeline
import gc

logger = get_logger(__name__, log_level="INFO")

# > tools -----------------------------------------------------------------------------

# fix env for deepspeed
def fix_env_for_deepspeed():
    for src, dst in [
        ("OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"),
        ("OMPI_COMM_WORLD_RANK", "RANK"),
        ("OMPI_COMM_WORLD_SIZE", "WORLD_SIZE"),
    ]:
        if src in os.environ and dst not in os.environ:
            os.environ[dst] = os.environ[src]

    for k in [
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_NODE_RANK",
    ]:
        os.environ.pop(k, None)


# args parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for Qwen Image Edit (Accelerate+DeepSpeed)")

    # Paths / Basics
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--logging_dir", type=str, default="./logger")
    parser.add_argument("--pretrained_model", type=str, default="Qwen-Image-Edit-2509")

    # LoRA / Quant
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--quantize", action="store_true", help="Enable 8-bit quantization for blocks")
    parser.add_argument("--adam8bit", action="store_true", help="Use bitsandbytes Adam8bit optimizer")

    # Optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    # Training loop controls
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=5000)

    # LR schedule
    parser.add_argument("--lr_warmup_steps", type=int, default=10)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    # System / misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=250)

    # Caches
    parser.add_argument("--txt_cache_dir", type=str, default="text_embs/")
    parser.add_argument("--img_cache_dir", type=str, default="img_embs/")
    parser.add_argument("--control_img_cache_dir", type=str, default="img_embs_control/")

    return parser.parse_args()


# > main -----------------------------------------------------------------------------

def main():
    # > fix env for deepspeed
    fix_env_for_deepspeed()

    # > config
    args = parse_args()
    args.weight_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    args.output_dir, args.logging_dir, args.pretrained_model, args.txt_cache_dir, args.img_cache_dir, args.control_img_cache_dir = path_done_well(
        args.output_dir, args.logging_dir, args.pretrained_model, args.txt_cache_dir, args.img_cache_dir, args.control_img_cache_dir)

    # > deepSpeed & Accelerator
    project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=args.output_dir / args.logging_dir,
    )
    deepspeed = DeepSpeedPlugin(zero_stage=3, gradient_clipping=1.0)# zero-stage3 is hard in harmony with peft
    accelerator = Accelerator(deepspeed_plugin=deepspeed, log_with="wandb", project_config=project_config)


# > load model in -----------------------------------------------------------------------------
    
    # > define image_encoding_pipeline VAE
    vae_cfg_path = args.pretrained_model / "vae/config.json"
    vae = None
    with open(vae_cfg_path, "r") as f:
        vae = json.load(f)

    quantization_config = None
    model_kwargs = {"subfolder": "transformer"}
    if args.quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=args.weight_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = args.weight_dtype

    # > load flux
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model,
        **model_kwargs)
    
    # optional quant
    if args.quantize:
        flux_transformer = prepare_model_for_kbit_training(flux_transformer)
    else:
        flux_transformer.to(accelerator.device)

    
    
    # > LoRA config
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=2*args.rank,
        init_lora_weights='gaussian',
        target_modules=[
        "to_k", "to_q", "to_v", "to_out.0", 
        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",],
    )
    
    flux_transformer = get_peft_model(flux_transformer, lora_config)
    gc.collect()
    torch.cuda.empty_cache()

    # > training preparing -----------------------------------------------------------------------------
    
    # Freeze all parameters by default; enable grad only for LoRA layers.
    for n, p in flux_transformer.named_parameters():
        p.requires_grad = ("lora" in n)

    flux_transformer.train()
    flux_transformer.enable_gradient_checkpointing()

    # 统计可训练参数
    trainable_params = [p for p in flux_transformer.parameters() if p.requires_grad]
    print(sum(p.numel() for p in trainable_params) / 1e6, 'parameters (trainable)')

    if args.adam8bit:
        optimizer = bnb.optim.Adam8bit(trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),)
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # > dataloader
    train_dataloader = loader(
        args.train_batch_size,
        args.num_workers,
        txt_cache_dir=args.txt_cache_dir,
        img_cache_dir=args.img_cache_dir,
        ctrl_cache_dir=args.control_img_cache_dir,
    )

    # > lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    vae_scale_factor = 2 ** len(vae.get("temperal_downsample"))

    # > noise schedular
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model,
        subfolder="scheduler",
    )

    def get_sigmas(timesteps, n_dim=4, dtype=args.weight_dtype):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # > training prep -----------------------------------------------------------------------------

    # Accelerator Prepare
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    # > wandb init -----------------------------------------------------------------
    accelerator.init_trackers(
        project_name="qwen_lora",      # 自定义项目名
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "grad_accum_steps": args.gradient_accumulation_steps,
            "dtype": str(args.weight_dtype),
            "rank": args.rank,
            "num_gpus": torch.cuda.device_count(),
        }
    )


    # > training loop -----------------------------------------------------------------------------

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                img, prompt_embeds, prompt_embeds_mask, control_img = batch

                prompt_embeds = prompt_embeds.to(dtype=args.weight_dtype, device=accelerator.device)
                prompt_embeds_mask = prompt_embeds_mask.to(dtype=torch.int32, device=accelerator.device)
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                control_img = control_img.to(dtype=args.weight_dtype, device=accelerator.device)
                pixel_latents = img.to(dtype=args.weight_dtype, device=accelerator.device)

                # (B, C, F, H, W)
                pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
                control_img = control_img.permute(0, 2, 1, 3, 4)

                latents_mean = (
                    torch.tensor(vae.get("latents_mean"))
                    .view(1, 1, vae.get("z_dim"), 1, 1)
                    .to(pixel_latents.device, pixel_latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.get("latents_std")).view(1, 1, vae.get("z_dim"), 1, 1).to(
                    pixel_latents.device, pixel_latents.dtype
                )

                # 规范化（与预编码配置一致）
                pixel_latents = (pixel_latents - latents_mean) * latents_std
                control_img = (control_img - latents_mean) * latents_std

                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=pixel_latents.dtype)

                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none",
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=pixel_latents.device)

                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                # pack
                packed_noisy_model_input = QwenImageEditPlusPipeline._pack_latents(
                    noisy_model_input,
                    bsz,
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],
                )
                packed_control_img = QwenImageEditPlusPipeline._pack_latents(
                    control_img,
                    bsz,
                    control_img.shape[2],
                    control_img.shape[3],
                    control_img.shape[4],
                )

                img_shapes = [[(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                                (1, control_img.shape[3] // 2, control_img.shape[4] // 2)]] * bsz

                packed_noisy_model_input_concated = torch.cat([packed_noisy_model_input, packed_control_img], dim=1)
                

                # forward
                model_pred= flux_transformer(
                    hidden_states=packed_noisy_model_input_concated,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]

                # unpack
                model_pred = QwenImageEditPlusPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                "train_loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch,
                }, step=global_step)

                train_loss = 0.0
                if (global_step % args.checkpointing_steps) == 0:
                    save_path = args.output_dir / f"checkpoint-{global_step}"
                    if accelerator.is_main_process:
                        save_path.mkdir(exist_ok=True)
                        unwrapped_model = accelerator.unwrap_model(flux_transformer)
                        unwrapped_model.save_pretrained(save_path,safe_serialization=True)

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break   

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()
