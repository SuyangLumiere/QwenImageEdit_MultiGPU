import torch
from PIL import Image
import math
import argparse
from peft import PeftModel
from QwenEdit import VanillaPipeline, MultiGPUTransformer, get_image

# > tools -----------------------------------------------------------------------------

# args parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for Qwen Image Edit (Accelerate+DeepSpeed)")

    # Paths / Basics
    parser.add_argument("--output_img", type=str, default="qwen_test.png")
    parser.add_argument("--pretrained_model", type=str, default="qwen_image_edit")

    # LoRA / Quant
    parser.add_argument("--lora_weight", type=str, default="checkpoint-1/")

    # inputs
    parser.add_argument("--ctrl_img", type=str, default="input.png")
    parser.add_argument("--rank", type=int, default=64, help="Rank for LoRA.")
    parser.add_argument("--prompt", type=str, default="follow the words instruction to edit image")
    parser.add_argument("--neg_prompt", type=str, default="bounding box, red rectangle, text, words, letters, characters, labels, annotations, arrows, markers, highlights, sketches, borders, frames, outlines, watermarks, logos, signatures, captions, instructions, notes, diagrams, symbols, non-photorealistic elements, artifacts, residual traces, incomplete erasure, partial removal")

    # infer arguments
    parser.add_argument("--target_area", type=int, default=512*512, help="Approximate target area (H*W) for 32-aligned resize")
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=6.0, help="Classifier-Free Guidance scale.")

    # seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    return parser.parse_args()

    
# > main -----------------------------------------------------------------------------

def main():
    args = parse_args()
    dtype = torch.bfloat16

    pipe = VanillaPipeline.from_pretrained(args.pretrained_model,
                                                    torch_dtype=dtype).to("cpu")

    pipe.vae.to("cuda:0")
    pipe.text_encoder.to("cuda:0")


    flux_transformer = MultiGPUTransformer(pipe.transformer).auto_split()
    if args.lora_weight:
        print(f"Loading LoRA weights from: {args.lora_weight}")
        def _unwrap(m):
                        return m._orig_mod if hasattr(m, "_orig_mod") else m
        _unwrap_flux = _unwrap(flux_transformer)
        flux_transformer = PeftModel.from_pretrained(_unwrap_flux, args.lora_weight, low_cpu_mem_usage=False)
    flux_transformer.eval()
    pipe.transformer = flux_transformer


    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    image = get_image(args.ctrl_img)

    inputs = {
        "image": image,
        "prompt": args.prompt,
        "generator": generator,
        "true_cfg_scale": args.cfg_scale,
        "negative_prompt": args.neg_prompt,
        "num_inference_steps": args.infer_steps,
        "target_area":args.target_area,
        "max_sequence_length":1024
    }

    pipe.set_progress_bar_config(disable=None)

    with torch.inference_mode():
        output = pipe(**inputs)
        output_image = output.images[0]
        output_image.save(args.output_img)
    print(f"Image successfully saved to {args.output_img}")

if __name__ == "__main__":
    main()    


