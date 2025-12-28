import os
from PIL import Image
import torch
import fire
from QwenEdit import get_image
from diffusers import QwenImageEditPlusPipeline


def main(
    img_path: str,
    prompt: str = "return the edited image.",
    neg_prompt: str = " ",
    output_path: str = "output.png",
    seed: int = 42,
    cfg_scale: float = 4.0,
    steps: int = 50
):
    pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16, device_map="balanced")
    print("pipeline loaded")

    image = get_image(img_path)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": generator,
        "true_cfg_scale": cfg_scale,
        "negative_prompt": neg_prompt,
        "num_inference_steps": steps,
    }


    pipe.set_progress_bar_config(disable=None)

    with torch.inference_mode():
        output = pipe(**inputs)
        output_image = output.images[0]
        output_image.save(output_path)
        print("image saved at", os.path.abspath(output_path))


if __name__ == "__main__":
    fire.Fire(main)
    
