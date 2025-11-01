# QwenImageEdit_MultiGPU
A lightweight implementation of the Qwen-Image-Edit model for inference and LoRA fine-tuning on 8×V100 GPUs
---

## 📦 Installation

**Requirements:**
- Python 3.10

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Install the latest `diffusers` from GitHub:
   ```bash
   pip install git+https://github.com/huggingface/diffusers
   ```
--

## 🌟 QuickStart
**Confirm you are all ready for processing the arguments properly**

1. run produce.sh to precompute embedds.

2. run consume.sh to train lora on your Qwen-Image-Edit model.
