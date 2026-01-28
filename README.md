# <img src="img/logo.png" width="180" align="left" />QwenImageEdit_MultiGPU



A lightweight, clean implementation of **Qwen-Image-Edit** supporting inference + LoRA fine‑tuning on **multi‑GPU (8×V100)** setups.





## 📦 Installation

**Requirements:**

- Python 3.10

1. Install required packages:
   ⚡️ NEW: Our `requirements.txt` handles the project installation and critical version upgrades in one go:
   ```bash
   pip install -r requirements.txt
   ```

2. In case you encounter an error like the following:  
   ```
   AttributeError: 'dict' object has no attribute 'to_dict'
   ```
   **How to fix it**
   ```bash
   pip install --upgrade diffusers transformers accelerate
   ```
   > **⚠️ Warning:transformers has release version 5.0.0, which is incompatible to present environment,use "transformers<5.0.0" to instead**
---

## 🏋️ Training Workflow

Training now follows a **two‑stage pipeline**:

<img src="img/2stage.png" width="1000" />

### 1. Precompute embeddings  
Run the producer script:
```bash
bash produce.sh
```
This step processes your dataset and saves precomputed embeddings for the trainer.

### 2. Train LoRA  
After produce has finished:
```bash
bash consume.sh
```
This launches the LoRA trainer based on the new architecture.

<br>
⚡️ New Features / Updates

**Added ddp_consumer**:
A version of the consumer that supports DistributedDataParallel (DDP).
> **⚠️ Note:** PEFT and DeepSpeed may have limited compatibility, especially on V100 GPUs when using quantization.

<br>

---

## 🚀 Inference
<img src="img/demo.png" width="1000" />

You now have **two** inference choices:

### Option A — Rewritten Fast Pipeline (recommended)
Located in `qwen_infer/vanillaPipeline.py`.

Run:

```bash
python quick_infer.py
```

**Advantages:**
- Completely rewritten pipeline  
- customized transformer behavior(optional)  
- Generates results much quicker than the official pipeline  

### Option B — Official Pipeline w/ Multi‑GPU  
If you still need the “official” behavior:

```bash
cd scripts
python infer.py
```

**Note:**  
The official pipeline may take **~1h20m per image**.  
The rewritten pipeline takes about **20 minutes** and is suitable for most use cases.

---

## 📏 Recommended Resolution for Quick Validation

If you just want to **inspect LoRA training quality**,  
use **512×512** resolution during inference.

This significantly reduces compute load and speeds up iteration.The whole generation will cost **only 6 mins**(50 steps).
> By going a step further and turning off CFG entirely—for instance, by providing an empty negative prompt-the runtime drops to around 3 minutes.

---

## 📂 Project Structure

```
qwen_image/
│
├── README.md
├── setup.py
├── requirements.txt
│
├── QwenEdit/
│   ├── __init__.py
│   ├── pipe.py
│   ├── utils.py
│   └── data.py
│
├── scripts/
│   ├── infer.py
│   ├── quick_infer.py
│   ├── producer.py
│   ├── pp_consumer.py
│   └── ddp_consumer.py
│
├── produce.sh
├── consume.sh
└── quick_infer.sh
```

---

## 🌟 Summary

- Environment unified → install once at top level  
- Training = **produce → consume**  
- Inference = **rewritten fast pipeline** (recommended) or **official pipeline**  
- Transformer behaviors can be easily customized through modifications to wrapped_tool.py
- Use **512 resolution** when quickly checking training results  

Enjoy your multi‑GPU Qwen‑Image‑Edit workflow. ❤️


## Star History

If you find this project helpful or interesting, a star would be greatly appreciated! Your support motivates us to keep improving. ⭐


[![Star History Chart](https://api.star-history.com/svg?repos=SuyangLumiere/QwenImageEdit_MultiGPU&type=date&legend=top-left)](https://www.star-history.com/#SuyangLumiere/QwenImageEdit_MultiGPU&type=date&legend=top-left)
