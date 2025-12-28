import os
import torch
from accelerate import dispatch_model


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



