import os
import torch
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration

# ------------------- 配置部分 ------------------- #
# 要下载的 Hugging Face Hub 模型ID
hub_model_id = "liuhaotian/llava-v1.5-7b"

# 您想要将修复好的、完整的模型保存在本地的文件夹名称
# 脚本会自动创建这个文件夹
local_save_path = "/pasteur/u/lihe50hz/models/llava-v1.5-7b-complete"
# ------------------------------------------------ #

print(f"即将从 Hugging Face Hub 下载模型 '{hub_model_id}'...")
print(f"并将修复好的完整版保存到本地文件夹: '{local_save_path}'")

# 检查目标文件夹是否已存在
if os.path.exists(local_save_path):
    print(f"\n错误：目标文件夹 '{local_save_path}' 已经存在。")
    print("请删除该文件夹，或在脚本中指定一个新的文件夹名称，然后重试。")
    exit()

# 下载并加载所有必要的组件
# 这可能需要一些时间，并会占用数GB的网络流量和磁盘空间
try:
    print("\n[1/3] 正在加载并创建 Processor (包含 Tokenizer 和 Image Processor)...")
    processor = AutoProcessor.from_pretrained(hub_model_id)
    
    print("[2/3] 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hub_model_id)

    print("[3/3] 正在加载模型权重...")
    model = LlavaForConditionalGeneration.from_pretrained(
        hub_model_id,
        torch_dtype=torch.bfloat16, # 使用 bfloat16 以节省内存
        device_map="auto" # 自动分配到 GPU (如果可用)
    )

    print("\n所有组件加载成功！")
    
    # 将所有组件保存到指定的本地文件夹
    # 这个过程会自动创建 preprocessor_config.json
    print(f"正在将所有组件保存到 '{local_save_path}'...")
    
    processor.save_pretrained(local_save_path)
    tokenizer.save_pretrained(local_save_path)
    model.save_pretrained(local_save_path)
    
    print(f"\n✅ 操作成功！模型已完整保存在 '{os.path.abspath(local_save_path)}'")
    print("现在请在 LLaMA-Factory 的配置中将 'model_name_or_path' 指向这个新路径。")

except Exception as e:
    print(f"\n❌ 操作失败：{e}")
    print("请检查您的网络连接、磁盘空间以及依赖库是否已正确安装。")