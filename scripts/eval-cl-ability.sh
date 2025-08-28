export WANDB_PROJECT="VLM-Transfer"

# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/0-Baseline/APP.yaml
# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/0-Baseline/Math.yaml
# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/0-Baseline/OCR.yaml
# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/0-Baseline/VP.yaml

# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/qwen2_5vl-3b/lora-sft-cl/OCR/export/config.json ]; do
#     sleep 10
# done
# echo "OCR exported"
# sleep 180

# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/1-OCR/APP.yaml
# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/1-OCR/Math.yaml
# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/1-OCR/OCR.yaml
# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/1-OCR/VP.yaml

# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/qwen2_5vl-3b/lora-sft-cl/Math/export/config.json ]; do
#     sleep 10
# done
# echo "Math exported"
# sleep 180

# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/2-Math/APP.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/2-Math/Math.yaml
# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/2-Math/OCR.yaml
# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/2-Math/VP.yaml

# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/qwen2_5vl-3b/lora-sft-cl/VP/export/config.json ]; do
#     sleep 10
# done
# echo "VP exported"
# sleep 180

# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/3-VP/APP.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/3-VP/Math.yaml
# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/3-VP/OCR.yaml
# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/3-VP/VP.yaml

# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/qwen2_5vl-3b/lora-sft-cl/APP/export/config.json ]; do
#     sleep 10
# done
# echo "APP exported"
# sleep 180

# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/4-APP/APP.yaml
llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/4-APP/Math.yaml
# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/4-APP/OCR.yaml
# llamafactory-cli train Qwen2.5-VL-3B/temp/CL-ability/4-APP/VP.yaml

