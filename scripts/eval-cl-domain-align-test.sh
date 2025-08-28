export WANDB_PROJECT="VLM-Transfer"

# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl-align-fix/RS/export/config.json ]; do
#     sleep 10
# done
# echo "RS exported (fix)"
# sleep 240

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/1-RS/AD.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/1-RS/Fin.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/1-RS/Med.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/1-RS/RS.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/1-RS/Sci.yaml

# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl-align-projector/RS/export/config.json ]; do
#     sleep 10
# done
# echo "RS exported (projector)"
# sleep 240

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/1-RS/AD.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/1-RS/Fin.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/1-RS/Med.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/1-RS/RS.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/1-RS/Sci.yaml

# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl-align-fix/Med/export/config.json ]; do
#     sleep 10
# done
# echo "Med exported (fix)"
# sleep 240

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/2-Med/AD.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/2-Med/Fin.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/2-Med/Med.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/2-Med/RS.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/2-Med/Sci.yaml

# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl-align-projector/Med/export/config.json ]; do
#     sleep 10
# done
# echo "Med exported (projector)"
# sleep 240


# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/2-Med/AD.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/2-Med/Fin.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/2-Med/Med.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/2-Med/RS.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/2-Med/Sci.yaml


# while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl-align-fix/AD/export/config.json ]; do
#     sleep 10
# done
# echo "AD exported (fix)"
# sleep 240

# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/3-AD/AD.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/3-AD/Fin.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/3-AD/Med.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/3-AD/RS.yaml
# llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/3-AD/Sci.yaml

while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl-align-projector/AD/export/config.json ]; do
    sleep 10
done
echo "AD exported (projector)"
# sleep 240


llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/3-AD/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/3-AD/Fin.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/3-AD/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/3-AD/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/3-AD/Sci.yaml


while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl-align-fix/Sci/export/config.json ]; do
    sleep 10
done
echo "Sci exported (fix)"
# sleep 240

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/4-Sci/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/4-Sci/Fin.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/4-Sci/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/4-Sci/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/4-Sci/Sci.yaml

while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl-align-projector/Sci/export/config.json ]; do
    sleep 10
done
echo "Sci exported (projector)"
# sleep 240

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/4-Sci/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/4-Sci/Fin.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/4-Sci/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/4-Sci/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/4-Sci/Sci.yaml

while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl-align-fix/Fin/export/config.json ]; do
    sleep 10
done
echo "Fin exported (fix)"
# sleep 240

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/5-Fin/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/5-Fin/Fin.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/5-Fin/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/5-Fin/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-fix/5-Fin/Sci.yaml

while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl-align-projector/Fin/export/config.json ]; do
    sleep 10
done
echo "Fin exported (projector)"
# sleep 240

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/5-Fin/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/5-Fin/Fin.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/5-Fin/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/5-Fin/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL-Align-Projector/5-Fin/Sci.yaml