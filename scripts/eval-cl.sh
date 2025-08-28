export WANDB_PROJECT="VLM-Transfer"

while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl/RS/export/config.json ]; do
    sleep 10
done
echo "RS exported"
sleep 180

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/1-RS/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/1-RS/Fin.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/1-RS/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/1-RS/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/1-RS/Sci.yaml

while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl/Med/export/config.json ]; do
    sleep 10
done
echo "Med exported"
sleep 180

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/2-Med/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/2-Med/Fin.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/2-Med/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/2-Med/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/2-Med/Sci.yaml

while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl/AD/export/config.json ]; do
    sleep 10
done
echo "AD exported"
sleep 180

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/3-AD/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/3-AD/Fin.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/3-AD/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/3-AD/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/3-AD/Sci.yaml

while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl/Sci/export/config.json ]; do
    sleep 10
done
echo "Sci exported"
sleep 180

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/4-Sci/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/4-Sci/Fin.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/4-Sci/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/4-Sci/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/4-Sci/Sci.yaml

while [ ! -f /pasteur/u/lihe50hz/VLMTrans/working/llava-1.5-7b/lora-sft-cl/Fin/export/config.json ]; do
    sleep 10
done
echo "Fin exported"
sleep 180

llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/5-Fin/AD.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/5-Fin/Fin.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/5-Fin/Med.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/5-Fin/RS.yaml
llamafactory-cli train /pasteur/u/lihe50hz/VLMTrans/LLaVA-1.5-7B/temp/LoRA-CL/5-Fin/Sci.yaml