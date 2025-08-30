echo "-------baseline--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl/Baseline
echo "-------After RS--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/RS
echo "-------After Med--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/Med
echo "-------After AD--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/AD
echo "-------After Sci--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/Sci
echo "-------After Fin--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/Fin
echo "-------After sample20-full-10--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/sample20-full/checkpoint-10
echo "-------After sample20-full-20--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/sample20-full/checkpoint-20
echo "-------After sample20-full-30--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/sample20-full/checkpoint-30
echo "-------After sample20-full-40--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/sample20-full/checkpoint-40

# echo "-------baseline--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl/Baseline
# echo "-------After RS--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/RS
# echo "-------After Med--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/Med
# echo "-------After AD--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/AD
# echo "-------After Sci--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/Sci
# echo "-------After Fin--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector/Fin