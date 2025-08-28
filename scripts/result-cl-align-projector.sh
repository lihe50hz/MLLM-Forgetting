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