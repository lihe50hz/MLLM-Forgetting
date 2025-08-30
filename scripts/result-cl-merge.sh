echo "-------baseline--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl/Baseline
echo "-------After RS--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-merge/RS
echo "-------After Med--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-merge/Med
echo "-------After AD--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-merge/AD
echo "-------After Sci--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-merge/Sci
echo "-------After Fin--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-merge/Fin