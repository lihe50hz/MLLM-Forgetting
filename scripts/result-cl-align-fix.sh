echo "-------baseline--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl/Baseline
echo "-------After RS--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-fix/RS
echo "-------After Med--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-fix/Med
echo "-------After AD--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-fix/AD
echo "-------After Sci--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-fix/Sci
echo "-------After Fin--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-fix/Fin
echo "-------After sample20--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-fix/sample20