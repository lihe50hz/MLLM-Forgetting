# echo "-------baseline--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl/Baseline
# echo "-------After RS--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align/RS
# echo "-------After Med--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align/Med
# echo "-------After AD--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align/AD
# echo "-------After Sci--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align/Sci
# echo "-------After Fin--------"
# python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align/Fin


echo "-------baseline--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl/Baseline
echo "-------After RS--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align/RS
echo "-------After Med--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align/Med
echo "-------After AD--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align/AD
echo "-------After Sci--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align/Sci
echo "-------After Fin--------"
python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/lora-sft-cl-align/Fin