echo "-------baseline--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl/Baseline
echo "-------After RS--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector-1.0e-4-0.03/RS
echo "-------After Med--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector-1.0e-4-0.03/Med
echo "-------After AD--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector-1.0e-4-0.03/AD
echo "-------After Sci--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector-1.0e-4-0.03/Sci
echo "-------After Fin--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector-1.0e-4-0.03/Fin
