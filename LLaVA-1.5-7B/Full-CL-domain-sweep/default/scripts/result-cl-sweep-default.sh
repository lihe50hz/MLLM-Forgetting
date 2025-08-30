echo "-------baseline--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/full-sft-cl/Baseline
echo "-------After RS--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/full-sft-cl-align-projector-default/RS
echo "-------After Med--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/full-sft-cl-align-projector-default/Med
echo "-------After AD--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/full-sft-cl-align-projector-default/AD
echo "-------After Sci--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/full-sft-cl-align-projector-default/Sci
echo "-------After Fin--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/full-sft-cl-align-projector-default/Fin
