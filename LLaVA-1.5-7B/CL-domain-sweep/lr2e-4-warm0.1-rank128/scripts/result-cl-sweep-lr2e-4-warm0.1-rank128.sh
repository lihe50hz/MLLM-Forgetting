echo "-------baseline--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl/Baseline
echo "-------After RS--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector-lr2e-4-warm0.1-rank128/RS
echo "-------After Med--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector-lr2e-4-warm0.1-rank128/Med
echo "-------After AD--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector-lr2e-4-warm0.1-rank128/AD
echo "-------After Sci--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector-lr2e-4-warm0.1-rank128/Sci
echo "-------After Fin--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-align-projector-lr2e-4-warm0.1-rank128/Fin
