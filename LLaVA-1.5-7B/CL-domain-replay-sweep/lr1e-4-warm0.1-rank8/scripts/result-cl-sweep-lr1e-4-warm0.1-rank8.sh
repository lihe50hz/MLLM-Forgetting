echo "-------baseline--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl/Baseline
echo "-------After RS--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-replay-align-projector-lr1e-4-warm0.1-rank8/RS
echo "-------After Med--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-replay-align-projector-lr1e-4-warm0.1-rank8/Med
echo "-------After AD--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-replay-align-projector-lr1e-4-warm0.1-rank8/AD
echo "-------After Sci--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-replay-align-projector-lr1e-4-warm0.1-rank8/Sci
echo "-------After Fin--------"
python data/evaluators/CL-evaluators/eval_all.py --result-dir working/llava-1.5-7b/lora-sft-cl-replay-align-projector-lr1e-4-warm0.1-rank8/Fin
