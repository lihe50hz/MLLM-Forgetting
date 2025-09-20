# echo "-------baseline--------"
# python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/ml4h/lora-sft-cl/Baseline
# echo "-------After Pathology-VQA--------"
# python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/ml4h/lora-sft-cl-align-projector-lr1e-4-warm0.1-rank16/Pathology-VQA
# echo "-------After Surgical-VQA--------"
# python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/ml4h/lora-sft-cl-align-projector-lr1e-4-warm0.1-rank16/Surgical-VQA
# echo "-------After Cell-VQA--------"
# python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/ml4h/lora-sft-cl-align-projector-lr1e-4-warm0.1-rank16/Cell-VQA
# echo "-------After Radiology-VQA--------"
# python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/ml4h/lora-sft-cl-align-projector-lr1e-4-warm0.1-rank16/Radiology-VQA

echo "-------After Baseline--------"
python data/evaluators/CL-evaluators/ml4h.py --result-dir working/llava-1.5-7b/ml4h/lora-sft-cl-align-projector-lr1e-4-warm0.1-rank16/Baseline

echo "-------After Pathology-VQA--------"
python data/evaluators/CL-evaluators/ml4h.py --result-dir working/llava-1.5-7b/ml4h/lora-sft-cl-align-projector-lr1e-4-warm0.1-rank16/Pathology-VQA

echo "-------After Surgical-VQA--------"
python data/evaluators/CL-evaluators/ml4h.py --result-dir working/llava-1.5-7b/ml4h/lora-sft-cl-align-projector-lr1e-4-warm0.1-rank16/Surgical-VQA

echo "-------After Cell-VQA--------"
python data/evaluators/CL-evaluators/ml4h.py --result-dir working/llava-1.5-7b/ml4h/lora-sft-cl-align-projector-lr1e-4-warm0.1-rank16/Cell-VQA

echo "-------After Radiology-VQA--------"
python data/evaluators/CL-evaluators/ml4h.py --result-dir working/llava-1.5-7b/ml4h/lora-sft-cl-align-projector-lr1e-4-warm0.1-rank16/Radiology-VQA
