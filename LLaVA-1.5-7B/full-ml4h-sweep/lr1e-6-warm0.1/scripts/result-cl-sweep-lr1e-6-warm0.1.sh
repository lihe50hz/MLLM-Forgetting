# echo "-------baseline--------"
# python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/ml4h/full-sft-cl/Baseline
# echo "-------After Pathology-VQA--------"
# python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/ml4h/full-sft-cl-align-projector-lr1e-6-warm0.1/Pathology-VQA
# echo "-------After Surgical-VQA--------"
# python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/ml4h/full-sft-cl-align-projector-lr1e-6-warm0.1/Surgical-VQA
# echo "-------After Cell-VQA--------"
# python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/ml4h/full-sft-cl-align-projector-lr1e-6-warm0.1/Cell-VQA
# echo "-------After Radiology-VQA--------"
# python data/evaluators/CL-evaluators/eval_all_orginal.py --result-dir working/llava-1.5-7b/ml4h/full-sft-cl-align-projector-lr1e-6-warm0.1/Radiology-VQA

echo "-------After Baseline--------"
python data/evaluators/CL-evaluators/ml4h.py --result-dir working/llava-1.5-7b/ml4h/full-sft-cl-align-projector-lr1e-6-warm0.1/Baseline

echo "-------After Pathology-VQA--------"
python data/evaluators/CL-evaluators/ml4h.py --result-dir working/llava-1.5-7b/ml4h/full-sft-cl-align-projector-lr1e-6-warm0.1/Pathology-VQA

echo "-------After Surgical-VQA--------"
python data/evaluators/CL-evaluators/ml4h.py --result-dir working/llava-1.5-7b/ml4h/full-sft-cl-align-projector-lr1e-6-warm0.1/Surgical-VQA

echo "-------After Cell-VQA--------"
python data/evaluators/CL-evaluators/ml4h.py --result-dir working/llava-1.5-7b/ml4h/full-sft-cl-align-projector-lr1e-6-warm0.1/Cell-VQA

echo "-------After Radiology-VQA--------"
python data/evaluators/CL-evaluators/ml4h.py --result-dir working/llava-1.5-7b/ml4h/full-sft-cl-align-projector-lr1e-6-warm0.1/Radiology-VQA
