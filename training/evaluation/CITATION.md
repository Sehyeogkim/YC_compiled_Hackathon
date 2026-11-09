# External result for the pitch

Suggested wording:

> An independent reproduction reports 75.5% success for HuggingFaceVLA/smolvla_libero across 200 LIBERO-Spatial episodes.

한국어:

> 외부 재현 실험에서 HuggingFaceVLA/smolvla_libero는 LIBERO-Spatial 200회 평가에서 75.5% 성공률을 기록했다.

Source: Xingdong Zuo, *SmolVLA on LIBERO-Spatial: Reproducible Evaluation Notes*, published June 28, 2026. [Main result](https://huggingface.co/blog/zuoxingdong/smolvla-libero-spatial-eval#main-result).

The cited row uses checkpoint `HuggingFaceVLA/smolvla_libero@6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, 10 spatial tasks × 20 episodes, MuJoCo 3.3.2, OSMesa, FP32, relative control, `n_action_steps=1`, `num_steps=10`, and default 360-pixel observation rendering. This is the experiment author's report, not a measurement by WeRescue and not an official guarantee for a single soup-can task.

Our additional 50-trial check concerns the **LIBERO-Object soup-can task**, with 25 development positions × 2 policy RNG seeds and an explicitly recorded ±4cm-per-axis target-position variation. It is a separate local-condition result, not a reproduction of the cited suite average. Preserve the final post-training evaluation pool separately.
