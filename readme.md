# 2026 October ARR: Cross-lingual Representation Alignment

## Set Conda Environment
```
conda create -n octarr python=3.11 -y
conda activate octarr
pip install -r requirements.txt
```

## 1. 연구 목표

기존 언어 정렬은 동일 의미의 언어 A/B 문장 representation을 직접
일치시키거나 cosine similarity를 1에 가깝게 만드는 경우가 많다. 그러나
이러한 방식은 의미 정보뿐 아니라 언어 정체성, 어순, 형태론 등
language-specific information까지 제거할 수 있다.

본 연구의 목표는 **언어 간 representation gap을 0으로 제거하는 것이
아니다.** 선행 연구에서 관찰된 systematic language-specific structure를
보존하면서, 동일 의미 문장 사이에서 의미와 관련된 성분만 일관되게
대응시키는 것을 목표로 한다.

Representation을 개념적으로 다음처럼 분해한다.

\[
h_i^L = c_L + s_i^L
\]

- \(c_L\): 언어 \(L\)에 공통적인 language-specific component 또는
  language centroid
- \(s_i^L\): 문장 \(i\)의 의미와 관련된 centered component

언어 A/B의 동일 의미 문장 쌍에 대해 raw representation
\(h_i^A, h_i^B\)를 직접 같게 만드는 대신, 언어 성분을 제거한
\(s_i^A, s_i^B\)를 정렬한다. 동시에 \(c_B-c_A\)로 나타나는 언어 간
gap은 유지한다.

### Gap consistency

문장 쌍 \(i\)에 대해 다음을 정의한다.

\[
d_i^{A\rightarrow B}=h_i^B-h_i^A
\]

\[
\mathcal{L}_{gap}
=
\operatorname{Var}_i(d_i^{A\rightarrow B})
=
\frac{1}{N}\sum_i
\left\|
d_i^{A\rightarrow B}-\bar d^{A\rightarrow B}
\right\|_2^2
\]

여기서 \(\bar d^{A\rightarrow B}\)는 배치 또는 데이터셋에서 추정한 평균
언어 offset이다. 이 loss를 최소화하면 모든 문장 쌍의 차이를 0으로 만드는
것이 아니라 다음 관계를 유도한다.

\[
h_i^B-h_i^A \approx \bar d^{A\rightarrow B}
\]

즉, 동일 의미 문장 사이에는 **0이 아닌 일관된 language-specific
offset**이 존재할 수 있다. 위 식은 다음 centered alignment와 동치이다.

\[
\left(h_i^B-c_B\right)
\approx
\left(h_i^A-c_A\right)
\]

따라서 언어별 출력 형태와 전역 위치는 유지하면서, centroid를 제거한
의미 공간의 상대적 위치만 가깝게 만드는 것이 핵심 가설이다.

### Gap preservation

다만 분산 항만 최소화하면 평균 offset
\(\bar d^{A\rightarrow B}\)의 크기와 방향은 제약되지 않는다. 즉, 학습
도중 평균 gap이 0으로 이동하거나 전체 representation이 collapse해도
분산은 작아질 수 있다. “언어 간 격차 유지”를 목적함수 수준에서 보장하려면
frozen pretrained model에서 기준 offset을 계산해야 한다.

\[
\mu_{AB}^{0}
=
\mathbb{E}_i
\left[h_{0,i}^B-h_{0,i}^A\right]
\]

학습 중 offset은 다음과 같다.

\[
\mu_{AB}^{\theta}
=
\mathbb{E}_i
\left[h_{\theta,i}^B-h_{\theta,i}^A\right]
\]

이에 대한 보존 항을 다음처럼 둘 수 있다.

\[
\mathcal{L}_{preserve}
=
\left\|
\mu_{AB}^{\theta}-\mu_{AB}^{0}
\right\|_2^2
\]

필요하면 offset의 방향과 크기를 분리해 보존한다.

\[
\mathcal{L}_{direction}
=
1-\operatorname{cos}\left(\mu_{AB}^{\theta},\mu_{AB}^{0}\right)
\]

\[
\mathcal{L}_{magnitude}
=
\left(
\left\|\mu_{AB}^{\theta}\right\|_2
-
\left\|\mu_{AB}^{0}\right\|_2
\right)^2
\]

### Semantic alignment

의미 정렬은 raw representation이 아니라 centered representation에서
계산한다.

\[
\tilde h_i^L=h_i^L-c_L
\]

예를 들어 다음과 같은 centered cosine loss를 사용할 수 있다.

\[
\mathcal{L}_{semantic}
=
1-\operatorname{cos}\left(\tilde h_i^A,\tilde h_i^B\right)
\]

또는 InfoNCE를 \(h_i^L\)가 아니라 \(\tilde h_i^L\)에 적용할 수 있다.
Learnable projector를 도입한다면 shared semantic subspace에만 alignment
loss를 적용하고, 나머지 language-specific subspace는 pretrained
representation에 대한 distillation으로 보존하는 방식도 가능하다.

최종 목적함수의 개념적 형태는 다음과 같다.

\[
\mathcal{L}
=
\mathcal{L}_{task}
+\lambda_{sem}\mathcal{L}_{semantic}
+\lambda_{gap}\mathcal{L}_{gap}
+\lambda_{pres}\mathcal{L}_{preserve}
\]

각 항의 역할은 다음과 같다.

- \(\mathcal{L}_{semantic}\): 동일 의미의 centered component 정렬
- \(\mathcal{L}_{gap}\): 문장 내용과 무관하게 언어 offset을 일정하게 유지
- \(\mathcal{L}_{preserve}\): 기존 언어 gap의 방향과 크기가 0으로
  사라지는 것을 방지
- \(\mathcal{L}_{task}\): downstream 능력 유지 및 representation collapse
  방지

따라서 본 연구의 주장은 “두 언어 representation을 동일하게 만든다”가
아니라 다음과 같다.

> Preserve language-specific offsets while aligning the shared semantic
> structure of parallel sentences.

이를 검증하기 위해 retrieval/task 성능뿐 아니라 language-ID probing,
언어 centroid 간 거리, pretrained 대비 offset cosine 및 magnitude 보존율을
함께 측정해야 한다.

이 README의 중심은 제안 방법 자체보다 **비교 가능한 베이스라인 실험
프로토콜과 결과표 구성**이다.

---

## 2. 현재 언어 정의

| 구분 | 언어 | Alignment 학습 | Downstream 학습 | 평가 의미 |
|---|---|---:|---:|---|
| Anchor | en | 사용 | 사용 | In-language |
| Training languages | ko, ja, es | 사용 | 사용 | In-language |
| Out languages | fr, de, it | 미사용 | 미사용 | Fully-unseen transfer |

따라서 현재 정의는 다음과 같다.

- **In-language**: en, ko, ja, es
- **Out-language**: fr, de, it
- Out-language는 downstream 학습뿐 아니라 alignment 학습에서도 사용하지
  않은 **fully unseen language**이다.

논문에는 다음을 명시해야 한다.

> Out languages are excluded from both alignment training and downstream task
> training.

### Aligned-transfer와 fully-unseen transfer

현재 코드는 alignment 언어와 downstream 학습 언어에 동일한
training_lang을 사용한다. 따라서 다음 두 조건 중 fully-unseen 조건만
평가한다.

| 조건 | Alignment 데이터 | Downstream label | 예시 |
|---|---:|---:|---|
| Aligned-transfer | 사용 | 미사용 | Alignment에는 fr을 사용하고 task에는 미사용 |
| Fully-unseen transfer | 미사용 | 미사용 | 현재의 fr, de, it |

논문의 주장이 fully-unseen 일반화라면 현재 설정은 일관적이다. 반면
“target 언어의 parallel data만으로 task 능력을 전달할 수 있다”는 주장까지
하려면 alignment-only 언어 집합을 별도로 추가해야 한다.

---

## 3. 비교 베이스라인

논문의 핵심 비교군은 Pretrained 1개와 학습이 필요한 4개, 총 5개이다.

### B0. Pretrained

- 어떤 alignment/downstream update도 적용하지 않는다.
- 모든 학습 방법과 동일한 layer 및 pooling으로 representation을 추출한다.
- 주 평가는 sentence retrieval이다.
- Downstream zero-shot 성능은 supervised transfer가 아니라 참고용
  zero-shot diagnostic으로 표기한다.

### B1. Task-only transfer

- Alignment loss 없이 downstream task만 학습한다.
- 현재 MASSIVE 학습 언어: en, ko, ja, es
- In-language task 성능과 fully-unseen fr, de, it task 성능을 모두
  측정한다.
- 동일한 최종 checkpoint로 retrieval도 측정하여 task tuning이 언어 간
  representation geometry에 미치는 영향을 확인한다.

### B2. InfoNCE-only alignment

- Parallel sentence pair에 symmetric InfoNCE만 적용한다.
- 주 평가는 언어 간 retrieval이다.
- Downstream label을 전혀 사용하지 않았으므로 이 checkpoint의 task
  성능은 **language transfer가 아니라 zero-shot task diagnostic**이다.
- 이 행의 task 점수를 B1/B3/B4의 supervised transfer 점수와 같은 의미로
  해석하면 안 된다.

### B3. InfoNCE then task transfer

- Stage 1: InfoNCE alignment
- Stage 2: Stage-1 checkpoint에서 downstream task 학습
- 최종 checkpoint로 in/out task 성능과 retrieval을 모두 측정한다.
- Stage-1 종료 checkpoint의 retrieval을 추가로 측정하면 downstream
  tuning 전후의 alignment 변화도 분석할 수 있다.

#### 현재 구현에서 반드시 정정할 점

현재 contrastive_then_transfer는 하나의 100k-step Trainer 안에서 50k
step에 objective만 변경한다. 따라서 다음 상태가 Stage 2로 이어진다.

- optimizer momentum/state
- learning-rate scheduler progress
- Stage 1에서 이미 감소한 learning rate

이는 엄밀한 “InfoNCE checkpoint 이후 일반 downstream training”과 같지
않다. 논문용 sequential baseline은 다음 중 하나로 확정해야 한다.

1. **권장**: Stage 1 종료 후 adapter checkpoint를 로드하고 optimizer와
   scheduler를 새로 만들어 Stage 2를 50k step 수행
2. 현재처럼 optimizer/scheduler를 이어 쓰되 이를 명시하고, reset 버전을
   ablation으로 함께 보고

최종 baseline 명칭이 “InfoNCE → Transfer”라면 1번이 더 자연스럽다.

### B4. Alternating InfoNCE/task training

- 한 optimizer update마다 objective를 교대한다.
- 기본 순서: alignment, downstream, alignment, downstream, ...
- 100k total steps에서 alignment 50k, downstream 50k update를 수행한다.
- 동일 checkpoint에서 task 성능과 retrieval을 측정한다.

### 추가 권장 비교군

메인 다섯 방법은 유지하되 reviewer 대응을 위해 다음을 고려한다.

- **Joint weighted sum**:
  \(\mathcal{L}_{task}+\lambda\mathcal{L}_{NCE}\)
- Positive-pair cosine/MSE alignment
- Relational distance 또는 Gram-matrix alignment
- InfoNCE + proposed gap loss

Alternating만 비교하면 “동시에 두 loss를 합친 표준 multi-task
baseline보다 좋은가?”라는 질문이 남을 수 있으므로 joint weighted-sum
baseline은 강하게 권장한다.

---

## 4. Update budget과 공정한 비교

메인 결과는 **objective exposure-matched** 설정으로 구성한다.

| Method | Alignment updates | Task updates | Total updates | 현재 스크립트 |
|---|---:|---:|---:|---|
| Pretrained | 0 | 0 | 0 | 없음 |
| Task-only | 0 | 50k | 50k | scripts/transfer_only.sh |
| InfoNCE-only | 50k | 0 | 50k | scripts/contrastive_only.sh |
| InfoNCE → Task | 50k | 50k | 100k | scripts/contrastive_then_transfer.sh |
| Alternating | 50k | 50k | 100k | scripts/alternative.sh |

이 설정은 task를 사용하는 방법끼리 task update 50k를 맞추고, alignment를
사용하는 방법끼리 alignment update 50k를 맞춘다.

다만 total compute는 B3/B4가 B1/B2의 두 배이므로 다음을 함께 보고한다.

- 메인 표: objective exposure-matched 결과
- 부록: total-update/compute-matched 결과
- 모든 방법: wall-clock time, GPU-hours, peak memory

### Learning-rate schedule 주의

현재 Trainer 기본 scheduler는 total step을 기준으로 동작한다. 따라서
50k 방법과 100k 방법은 objective별 learning-rate trajectory가 다르다.

- Sequential baseline은 Stage 2 optimizer/scheduler reset 여부를 반드시
  확정한다.
- Alternating과 task-only의 objective별 LR 공정성을 위해 constant LR
  실험 또는 objective-aware scheduler ablation을 고려한다.
- 논문에는 scheduler 종류, warmup, optimizer reset 여부를 명시한다.

---

## 5. Representation 및 retrieval 프로토콜

### Representation 추출

모든 방법에서 다음 설정을 동일하게 유지한다.

- Base model
- Hidden-state layer
- Pooling: last-token 또는 mean pooling
- Tokenization 및 maximum length
- Similarity function

현재 기본값:

- Layer: -1
- Pooling: last_token
- InfoNCE temperature: 0.05

Layer/pooling을 validation에서 선택했다면 모든 baseline에 동일하게
적용하고, test 결과를 보고 설정을 바꾸면 안 된다.

### Retrieval 정의

평행한 held-out 데이터
\(\{(x_i^A,x_i^B)\}_{i=1}^{N}\)에서 A 문장 하나가 query일 때 전체 B
문장 \(N\)개를 candidate pool로 사용한다. 정답은 동일 index \(i\)의
B 문장이다.

\[
\hat j
=
\arg\max_j
\operatorname{sim}(h_i^A,h_j^B)
\]

### Retrieval 평가 규칙

- Training parallel data와 완전히 분리된 fixed held-out set 사용
- 모든 방법에 동일한 candidate pool과 동일한 순서 사용
- Batch 내부 후보만 사용하는 평가 금지
- A→B와 B→A를 모두 평가
- 메인 지표: Recall@1
- 보조 지표: Recall@5, MRR
- 언어별 결과와 language/direction macro average 모두 보고
- OPUS train data와 겹치지 않는 FLORES 계열 외부 평가셋 사용 권장

현재 alignment loader는 존재하지 않는 역방향 OPUS config의 오류를
출력하고 계속 진행할 수 있다. 실제로 로드된 language pair와 pair별
sample 수를 run_metadata.json에서 확인해야 하며, 누락된 pair를 조용히
허용한 실험 결과를 사용하면 안 된다. InfoNCE 자체가 symmetric이므로 동일
sentence pair를 역방향 dataset config로 중복 로드할 필요가 있는지도
프로토콜에서 명확히 한다.

---

## 6. Downstream 및 language-transfer 평가

### MASSIVE

- Task: generative slot filling
- In-language: en, ko, ja, es
- Out-language: fr, de, it
- 주 지표: slot micro-F1
- 보조 지표: exact match, language별 F1
- In/out 평균은 language macro average로 계산

학습 데이터 수가 언어마다 다르면 high-resource 언어가 전체 평균을
지배하지 않도록 micro-over-all-examples뿐 아니라 macro-over-languages를
반드시 보고한다.

### XNLI 확장

- Task metric: accuracy
- 동일한 in/out 언어 원칙 적용
- MASSIVE와 별도 표 또는 동일 구조의 두 번째 task block으로 제시
- 각 task에서 같은 baseline 및 update-budget 원칙 유지

### Transfer gap

절대적인 out-language 성능을 주 지표로 사용한다. 다음 값은 보조 분석으로
사용할 수 있다.

\[
\text{Transfer Gap}
=
\text{In-language score}
-
\text{Out-language score}
\]

Gap이 작더라도 in/out 성능이 모두 낮을 수 있으므로 gap만으로 방법을
평가하면 안 된다.

---

## 7. 메인 결과표

\(N_A\)는 alignment optimizer update 수, \(N_T\)는 downstream task
optimizer update 수이다.

| Method | Alignment objective | Schedule | \(N_A\) | \(N_T\) | MASSIVE In Slot-F1 ↑ | MASSIVE Out Slot-F1 ↑ | Transfer Gap ↓ | Retrieval In R@1 ↑ | Retrieval Out R@1 ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Pretrained | – | – | 0 | 0 | –/ZS | –/ZS | – |  |  |
| Task-only | – | Task only | 0 | 50k |  |  |  |  |  |
| InfoNCE-only | InfoNCE | Alignment only | 50k | 0 | ZS | ZS | – |  |  |
| InfoNCE → Task | InfoNCE | Sequential | 50k | 50k |  |  |  |  |  |
| Alternating | InfoNCE | 1:1 alternating | 50k | 50k |  |  |  |  |  |
| Proposed method | Gap-based | 확정 필요 |  |  |  |  |  |  |  |

표 작성 규칙:

- 최소 3 seeds, 가능하면 5 seeds의 mean ± standard deviation
- 최고값 bold, 두 번째 값 underline
- Pretrained/InfoNCE-only task 결과는 ZS로 명시
- 동일한 최종 checkpoint로 task와 retrieval 평가
- Test set으로 checkpoint나 hyperparameter를 선택하지 않음
- 유의성 검정 방법과 seed를 appendix에 명시

---

## 8. 세부 결과표

### 8.1 언어별 downstream/transfer

| Method | en | ko | ja | es | In Macro | fr | de | it | Out Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained (ZS) |  |  |  |  |  |  |  |  |  |
| Task-only |  |  |  |  |  |  |  |  |  |
| InfoNCE-only (ZS) |  |  |  |  |  |  |  |  |  |
| InfoNCE → Task |  |  |  |  |  |  |  |  |  |
| Alternating |  |  |  |  |  |  |  |  |  |
| Proposed method |  |  |  |  |  |  |  |  |  |

### 8.2 언어별 bidirectional retrieval

각 A↔B 값은 A→B와 B→A의 평균이다. 방향별 수치는 appendix에 별도로
보고한다.

| Method | en↔ko | en↔ja | en↔es | In Macro | en↔fr | en↔de | en↔it | Out Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained |  |  |  |  |  |  |  |  |
| Task-only |  |  |  |  |  |  |  |  |
| InfoNCE-only |  |  |  |  |  |  |  |  |
| InfoNCE → Task |  |  |  |  |  |  |  |  |
| Alternating |  |  |  |  |  |  |  |  |
| Proposed method |  |  |  |  |  |  |  |  |

### 8.3 학습 비용

| Method | Trainable params | Alignment updates | Task updates | Total updates | GPU-hours | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| Task-only |  | 0 | 50k | 50k |  |  |
| InfoNCE-only |  | 50k | 0 | 50k |  |  |
| InfoNCE → Task |  | 50k | 50k | 100k |  |  |
| Alternating |  | 50k | 50k | 100k |  |  |
| Proposed method |  |  |  |  |  |  |

---

## 9. Proposed gap loss ablation

제안 방법의 효과가 단순한 positive-pair attraction이나 InfoNCE 효과가
아님을 보이기 위해 다음 ablation을 권장한다.

| Variant | Task loss | InfoNCE | Gap variance | Anti-collapse |
|---|---:|---:|---:|---:|
| Task-only | ✓ |  |  | Task supervision |
| InfoNCE-only |  | ✓ |  | Negatives |
| Positive cosine/MSE |  |  | 직접 거리 | 별도 필요 |
| Gap-only |  |  | ✓ | 별도 필요 |
| InfoNCE + Gap |  | ✓ | ✓ | Negatives |
| Task + Gap | ✓ |  | ✓ | Task supervision |
| Task + InfoNCE + Gap | ✓ | ✓ | ✓ | Task + negatives |

Gap variance는 language pair별로 계산해야 한다. 서로 다른 언어 offset이
존재할 수 있으므로 en-ko, en-ja, en-es 샘플을 구분하지 않고 하나의
variance로 계산하면 언어별 offset 차이까지 불필요하게 제거할 수 있다.
따라서 pair-balanced batch 또는 pair별 loss 계산 후 평균을 사용한다.

---

## 10. 현재 구현 상태

### 현재 동작하는 부분

- 네 학습 모드 routing
- 언어별/objective별 validation loss (13개 eval dataset)
- Validation 샘플 단위 입출력/loss 기록 (`eval_samples/step-N.json`)
- Symmetric InfoNCE
- MASSIVE downstream training
- LoRA + 4-bit quantization
- W&B training logging 설정
- Objective별 alignment_loss, downstream_loss
- Objective별 누적 update count
- Learning rate, gradient norm, 전체 training loss
- Package/CUDA/data-size/parameter-count metadata
- 500 optimizer step마다 checkpoint 저장
- LoRA adapter와 optimizer/scheduler/RNG/Trainer state 저장
- Checkpoint resume를 위한 adapter reload

### 학습 중 validation (구현됨)

언어별/objective별 validation loss를 `eval_steps`마다 측정한다.

- Validation dataset은 objective x 언어로 분리해 총 13개를 구성한다.
  MASSIVE 7언어(en, ko, ja, es, fr, de, it)와 alignment 6쌍이다.
- Alignment을 언어쌍별로 분리하는 이유는 InfoNCE가 in-batch negative를
  쓰기 때문이다. en-ko와 en-ja가 섞인 batch의 loss는 어느 언어쌍의
  정렬 품질도 아니라 혼합 negative pool에서의 난이도가 된다.
- `Trainer`가 dict eval_dataset을 지원하므로 metric은
  `eval_massive_out_de_loss`, `eval_align_in_en-ko_loss` 형태로 자동
  생성된다.
- `eval_on_start=True`이므로 step 0에 학습 전 baseline(B0 Pretrained
  행에 해당)이 모든 run에서 자동으로 기록된다.
- OPUS-100 config는 두 언어 코드를 알파벳 순으로 이어 붙인 이름 하나만
  존재한다. anchor를 앞에 두면 anchor보다 앞서는 언어를 놓치므로
  (`en-de`는 없고 `de-en`이 있다) `opus_config_name`으로 방향을
  해석한다.

Validation은 loss만 측정한다. Retrieval R@1과 slot-F1은 비용이 크므로
학습 루프에 넣지 않고 `evaluator.py`로 최종 checkpoint에 대해서만
수행한다.

전체 validation split을 쓰는 비용은 1 round당 forward 약 38,000회
(MASSIVE 7x2033 + alignment 6x2000x2)로, 학습 약 400 step에 해당한다.
`eval_steps=2500`이면 100k step 학습 대비 약 16% 오버헤드이다.

### LoRA target module과 커버리지

PEFT 0.20.0의 기본 mapping은 `llama`/`qwen2`/`qwen3`을
`["q_proj","v_proj"]`로 보내지만 `qwen3_5` 항목이 없어
`get_peft_model`이 `ValueError`를 던진다. 따라서 target module을
`utils.LORA_TARGET_MODULES`에서 `model_type`으로 해석한다.
`--peft_target_modules`로 덮어쓸 수 있고, 해석된 목록은
`run_metadata.json`의 `derived.lora_coverage`에 기록된다.

범위는 QLoRA(Dettmers et al., 2023)의 권고를 따라 **transformer block의
모든 linear layer**(attention + MLP)로 둔다. 해당 논문의 발견은 adapter
개수가 rank보다 중요하고 full finetuning 성능에 맞추려면 모든 linear
layer를 적응시켜야 한다는 것이다. `lm_head`와 embedding은 제외한다.

이 범위 선택은 공정성 문제도 함께 해결한다. Qwen3.5는 하이브리드
어텐션 모델로 `layer_types`가 `linear_attention`과 `full_attention`을
번갈아 두고(`full_attention_interval=4`), q/k/v/o_proj는 full_attention
층에만 존재한다. 따라서 `["q_proj","v_proj"]`만 targeting하면 Qwen3.5는
전체 층의 1/4에만 adapter가 붙는다.

| Model | 층 | `q_proj,v_proj` 커버리지 / trainable | 전체 linear 커버리지 / trainable |
|---|---:|---:|---:|
| Llama-3.2-1B-Instruct | 16 | 100% / 1,703,936 | 100% / 11,272,192 |
| Llama-3.2-3B-Instruct | 28 | 100% / 4,587,520 | 100% / 24,313,856 |
| Qwen2.5-1.5B-Instruct | 28 | 100% / 2,179,072 | 100% / 18,464,768 |
| Qwen2.5-3B-Instruct | 36 | 100% / 3,686,400 | 100% / 29,933,568 |
| Qwen3.5-2B | 24 | **25% / 835,584** | 100% / 15,630,336 |
| Qwen3.5-4B | 32 | **25% / 1,835,008** | 100% / 30,474,240 |

`q_proj,v_proj`만 쓰면 Qwen3.5-2B의 adapter가 더 작은 Llama-3.2-1B의
절반도 되지 않는다. 전체 linear로 바꾸면 학습 파라미터 비율이 모든
모델에서 0.63%~1.03%로 모이고 모델 크기에 따라 단조 증가한다.

Qwen3.5의 `linear_attn` 계열에서 `in_proj_a`와 `in_proj_b`는
`hidden_size -> num_heads`(2048 -> 16) 사상이므로 제외한다. `r=16`
adapter를 붙이면 rank가 출력 차원과 같아져 low-rank가 아니게 된다.

`AutoModelForCausalLM`은 `qwen3_5`를 `Qwen3_5ForCausalLM`으로 매핑하며,
이 클래스의 서브모듈은 `model`과 `lm_head`뿐이다. 체크포인트의 vision/MTP
가중치는 로드되지 않으므로 `language_model.*` 범위 지정은 필요하지 않다.

미등록 `model_type`은 조용히 넘어가지 않고 `ValueError`로 중단하며,
`utils.py`에 항목을 추가하거나 인자를 명시하라고 안내한다.

### Best-checkpoint selection

`metric_for_best_model`은 의도적으로 설정하지 않는다. 언어 macro
평균은 Trainer가 내보내는 키에 없고, `_determine_best_metric`은 없는
키에 대해 `KeyError`를 던진다. 또한 `CustomModel`은 `PeftModel`의
subclass가 아니므로 `load_best_model_at_end=True`는 학습 종료 시점에
`_load_best_model`에서 실패한다.

대신 `trainer_state.json`의 `log_history`에 모든 언어별 loss가 남으므로
선택은 오프라인에서 수행한다.

    python3 scripts/select_checkpoint.py RUN_DIR --rule massive_in --table

Selection에는 **in-language validation만** 사용한다. 기본 rule이
`massive_in`인 이유이다. fr/de/it의 validation loss로 checkpoint를 고르면
gradient에는 out-language label을 쓰지 않았더라도 모델 선택 경로로
target-language 감독이 들어가 fully-unseen transfer 주장과 충돌한다.
`align_out` 역시 out-language parallel data를 selection에 쓰는 것이므로
같은 문제가 있다. out-language 그룹을 지정하면 경고가 출력된다.

| Method | 권장 rule |
|---|---|
| transfer_only, contrastive_then_transfer, alternative | `massive_in` |
| contrastive_only (task 미학습) | `align_in` |

fr/de/it validation/test는 selection과 hyperparameter tuning에 사용하지
않고 최종 평가에만 쓴다.

또한 `eval_steps=2500`에서 50k 방법은 validation 후보가 21개, 100k 방법은
41개다. 모든 step의 최솟값을 고르면 긴 방법이 더 많은 시행을 갖는
selection advantage가 생긴다. Final checkpoint를 쓰거나 objective exposure
기준의 동일한 selection grid를 사전에 확정해야 한다. 스크립트가 후보 수를
출력한다.

이 방식은 selection rule을 재학습 없이 바꿀 수 있다는 장점도 있다.
`eval_steps`는 `save_steps`의 배수여야 한다. 그렇지 않으면 best로
선택된 step에 checkpoint가 존재하지 않는다.

`save_total_limit`은 `None`으로 유지해야 한다. `metric_for_best_model`을
설정하지 않으면 `state.best_model_checkpoint`가 None이고,
`rotate_checkpoints`가 보호할 대상을 모르기 때문에 best checkpoint가
삭제될 수 있다. checkpoint 1개는 약 24MB이다.

### 현재 수행하지 않는 부분

- Retrieval / MASSIVE generation의 학습 중 측정 (최종 평가로만 수행)
- XNLI evaluator
- Proposed gap loss (`compute_gap_consistency_loss`는 구현되어 있으나
  `forward`에 연결되지 않았다)

---

## 11. 로깅 및 저장 구조

W&B:

- Project: Oct_ARR
- Run name:
  model__training_type__in_languages__out_languages__timestamp
- 기본 logging interval: 10 optimizer steps
- W&B에 model checkpoint artifact는 기본 업로드하지 않음
- 서버 checkpoint와 W&B metric logging을 분리

서버 저장 경로:

    results/
    └── meta-llama__Llama-3.2-1B/
        └── run_name/
            ├── run_metadata.json
            ├── train_results.json
            ├── trainer_state.json
            ├── adapter_model.safetensors
            ├── adapter_config.json
            ├── experiment_config.json
            ├── eval_samples/
            │   ├── step-0.json
            │   ├── step-2500/
            │   └── ...
            ├── checkpoint-500/
            ├── checkpoint-1000/
            └── ...

`eval_samples/step-N.json`은 `"<objective>/<언어>"`를 키로 하고, 각
샘플의 입력/정답/loss를 담는다. MASSIVE는 `utt`, `target`, `loss`,
`num_target_tokens`를, alignment는 `source_text`, `target_text`,
`loss`, `positive_cosine`을 기록한다.

Eval sampler가 `SequentialSampler`이므로 기록되는 샘플은 매 round마다
각 언어의 동일한 앞쪽 N개다. 동일 샘플의 loss 변화를 추적하기에는
적합하지만 무작위 표본은 아니므로 집계 통계로 쓰면 안 된다. 언어별
평균은 `eval_*_loss`를 사용한다.

각 checkpoint에는 다음이 저장된다.

- LoRA adapter
- Adapter config
- Optimizer state
- Scheduler state
- RNG state
- Trainer state
- Tokenizer
- Experiment config

Base model은 model_name으로 다시 로드할 수 있으므로 4-bit base weight
전체를 매 checkpoint마다 중복 저장하지 않는다.

wandb login은 실행 스크립트마다 호출하지 않고 서버 계정에서 한 번만
수행한다. API key를 shell script에 기록하지 않는다.

---

## 12. 실행

실행 환경을 먼저 활성화해야 한다. 기본 `python3`에는 bitsandbytes가 없어
4-bit 로딩 단계에서 실패한다.

    conda activate octarr
    nohup bash scripts/transfer_only.sh    > logs/transfer_only.log 2>&1 &      # GPU 0
    nohup bash scripts/contrastive_only.sh > logs/contrastive_only.log 2>&1 &   # GPU 1

두 스크립트가 각각 모델 6개 x method 2개 = 12 run을 순차 실행한다.
`alternative.sh`와 `contrastive_then_transfer.sh`는 단일 모델용 스크립트로
남겨둔다.

### num_steps와 objective update 수는 다르다

1 step은 **하나의 objective**에 대한 1 optimizer update이다. 따라서
objective exposure를 맞추려면 method마다 num_steps가 달라야 한다.

| Method | num_steps | alignment | task | 스크립트 |
|---|---:|---:|---:|---|
| transfer_only | 50k | 0 | 50k | transfer_only.sh |
| contrastive_only | 50k | 50k | 0 | contrastive_only.sh |
| contrastive_then_transfer | **100k** | 50k | 50k | transfer_only.sh |
| alternative | **100k** | 50k | 50k | contrastive_only.sh |

`contrastive_then_transfer`와 `alternative`는 두 objective를 모두 쓰므로
num_steps가 두 배여야 한다. 이 값은 `planned_objective_updates`와
`objective_update_counts`가 계산하며 `run_metadata.json`에 기록되므로
실행 후 반드시 확인한다.

### Learning rate와 scheduler

`--learning_rate`, `--lr_scheduler_type`, `--warmup_ratio`로 제어한다.
기본값은 기존 동작인 `1e-3` / `linear` / `0.0`이다. transformers v5는
`warmup_ratio`를 제거하고 `warmup_steps`만 두므로 `main.py`가
`int(warmup_ratio * num_steps)`로 변환한다. 비율로 받는 이유는 방법마다
`num_steps`가 50k/100k로 달라 고정 step이 서로 다른 비율이 되기
때문이다.

현재 `1e-3`은 관행보다 높다. 원 LoRA 논문은 GPT-3에 2e-4를, QLoRA는
7B/13B에 2e-4, 33B/65B에 1e-4를 쓴다. 커뮤니티 레시피도 1B~8B에서
1e-4~3e-4 범위다. 여기에 `alpha/r = 32/16 = 2`가 LoRA 업데이트를 2배
증폭하므로 유효 스텝은 scaling=1 기준 약 2e-3이 된다. target module을
전체 linear로 확장한 뒤 adapter가 8배 커진 것도 같은 방향으로 작용한다.

다만 q/v-only + 1e-3에서 학습은 안정적이었다(Qwen2.5-3B,
`transfer_only`가 step 2000에 loss 0.19, grad_norm 약 2.0). LR을 바꾸기
전에 `scripts/lr_sweep.sh`로 확인한다.

#### Scheduler 선택이 objective 예산에 미치는 영향

전역 decay는 방법별로 objective가 경험하는 평균 LR을 다르게 만든다.
peak를 1로 정규화한 측정값이다.

| Method | linear (align/task) | cosine (align/task) | constant |
|---|---|---|---|
| transfer_only (50k) | – / 0.501 | – / 0.501 | – / 1.0 |
| contrastive_only (50k) | 0.501 / – | 0.501 / – | 1.0 / – |
| **contrastive_then_transfer (100k)** | **0.751 / 0.251** | **0.819 / 0.182** | 1.0 / 1.0 |
| alternative (100k) | 0.501 / 0.500 | 0.501 / 0.500 | 1.0 / 1.0 |

`alternative`는 두 objective가 전 구간에 균일하게 섞이므로 이미
`transfer_only`/`contrastive_only`와 일치한다. 어긋나는 것은
`contrastive_then_transfer` 하나이고, task phase가 `transfer_only`의 절반
LR만 경험한다. 스위치 지점 LR은 linear와 cosine 모두 정확히 0.5 x peak라
cosine으로 바꿔도 문제 위치는 그대로이고, 뒷구간을 더 눌러 비대칭만
커진다.

따라서 cosine 전환은 이 문제의 해법이 아니다. 해법은 두 가지다.

1. **권장**: sequential baseline의 stage-2에서 optimizer와 scheduler를
   재생성한다. 그러면 각 stage가 독립적으로 peak -> 0을 경험해 align
   0.501 / task 0.501로 모든 방법과 일치한다. 10절의 sequential baseline
   정정과 같은 작업이다.
2. `--lr_scheduler_type constant`로 스케줄을 변수에서 제거한다. 목적함수
   수준의 공정성은 확보되지만 annealing이 없어 절대 성능이 낮을 수 있다.

### 재현성

`--training_seed`는 `set_seed`로 **모델 생성 전에** 적용한다. `Trainer`도
`__init__`에서 `set_seed`를 호출하지만 그 시점은 이미 LoRA의 A 행렬이
초기화된 뒤다. 이 호출이 없으면 동일 seed로도 매 run마다 LoRA 초기값이
달라진다(B는 0 초기화라 step 0의 출력은 같지만 이후 궤적이 갈린다).

`run_name`에 seed를 포함하므로 seed sweep 시 디렉터리가 충돌하지 않는다.

### 프롬프트 템플릿

`Qwen3.5` 계열의 chat template은 generation prompt에서 `<think>` 블록을
연다. 그대로 두면 모델이 답 앞에 `</think>`를 생성하고, slot 파서가 이를
첫 slot 이름의 일부로 읽어 false positive로 집계한다(완벽한 예측의 F1이
1.00에서 0.50으로 떨어진다). 또한 prompt/full 렌더링의 공백 토큰 병합이
달라져 label 마스킹의 prefix 가정이 깨진다.

따라서 `_apply_chat_template`은 `enable_thinking=False`를 전달한다. 이
플래그를 참조하지 않는 Llama/Qwen2.5 템플릿은 그냥 무시한다. 마스킹은
길이를 믿지 않고 실제 공통 prefix를 측정하며, dataset 생성 시
`check_prompt_prefix`가 가정 위반을 한 번 경고한다.

---

## 12.1 진단 스크립트

메인 스윕과 분리된 짧은 실행이다. 출력은 `scripts/compare_runs.py`로 읽는다.

### LR sweep

    conda activate octarr
    nohup bash scripts/lr_sweep.sh > logs/lr_sweep.log 2>&1 &
    python3 scripts/compare_runs.py results_lr_sweep

`contrastive_only`를 1e-3 / 5e-4 / 2e-4로 2000 step씩 돌린다. 판정할
질문은 다음이다. lr=1e-3에서 `align_in`이 2.95 -> 0.70으로 잘 학습되는
동안 `massive_in`이 4.60 -> 9.70으로 악화했다. 이것이 LR 때문인지
InfoNCE 자체의 성질인지 구분해야 한다.

- 낮은 LR에서 `align_in`이 비슷하게 내려가면서 `massive_in` 악화가
  작다면 LR 문제이고, 1e-3 결과는 피해를 과대평가한 것이다.
- 악화가 정렬 개선폭에 비례한다면 objective 자체의 성질이며, 제안하는
  분산 기반 loss의 동기를 뒷받침한다.

논문에서 "InfoNCE-only가 task 능력을 파괴한다"를 주장하려면 LR 하나만
가지고는 방어가 어렵다.

### Layer probe

    nohup bash scripts/layer_probe.sh > logs/layer_probe.log 2>&1 &
    python3 scripts/compare_runs.py results_layer_probe --step 0

`eval_on_start`가 첫 optimizer step 전에 평가하므로 **학습 없이**
사전학습 모델의 layer별 정렬 품질을 읽을 수 있다. layer 8 / 18 / 27 / -1
과 last_token / mean을 조합해 step 0의 `eval_align_in_*`을 비교한다.

현재 기본값인 layer 8은 Qwen2.5-3B에서 36층 중 22% 깊이이고, step 0
`align_in`이 2.95로 batch 16의 우연 수준 `ln(16) = 2.77`보다 나빴다.
사전학습 표현이 문장 임베딩으로 거의 작동하지 않는다는 뜻이므로 layer를
측정에 근거해 고르고, 이후 모든 baseline에 동일하게 적용한다.

두 스크립트 모두 `--eval_language_scope in`을 쓴다. out-language 데이터는
selection과 tuning에 쓰지 않고 최종 평가에만 사용한다.

---

## 13. 논문 결과 생성 전 체크리스트

- [ ] Sequential baseline의 optimizer/scheduler reset 정책 확정
- [ ] 모든 baseline에 동일한 layer/pooling 적용
- [ ] 실제 로드된 alignment pair와 pair별 sample 수 확인
- [ ] Train/validation/test parallel data leakage 검사
- [ ] Fixed full-corpus retrieval candidate pool 구축
- [ ] A→B/B→A 양방향 retrieval 구현
- [ ] MASSIVE slot micro-F1 및 language macro 구현
- [ ] XNLI accuracy evaluator 구현
- [ ] Final checkpoint 또는 validation-selected checkpoint 정책 사전 확정
- [ ] 최소 3개 seed 실행
- [ ] Exposure-matched와 compute-matched 결과 분리
- [ ] GPU-hours와 peak memory 기록
- [ ] Gap loss의 collapse 방지 및 pair별 계산 검증
