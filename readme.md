# 2026 October ARR: Cross-lingual Representation Alignment

구현 대조 기준: **2026-10-10**. alignment 목적함수는 기존 5가지와 Gap Distance 변형 2가지이며,
학습 스케줄(`training_type`) 4가지와 별도로 선택한다.
옵션의 기준은 [config.py](config.py), 수식의 기준은 [models.py](models.py)다.
실험별 실제 설정은 해당 run의 `experiment_config.json`을 확인한다.

**Main table은 모델별 baseline 5개 + Ours 후보 3개, 총 8행으로 구성한다.**
Baseline은 Pretrained, Task-only SFT, InfoNCE-only, InfoNCE then SFT,
**Alt. InfoNCE↔SFT**다.
Ours 후보는 ③ Gap Distance, ④ Centered Cosine, ⑤ Gap Direction Cosine InfoNCE이며
모두 **Alt. 제안 loss↔SFT**로 학습한다. 핵심 1:1 대조군은 InfoNCE-Alt다.
최종 논문에서는 공통 validation 기준으로 선정한 Ours 하나를 main에 남기고,
나머지 후보·기존 ② Gap variance·제안 loss의 순차 학습 및 A 실행은 ablation/분석에 둔다.

[지표를 포함한 Main table](reports/main_table_20260928/main_table.md)에
세 모델을 독립된 표로 정리했다. In/Out은 별도 열이며 모델명·Ours·헤더는 병합했다.
[Baseline 포함 재검토 기록](reports/main_table_20260928/result_audit.md)은
날짜 제한 없이 원본 평가 파일을 확인한 결과다. 기존 InfoNCE baseline도 채우고,
제안 방법과 layer/dtype이 다른 행에는 `†`를 표시했다.

평가 수치는 [2026-09-28 결과 보고서](reports/evaluation_20260928/summary.md)에
정리되어 있다. 이 보고서는 9월 22일 이후 시작한 run을 대상으로 하며,
완료/부분 결과의 구분은 보고서 생성 시점 기준이다. 이전 baseline과 거리 분산
실험은 [2026-09-21 전체 실험표](reports/experiment_inventory_20260921/summary.md)를
참고한다. 두 보고서의 layer/dtype 등 조건을 맞추지 않고 loss만의 효과로 비교하지 않는다.

## Set Conda Environment
```
conda create -n octarr python=3.11 -y
conda activate octarr
python -m pip install -r requirements.txt
```

학습 의존성은 2026-10-08 실행 환경의 핵심 버전으로 고정했다.
Qwen3.5 학습 환경과 COMET 환경은 분리한다. 학습 환경의 torch를 COMET용
2.5.1로 낮추지 않는다. 새 서버의 CUDA/드라이버 호환성은 아래 preflight로 확인한다.

## 새 Gap 변형: 다른 서버에서 MASSIVE 실행 (2026-10-10)

`gap_distance_detach`는 정답쌍 평균 거리의 gradient만 차단한다.
`gap_distance_rms`는 전체 후보 거리의 RMS로 거리 편차를 나누며,
평균 거리와 RMS 모두 gradient를 유지한다. RMS의 수치 하한은 1e-6이다.
두 방법 모두 `--alignment_loss`로 선택하며 `same_pair` 배치를 사용한다.

전용 실행기는 **Llama detach → RMS**, **Qwen RMS**를 기본 순서로 실행한다.
각 실험은 MASSIVE `alternative`, 최종층 `-1`, seed 42, BF16/4-bit,
batch 16, learning rate 1e-4, LoRA r=16/alpha=32, 100,000 optimizer updates
(50,000 alignment + 50,000 SFT)다. 검증은 In 언어에서 2,500 step마다,
checkpoint는 1,000 step마다 저장한다. 각 학습 후 In retrieval와
In/Out Slot F1·EM 평가를 완료한 다음 실험으로 넘어간다.

이 실행에는 **준비된 MT/ALMA 파일이나 COMET 환경이 필요하지 않다.**
MASSIVE·OPUS·모델 가중치는 Hugging Face 캐시가 없으면 내려받는다.
Llama 저장소 접근 권한이 있는 계정과 CUDA/BF16 지원 GPU가 필요하다.
`scripts/massive_gap_variants.sh`는 Python 실행기를 호출해 학습·평가 순서를
관리한다. 활성화된 환경의 Python을 학습·평가에도 사용하며,
`PYTHON_BIN=/path/to/python`으로 직접 지정할 수도 있다.

처음 받는 서버에서는 다음과 같이 준비한다. 이미 clone한 경우에는 저장소에서
`git pull --ff-only origin main`을 실행하고 기존 환경을 활성화한다.

```bash
git clone https://github.com/SoeonPark/26_Oct_ARR.git
cd 26_Oct_ARR
conda create -n octarr python=3.11 -y
conda activate octarr
python -m pip install -r requirements.txt
hf auth login
```

설정을 먼저 출력하고 실행 환경을 검사할 수 있다. `--dry-run`은 학습·GPU 사용·
상태 파일 생성을 하지 않는다. `--check-only`는 모델 가중치를 로드하지 않는다.

```bash
bash scripts/massive_gap_variants.sh --model llama --gpu 0 --dry-run
bash scripts/massive_gap_variants.sh --model llama --gpu 0 --check-only
bash scripts/massive_gap_variants.sh --model qwen --gpu 1 --check-only
```

비어 있는 GPU 두 장에서 각각 실행한다. GPU가 한 장이면 두 명령을 순차 실행한다.

```bash
mkdir -p logs
nohup bash scripts/massive_gap_variants.sh --model llama --gpu 0 \
  > logs/massive_gap_llama.log 2>&1 &
nohup bash scripts/massive_gap_variants.sh --model qwen --gpu 1 \
  > logs/massive_gap_qwen.log 2>&1 &
```

단일 변형은 `--loss detach` 또는 `--loss rms`, 두 변형은 `--loss both`로
선택한다. W&B는 기본 disabled이며 `--wandb-mode online`으로 켤 수 있다.
`--state-dir`로 큐 기록 위치를 지정할 수 있다. 새 서버의 실행기는 현재 서버에서
동작하는 프로세스나 로컬 대기 큐를 가져오지 않는다. 완료된 adapter와 평가 결과는
검증 후 재사용하고, 미완료 학습을 발견하면 자동으로 덮어쓰지 않는다.

학습 중 진단 그래프가 필요하면 선택적으로 Matplotlib을 설치한다.
아래 CPU 모니터는 기존 validation 임베딩을 읽으며 추가 모델 추론을 하지 않는다.
각 실행기의 출력에 표시된 state directory를 전달한다.

```bash
python -m pip install matplotlib==3.10.6
python scripts/monitor_gap_variants.py --watch \
  --state-dir /path/to/queue-state --output-dir /path/to/monitor-output
```

기존 Gap 결과를 복사했다면 `--baseline /path/to/original-gap-run`을 추가할 수 있다.
모니터는 원래 B16 배치를 유지해 거리 RMS·정규화 점수 차이·배치 정답률·축소 방향
미분·MASSIVE 검증 loss를 비교한다. 이 배치 정답률은 전체 후보 retrieval가 아니다.

## 다른 서버에서 세 실험 순차 실행 (2026-10-08)

실행 순서는 다음과 같다. 모두 **BF16, 최종층 -1, seed 42, same_pair,
batch 16, 100,000 updates (50,000 representation + 50,000 SFT)**다.

1. Qwen/Qwen3.5-2B — MASSIVE — InfoNCE then SFT.
2. meta-llama/Llama-3.2-1B-Instruct — MT — Gap Distance then SFT.
3. Qwen/Qwen3.5-2B — MT — Gap Distance then SFT.

각 학습의 최종 adapter를 검증하고 **해당 평가까지 끝낸 뒤** 다음 실험으로
넘어간다. MASSIVE는 In Retrieval과 In/Out Slot F1·EM, MT는 In Retrieval과
In BLEU·COMET-22를 평가한다. Out Retrieval은 실행하지 않는다.
Qwen MT의 SFT microbatch는 8이며 optimizer update당 유효 batch는 16이다.
MT 생성 한도 16,384, MASSIVE 생성 한도 128, 평가 batch 16을 유지한다.

**1. 새 서버에서 코드와 두 환경 준비**

```bash
git pull --ff-only origin main
conda create -n octarr python=3.11 -y
conda activate octarr
python -m pip install -r requirements.txt
python -m venv .venv-comet22
.venv-comet22/bin/python -m pip install -r requirements-comet22.txt
# Llama 저장소 접근 권한이 있는 Hugging Face 계정으로 로그인한다.
hf auth login
mkdir -p data/wmt23_alma_ja_opus
```

이미 환경이 있으면 생성 단계를 생략하고 의존성을 갱신한다. COMET 환경에도
CUDA 지원 torch가 필요하다. 모델 가중치·MASSIVE·OPUS·COMET은 캐시가 없으면
처음 실행할 때 다운로드한다. Git에는 데이터·가중치·결과·로그를 포함하지 않는다.

**2. 준비된 MT 데이터 복사**

기존 서버의 `data/wmt23_alma_ja_opus`는 캐시 디렉터리를 가리키는 심볼릭
링크다. 기존 서버에서 아래처럼 **`-L`로 실제 파일을 복사**한다.
`USER@NEW_SERVER`와 대상 저장소 경로는 새 서버에 맞게 바꾼다.

```bash
rsync -aL --info=progress2 data/wmt23_alma_ja_opus/ \
  USER@NEW_SERVER:/path/to/26_Oct_ARR/data/wmt23_alma_ja_opus/
```

복사할 실제 디렉터리는 약 300 MB다. 원시 다운로드 전체는 필요하지 않다.
현재 실험의 `manifest.json` SHA256은 다음과 같으며 새 실행기의 기본값이다.

```text
c36ffdebbe1d3e16e23115e33003e7d14867e0ee6a8d81cfee61dafe53fab0eb
```

새 서버에서 `sha256sum data/wmt23_alma_ja_opus/manifest.json`으로 확인한다.
기존 디렉터리를 복사할 수 없으면 `scripts/prepare_wmt23.py --output_dir
data/wmt23_alma_ja_opus --corpus_profile alma_ja_opus --seed 42`로 준비할 수 있다.
재생성한 manifest는 환경 메타데이터 때문에 해시가 달라질 수 있다. 이때는
데이터 구성을 확인한 뒤 출력된 해시를 `--manifest-sha256`으로 지정한다.

**3. 새 서버에서 확인 후 시작**

```bash
conda activate octarr
python scripts/run_remote_experiments.py --gpu 0 --dry-run
python scripts/run_remote_experiments.py --gpu 0 --check-only
mkdir -p logs
nohup python -u scripts/run_remote_experiments.py --gpu 0 \
  > logs/remote_three_experiments.log 2>&1 &
```

`--gpu`는 사용할 물리 GPU 번호다. 실행기를 호출한 Python을 학습·평가에도
사용하므로 기존 서버의 절대 Python 경로를 수정할 필요가 없다. MT 데이터를
다른 위치에 복사했다면 모든 호출에 `--data-dir /absolute/path`를 추가한다.
COMET 환경도 다른 위치면 `--comet-python /absolute/path/bin/python`으로 지정한다.
기본 W&B는 disabled이며 필요하면 `--wandb-mode online`을 지정한다.

`--dry-run`은 실행 설정만 출력한다. `--check-only`는 데이터 파일 checksum,
설치 버전, CUDA/BF16, 두 모델 config 접근, COMET 환경을 확인하며 모델 가중치를
로드하거나 학습하지 않는다. 실제 학습 속도/메모리는 새 서버에서 확인해야 한다.

상태는 `logs/remote_three_experiments/state.json`, 고정 설정은 같은 디렉터리의
`manifest.json`, 단계별 출력은 `*.evaluation.log`와 `*.retrieval.log`에 기록된다.
평가 결과는 run별 `evaluations/remote_final_*`에 저장한다. 같은 명령을 다시
실행하면 완료된 최종 adapter와 평가 파일을 검증해 재사용한다. 평가 실패는
다음 학습을 막는다. 미완료 학습은 optimizer 상태가 없으므로 자동 재학습하지
않고 해당 run 경로를 알린다. 컨트롤러만 중단되고 자식이 남았다면 종료를 기다린다.
기존 서버의 큐는 별도이므로, 이 세 작업을 옮길 때 기존 대기 항목도 조정해야 한다.

## 1. 연구 목표와 구현된 다섯 목적함수

본 연구는 번역 문장 사이의 표현 차이에 언어별 공통 성분이 있을 수 있다는
가설을 바탕으로, 정답 번역쌍을 다른 후보와 구별하는 정렬 방법을 비교한다.
배치 평균이 순수한 언어 정보라는 가정이나 언어별 정보 보존은 수식만으로
보장되지 않는다. 공통 중심/거리/이동 방향 중 어떤 기준이 유효한지는
retrieval과 downstream 결과로 검증한다.

### 1.1 공통 표기

같은 언어쌍의 평행 문장 microbatch를 \(\{(a_i,b_i)\}_{i=1}^{B}\)로 둔다.
\(a_i,b_i\in\mathbb R^D\)는 선택한 hidden layer와 pooling으로 추출한
정규화 전 임베딩이다. \(i=j\)가 정답이고, \(i\ne j\)는 잘못 연결한 후보다.

\[
\mu_A=\frac1B\sum_i a_i,\quad \mu_B=\frac1B\sum_i b_i,
\qquad u_i=a_i-\mu_A,\quad v_i=b_i-\mu_B
\]

\[
g_{ij}=b_j-a_i,\quad d_{ij}=\|g_{ij}\|_2,\qquad
\bar d=\frac1B\sum_i d_{ii},\quad
\bar g=\frac1B\sum_i g_{ii}=\mu_B-\mu_A
\]

\(\bar d\)는 **정답 gap 길이의 평균**, \(\bar g\)는 **정답 gap 벡터의 평균**이다.
일반적으로 \(\bar d\ne\|\bar g\|_2\)다. 두 기준 모두 대각선 정답쌍으로
계산하며 전체 \(B^2\)개 후보의 평균을 사용하지 않는다.

### 1.2 공통 양방향 InfoNCE

①③④⑤는 \(B\times B\) score 행렬 \(S\)만 다르고 다음 목적을 공유한다.

\[
\mathcal C_\tau(S)=-\frac1{2B}\sum_{i=1}^{B}\left[
\log\frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{B}\exp(S_{ij}/\tau)}
+\log\frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{B}\exp(S_{ji}/\tau)}
\right].
\]

분자는 정답 번역쌍이고, 분모는 **각 query에 대한 정답 포함 전체 B개 후보**다.
첫 항은 A→B, 둘째 항은 B→A다. 모든 \(B^2\)개 score를 하나의 공통 분모로
정규화하지 않는다. 구현은 `logits=S/temperature`에 대한 두 방향 cross-entropy다.
모든 후보 score가 같으면 loss는 \(\log B\)이므로 정답만 당기는 cosine loss와 다르다.

### 1.3 방법별 수식과 해석

| 번호 | 방법 | `--alignment_loss` | 목적함수 |
|---|---|---|---|
| ① | Baseline InfoNCE | `infonce` | \(\mathcal C_\tau(S^{(1)})\) |
| ② | Gap distance variance | `gap_consistency` | 정답 거리의 population variance |
| ③ | Gap Distance InfoNCE (Dist) | `gap_distance_infonce` | \(\mathcal C_\tau(S^{(3)})\) |
| ④ | Centered Cosine InfoNCE (Center) | `centered_infonce` | \(\mathcal C_\tau(S^{(4)})\) |
| ⑤ | Gap Direction Cosine InfoNCE (Dir) | `gap_direction_infonce` | \(\mathcal C_\tau(S^{(5)})\) |

**① Baseline InfoNCE**

\[
S^{(1)}_{ij}=\cos(a_i,b_j),\qquad
\mathcal L_1=\mathcal C_\tau(S^{(1)}).
\]

원래 문장 임베딩의 방향을 비교하여 정답 번역쌍의 cosine을 오답보다 높인다.
언어별 평균을 제거하지 않는다.

**② 기존 Gap: 거리 분산**

\[
\mathcal L_2=\frac1B\sum_i(d_{ii}-\bar d)^2.
\]

정답 번역쌍의 거리만 일정하게 만든다. Negative와 temperature를 사용하지 않으며,
평균 거리를 특정 값에 고정하거나 gap 방향을 정렬하지 않는다. 방향이 정반대여도
길이가 모두 5이면 loss=0이다. 모든 문장 표현이 같아지는 해도 loss=0이므로,
분산 감소를 의미 정렬의 성공으로 바로 해석하지 않는다.

**③ Gap Distance InfoNCE**

\[
S^{(3)}_{ij}=-\frac{(d_{ij}-\bar d)^2}{s_0^2},\qquad
\mathcal L_3=\mathcal C_\tau(S^{(3)}).
\]

정답 거리의 평균에서 벗어난 정도를 음의 score로 바꾸어 정답과 오답을 구별한다.
\(s_0\)는 `--alignment_gap_scale`로 설정하는 고정 상수다. 정답의 거리 오차가
오답보다 작아지도록 학습하며, ②의 분산을 별도 보조 loss로 더하지 않는다.
오답 거리가 \(\bar d\)보다 훨씬 작아져도 오차가 커지므로, 오답의 raw 거리가
반드시 증가한다고 설명해서는 안 된다.

**④ Centered Cosine InfoNCE**

\[
S^{(4)}_{ij}=\cos(u_i,v_j)
=\cos(a_i-\mu_A,b_j-\mu_B),\qquad
\mathcal L_4=\mathcal C_\tau(S^{(4)}).
\]

언어별 배치 평균을 제거한 뒤 정답 문장 방향을 다른 후보와 구별한다.
구현 순서는 **raw 임베딩 → 언어별 평균 제거 → L2 정규화 → cosine**이다.
평균에는 배치의 의미·길이·주제 분포도 섞이므로, 순수한 언어 성분 제거라고
단정하지 않는다.

**⑤ Gap Direction Cosine InfoNCE**

\[
S^{(5)}_{ij}=\cos(g_{ij},\bar g)
=\cos(b_j-a_i,\mu_B-\mu_A),\qquad
\mathcal L_5=\mathcal C_\tau(S^{(5)}).
\]

정답 번역으로의 이동 방향이 공통 언어 이동 방향과 더 잘 맞도록 학습한다.
Negative는 \(b_j-a_i\;(i\ne j)\)이며, 다른 정답쌍의 gap \(b_j-a_j\)를
negative로 사용하지 않는다. 반대 방향에서는 gap과 평균 gap이 함께 부호가
바뀌므로 동일 score 행렬의 transpose를 사용한다.
오답 gap이 평균 gap과 같은 방향이면 길이가 달라도 cosine=1일 수 있다.
평균 gap이 0일 때 수학적 방향은 정의되지 않으며 구현의 epsilon은 수치 보호다.

③은 **길이의 차이** \((\|g_{ij}\|-\bar d)^2\)를 사용한다.
앞서 논의한 **벡터 잔차** \(\|g_{ij}-\bar g\|^2\)와 다르다.
벡터 잔차 score를 쓰는 별도 InfoNCE는 현재 다섯 옵션에 포함되지 않는다.

### 1.4 배치·정밀도·옵션

| 항목 | 현재 구현 |
|---|---|
| CLI 기본 loss | `infonce` |
| 언어쌍 배치 | ②③④⑤는 alignment 학습 시 `same_pair`, B≥2 필요. 평가도 단일 pair 배치 |
| 기준 평균 | 현재 microbatch에서 계산, `detach()` 없이 gradient 유지 |
| Gradient accumulation | microbatch별 loss를 누적. 평균이나 negative pool을 합치지 않음 |
| Temperature | `--alignment_temperature` 기본 0.05; ①③④⑤에 적용 |
| 거리 scale | `--alignment_gap_scale` 기본 1.0; ③에만 적용 |
| 정밀도 | ②는 FP16/BF16을 뺄셈 전에 FP32로 변환. ③④⑤는 score/CE 구간의 autocast를 끄고 FP32 이상으로 계산 |
| Cosine epsilon | ③④⑤ 중 cosine을 쓰는 ④⑤는 `1e-8`; FP64 검증 입력은 유지 |
| 프로세스 | 현재 ②③④⑤ 및 `same_pair` 학습은 단일 프로세스/GPU 지원 |
| 공통 검색 평가 | 모든 loss에서 원래 저장 임베딩의 cosine; ③④⑤의 score로 검색하지 않음 |

각 방법은 alignment loss 하나를 선택하는 방식이다. Loss 가중합, teacher 보존항,
variance/covariance 보조항, EMA centroid, 배치 간 negative 수집은 구현하지 않았다.
①③④⑤의 negative 경쟁도 모든 의미 정보의 보존이나 학습 안정성을 보장하지는 않는다.
배치 내 중복·동의 문장은 false negative가 될 수 있다.

`training_type`은 목적함수와 별개로 alignment-only, task-only, 교대, 순차 학습을
선택한다. `transfer_only`에서는 alignment update가 0이므로 저장된 loss 이름은
alignment validation/evaluation에만 영향을 준다.

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

## 3. 학습 스케줄과 비교 베이스라인

1절의 **다섯 alignment 목적함수**와 아래 **네 학습 스케줄**은 서로 다른
실험 축이다. `contrastive`라는 기존 옵션 이름은 ②의 분산 loss에도 사용한다.

| 표기 | `--training_type` | 동작 |
|---|---|---|
| A (기존 보고서의 C) | `contrastive_only` | 선택한 alignment loss만 학습 |
| T | `transfer_only` | MASSIVE만 학습; alignment loss는 검증에만 사용 |
| A→T (C→T) | `contrastive_then_transfer` | 전반 alignment, 후반 MASSIVE |
| Alt | `alternative` | alignment와 MASSIVE optimizer update를 1:1 교대 |

Main table의 baseline은 아래 B0~B4이며, B4 InfoNCE-Alt가 핵심 대조군이다.
B3 InfoNCE then SFT도 순차 baseline으로 포함한다. 주 비교는 **동일 Alt에서 ①을
③④⑤로 각각 교체하는 실험**이다. ②와 제안 방법의 A/A→T는 보조 ablation이다.

| 역할 | 비교 구성 | 확인할 효과 |
|---|---|---|
| Main baseline | Pretrained / Task-only SFT / InfoNCE-only / InfoNCE then SFT / InfoNCE-Alt | 학습 전·task 단독·alignment 단독·순차·교대 학습 비교 |
| 핵심 1:1 비교 | InfoNCE-Alt ↔ 후보 loss ③④⑤ 각각의 Alt | 동일 스케줄에서 alignment 목적함수 교체의 효과 |
| 보조 실험 | ② Gap variance, 제안 loss의 A/A→T 및 선정되지 않은 후보 | 목적함수 구성 요소와 스케줄 효과 |

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

### B3. InfoNCE then SFT (순차 baseline)

- Stage 1: InfoNCE alignment
- Stage 2: Stage-1 checkpoint에서 downstream task 학습
- 최종 checkpoint로 in/out task 성능과 retrieval을 모두 측정한다.
- Stage-1 종료 checkpoint의 retrieval을 추가로 측정하면 downstream
  tuning 전후의 alignment 변화도 분석할 수 있다.

#### 현재 구현의 순차 학습 정의

현재 contrastive_then_transfer는 하나의 100k-step Trainer 안에서 50k
step에 objective만 변경한다. 따라서 다음 상태가 Stage 2로 이어진다.

- optimizer momentum/state
- learning-rate scheduler progress
- Stage 1에서 이미 감소한 learning rate

현재 A→T 결과는 **optimizer/scheduler를 이어 쓰는 순차 학습**으로 표기한다.
Stage 2에서 optimizer와 scheduler를 재생성하는 reset 방식은 현재 구현에
포함되지 않으며 별도 ablation 후보이다. 총 step 수는 `--num_steps`로 정한다.

### B4. Alternating InfoNCE/task training — 핵심 대조군

- 한 optimizer update마다 objective를 교대한다.
- 기본 순서: alignment, downstream, alignment, downstream, ...
- 100k total steps에서 alignment 50k, downstream 50k update를 수행한다.
- 동일 checkpoint에서 task 성능과 retrieval을 측정한다.

### 제안 방법과 B4의 1:1 대응

아래 세 비교는 모두 `training_type=alternative`를 고정한다.
InfoNCE update와 task update를 교대하는 B4에서 **alignment update의 loss만**
바꾸고, task objective와 교대 순서는 유지한다. Alt 자체는 loss가 아니라
학습 스케줄이며, 비교 대상 loss는 그 안에서 사용되는 alignment 목적함수다.

| 핵심 baseline | 제안 방법 | 바꾸는 옵션 |
|---|---|---|
| ① InfoNCE-Alt | ③ Dist-Alt | `infonce` → `gap_distance_infonce` |
| ① InfoNCE-Alt | ④ Center-Alt | `infonce` → `centered_infonce` |
| ① InfoNCE-Alt | ⑤ Dir-Alt | `infonce` → `gap_direction_infonce` |

각 비교는 alignment 50k + task 50k = total 100k update를 맞춘다.
모델·초기화 seed·데이터·`same_pair` batching·batch size·layer/pooling·dtype·LoRA·
LR/scheduler·checkpoint 선택 및 평가 규칙도 동일하게 둔다. Baseline에도
`same_pair`를 적용해야 loss와 배치 구성 변경의 효과가 섞이지 않는다.
Temperature는 사용하는 방법 간 동일하게 두며, ②에는 적용되지 않는다.
③의 추가 scale은 별도로 명시한다.

② Gap variance-Alt와의 비교는 별도 ablation에 둔다.

Retrieval R@1이나 slot-F1처럼 높을수록 좋은 지표 \(M\)의 핵심 개선량은
\(\Delta M_k=M(\text{loss }k\text{-Alt})-M(\text{InfoNCE-Alt})\),
\(k\in\{3,4,5\}\)다. InfoNCE-only 및 InfoNCE then SFT 결과도 함께 보고하되,
스케줄까지 다른 비교만으로 loss 교체의 효과를 결론 내리지 않는다.

### 추가 권장 비교군

구현된 다섯 loss 외에 다음 비교군을 고려할 수 있다. 아래 가중합과 별도
목적함수는 현재 선택 가능한 옵션이 아니다.

- **Joint weighted sum**:
  \(\mathcal{L}_{task}+\lambda\mathcal{L}_{NCE}\)
- Positive-pair cosine/MSE alignment
- Relational distance 또는 Gram-matrix alignment
- InfoNCE + proposed gap loss

Joint weighted-sum은 결합 방식의 영향을 추가로 확인할 수 있는 보조
baseline 후보이다. 현재 핵심 비교는 위의 InfoNCE-Alt와 제안 방법 Alt다.

---

## 4. Update budget과 공정한 비교

메인 결과는 **objective exposure-matched** 설정으로 구성한다.

| Method | Alignment updates | Task updates | Total updates | 현재 스크립트 |
|---|---:|---:|---:|---|
| Pretrained | 0 | 0 | 0 | 없음 |
| Task-only | 0 | 50k | 50k | `scripts/massive_transfer_only.sh transfer_only` |
| Alignment-only (A) | 50k | 0 | 50k | `scripts/massive_contrastive_only.sh contrastive_only` |
| Alignment → Task (A→T) | 50k | 50k | 100k | `scripts/massive_transfer_only.sh contrastive_then_transfer` |
| Alternating (Alt) | 50k | 50k | 100k | `scripts/massive_contrastive_only.sh alternative` |

Baseline의 A/A→T/Alt에는 ① InfoNCE를 사용하며, 제안 방법의 주 비교는
③④⑤의 Alt다. 보조 분석의 ② 및 제안 loss A/A→T에도 같은 budget 원칙을 적용한다.
위 숫자는 비교 프로토콜과 현재 큐의 budget이며, Python CLI의
`num_steps` 기본값은 모드와 무관하게 100k다.
직접 실행할 때 A/T의 50k는 명시해야 한다.

이 설정은 task를 사용하는 방법끼리 task update 50k를 맞추고, alignment를
사용하는 방법끼리 alignment update 50k를 맞춘다.

다만 total update 수는 B3/B4가 B1/B2의 두 배이므로 다음을 함께 보고한다.
같은 100k Alt라도 loss별 연산 비용은 다를 수 있어 실제 비용을 별도로 측정한다.

- 메인 표: objective exposure-matched 결과
- 부록: total-update/compute-matched 결과
- 모든 방법: wall-clock time, GPU-hours, peak memory

### Learning-rate schedule 주의

현재 Trainer 기본 scheduler는 total step을 기준으로 동작한다. 따라서
50k 방법과 100k 방법은 objective별 learning-rate trajectory가 다르다.

- 현재 sequential baseline은 Stage 2에서 optimizer/scheduler를 reset하지 않는다.
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

Python CLI 기본값과 현재 두 메인 큐 스크립트의 설정은 구분한다.

| 항목 | `config.py` 기본값 | `massive_contrastive_only.sh` / `massive_transfer_only.sh` 설정 |
|---|---|---|
| Alignment loss | `infonce` | ③④⑤ 순회; T는 첫 loss로 한 번만 실행 |
| Alignment batching | `mixed` | `same_pair` |
| Train / eval batch size | 32 / 16 | 16 / 16 |
| Layer / pooling | -1 / `last_token` | -1 / `last_token` |
| Temperature / gap scale | 0.05 / 1.0 | 0.05 / 1.0 (scale은 CLI 기본값) |
| Quantization compute dtype | `float16` | `bfloat16` |
| LoRA rank / alpha | 8 / 32 | 16 / 32 |
| Learning rate / warmup ratio | 1e-4 / 0.1 | 1e-4 / 0.1 |
| Save / eval interval | 500 / 2500 | 1000 / 2500 |

현재 파일의 설정을 과거 run에 소급하지 않는다. 예를 들어 9월 21일 보고서의
layer 8 실험과 9월 28일 보고서의 layer -1 실험은 별도 조건이다.

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

- 모든 방법에서 **raw 임베딩을 L2 정규화한 cosine**을 사용한다.
  ④의 centroid 제거나 ③⑤의 gap score는 공식 retrieval에 적용하지 않는다.
  저장 임베딩도 raw 표현이다. 학습 loss가 사용하는 batch 내 B개 후보와
  retrieval의 전체 N개 후보를 구분한다.
- Training parallel data와 완전히 분리된 fixed held-out set 사용
- 모든 방법에 동일한 candidate pool과 동일한 순서 사용
- Cosine 점수 내림차순으로 순위를 매기며, 정확한 동점은 candidate index
  오름차순으로 처리한다. 결과 JSON에 `tie_break: candidate_index_ascending`을 기록한다.
- 동점은 계산된 float32 점수의 정확한 일치 기준이다. 비교 실험에서는
  행렬 연산의 반올림 차이를 줄이도록 `retrieval_chunk_size`도 동일하게 유지한다.
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

### XNLI 확장 (계획, evaluator 미구현)

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

모델별로 아래 8개 행을 사용한다. 현재 표의 모델은 실제 run 이름 기준
Llama-3.2-1B-Instruct, Qwen3.5-2B, Qwen2.5-1.5B-Instruct다.
**핵심 대조군은 Alt. InfoNCE↔SFT이며 Ours 세 후보도 모두 Alt를 사용한다.**

1. Pretrained
2. Task-only SFT
3. InfoNCE-only
4. InfoNCE then SFT
5. Alt. InfoNCE↔SFT
6. Ours — Gap Distance InfoNCE
7. Ours — Centered Cosine InfoNCE
8. Ours — Gap Direction Cosine InfoNCE

모델별 독립 표 3개와 출처는 [Main table](reports/main_table_20260928/main_table.md)에 있다.
모델명(8행), Ours(3행), Method·평가 헤더를 병합하고,
Alignment → In/Out → R@1·R@5·MRR, Downstream → In/Out → Slot F1·EM으로 표시한다.
**2026-09-30T13:40:59+09:00** 재검토에서 test 평가 **52개·지표 파일208개**를 검증했다.
Main MD·CSV·Excel은 **24행 중 21행·210개 지표**이며, 두 Qwen의 기본 InfoNCE-only를 추가했다.
완료 baseline 후보 24개 run은 현재 비교 설정 일치 → same_pair → 최신 학습 run 순으로 선택했다.

[Analysis](reports/main_table_20260928/analysis.md)는 모델별 **Task-only 1개 +
contrastive-only/순차/Alt 각각 InfoNCE와 제안 3종**, **39/39행·390개 지표**를 모두 담는다.
Llama Gap Distance→SFT의 완료 재실험도 반영했다. Transfer-only는 alignment loss를
사용하지 않아 모델당 하나의 공통 SFT 행으로 둔다. **남은 빈칸은 Main의 Pretrained 3행뿐**이다.

[최신 평가 큐](reports/main_table_20260928/queue_verification.md)는
`queue-20260930_114350`이며 **신규 3/3 완료·exit code 0**, GPU 0·1 모두 `finished`다.
새 두 Qwen InfoNCE-only는 layer-1/BF16으로 제안 방법과 같다.
`†`가 남은 기존 InfoNCE(Llama-only 및 세 모델 순차/Alt)는 layer8/FP16이므로
제안 방법과 loss만 통제한 비교가 아니다. `—`는 미평가, 0.00은 측정값이다.
[재검토 기록](reports/main_table_20260928/result_audit.md)에 출처와 빈칸 사유를 남겼다.

| 평가 영역 | 주 지표 | 보조 지표 | 집계 방식 |
|---|---|---|---|
| Alignment: OPUS-100 검색 | Recall@1 | Recall@5, MRR | 언어쌍별 양방향 평균 후 언어쌍 macro |
| Downstream: MASSIVE slot filling | Slot micro-F1 | Exact Match | 언어별 지표를 계산한 뒤 언어 macro |

R@1/R@5는 정답 번역이 상위 1/5개에 포함된 비율, MRR은 정답 순위 역수의
평균이다. EM은 예측 slot multiset 전체가 정답과 일치한 발화 비율로,
slot 순서를 무시하고 중복은 반영한다. 모든 점수는 ×100으로 표시한다.
Loss 값, 거리 분산, positive cosine은 별도 진단 지표다.

현재는 후보 3개를 비교하는 표다. 최종 논문 Main table에서는 공통 validation
기준으로 Ours 하나를 선정하여 **모델별 baseline 5개 + Ours 1개**를 남긴다.
선정되지 않은 후보, ② Gap variance, 제안 loss의 순차 학습 및 A 실행은
ablation/분석으로 옮긴다. 이 test 결과표만으로 Ours를 확정하지 않는다.

표 작성 규칙:

- ③④⑤ Alt 각각의 지표를 **B4 InfoNCE-Alt 대비** 1:1 비교하고 개선량도 함께 보고
- 최소 3 seeds, 가능하면 5 seeds의 mean ± standard deviation
- 최고값 bold, 두 번째 값 underline
- Pretrained 및 모든 alignment-only(A) 방법의 task 결과는 ZS로 명시
- 동일한 최종 checkpoint로 task와 retrieval 평가
- Test set으로 checkpoint나 hyperparameter를 선택하지 않음
- 유의성 검정 방법과 seed를 appendix에 명시

---

## 8. 세부 결과표

7절과 같은 baseline 5개와 제안 후보 Alt 3개를 사용한다.
언어별로도 ③④⑤의 핵심 대조군은 B4 InfoNCE-Alt다.

### 8.1 언어별 downstream/transfer

| Method | en | ko | ja | es | In Macro | fr | de | it | Out Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained (ZS) |  |  |  |  |  |  |  |  |  |
| Task-only |  |  |  |  |  |  |  |  |  |
| B2 InfoNCE-only (ZS) |  |  |  |  |  |  |  |  |  |
| B3 InfoNCE then SFT |  |  |  |  |  |  |  |  |  |
| **B4 InfoNCE-Alt** |  |  |  |  |  |  |  |  |  |
| ③ Dist / Alt |  |  |  |  |  |  |  |  |  |
| ④ Center / Alt |  |  |  |  |  |  |  |  |  |
| ⑤ Dir / Alt |  |  |  |  |  |  |  |  |  |

### 8.2 언어별 bidirectional retrieval

각 A↔B 값은 A→B와 B→A의 평균이다. 방향별 수치는 appendix에 별도로
보고한다.

| Method | en↔ko | en↔ja | en↔es | In Macro | en↔fr | en↔de | en↔it | Out Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained |  |  |  |  |  |  |  |  |
| Task-only |  |  |  |  |  |  |  |  |
| B2 InfoNCE-only |  |  |  |  |  |  |  |  |
| B3 InfoNCE then SFT |  |  |  |  |  |  |  |  |
| **B4 InfoNCE-Alt** |  |  |  |  |  |  |  |  |
| ③ Dist / Alt |  |  |  |  |  |  |  |  |
| ④ Center / Alt |  |  |  |  |  |  |  |  |
| ⑤ Dir / Alt |  |  |  |  |  |  |  |  |

### 8.3 학습 비용

| Method | Trainable params | Alignment updates | Task updates | Total updates | GPU-hours | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| Task-only |  | 0 | 50k | 50k |  |  |
| B2 InfoNCE-only |  | 50k | 0 | 50k |  |  |
| B3 InfoNCE then SFT |  | 50k | 50k | 100k |  |  |
| B4 InfoNCE-Alt |  | 50k | 50k | 100k |  |  |
| 제안 후보 Alt (③④⑤ 각각) |  | 50k | 50k | 100k |  |  |

제안 loss의 A/A→T 비용은 해당 보조 실험의 표에 같은 항목으로 기록한다.

---

## 9. 구현된 변형 비교와 추가 ablation

**주 비교: InfoNCE-Alt 대비 제안 loss의 Alt.** 아래에서는 스케줄을 Alt로
고정한다. 여기서 loss 숫자의 크기가 아니라 retrieval/transfer 지표를 비교한다.

| 비교 | 확인할 질문 |
|---|---|
| ① Alt vs ③ Alt | 평균 거리 기반 InfoNCE가 raw cosine InfoNCE보다 유효한가? |
| ① Alt vs ④ Alt | 언어별 배치 평균을 제거한 cosine이 raw cosine보다 유효한가? |
| ① Alt vs ⑤ Alt | 공통 gap 방향 기반 InfoNCE가 raw cosine InfoNCE보다 유효한가? |

**보조 비교:** 같은 Alt에서 ① vs ②는 InfoNCE와 거리 분산을,
② vs ③은 거리 분산과 거리 기반 대조학습을,
③ vs ⑤는 길이와 방향 기준을 비교한다. 각 제안 loss의 A/A→T/Alt 비교는
스케줄 ablation으로 보고한다.

②~⑤는 모두 `same_pair` microbatch를 사용한다. ①과 비교할 때도 같은
batching을 지정하고 모델, layer/pooling, dtype, batch size, LoRA, update
budget을 맞춘다. Loss의 단위와 score 범위가 달라 raw loss 숫자끼리 우열을
매기지 않는다. ③의 scale과 temperature는 모두 logit 크기에 영향을 주므로
함께 기록한다. ②에는 temperature가 적용되지 않는다.

가중합(`InfoNCE + Gap`, `Task + InfoNCE + Gap`), positive-only cosine/MSE,
벡터 잔차 InfoNCE, teacher preservation, variance/covariance 보조항,
EMA centroid는 **추가 구현이 필요한 ablation**이다. 현재 Alt/A→T는
선택한 alignment loss와 task loss를 다른 update에서 사용하며 가중합하지 않는다.
Negative나 task supervision이 있다는 사실만으로 collapse 방지를 보장하지 않는다.

---

## 10. 현재 구현 상태

### 현재 동작하는 부분

- 네 학습 모드 routing
- 언어별/objective별 validation loss (`eval_language_scope=both`이면 13개 dataset)
- Validation 샘플 단위 입출력/loss 기록 (`eval_samples/step-N.json`)
- ① InfoNCE, ② Gap variance, ③ Gap Distance InfoNCE, ④ Centered InfoNCE, ⑤ Gap Direction InfoNCE 선택
- 전체 평가 거리 통계와 실제 샘플 수로 가중한 평가 loss
- 셔플 전 원본 ID 기반 학습/검증 샘플 기록
- MASSIVE downstream training
- LoRA + 4-bit quantization
- W&B training logging 설정
- Objective별 alignment_loss, downstream_loss
- Objective별 누적 update count
- Learning rate, gradient norm, 전체 training loss
- Package/CUDA/data-size/parameter-count metadata
- `save_steps`마다 checkpoint 저장 (CLI 기본 500, 현재 메인 큐 1000)
- LoRA adapter, experiment config, Trainer state 저장
- Checkpoint resume를 위한 adapter reload

### 학습 중 validation (구현됨)

언어별/objective별 validation loss를 `eval_steps`마다 측정한다.

- 기본 `eval_language_scope=both`는 MASSIVE 7언어(en, ko, ja, es, fr, de, it)와
  alignment 6쌍, 총 13개 dataset이다. `in`이면 MASSIVE 4언어와 alignment 3쌍이다.
- Alignment을 언어쌍별로 분리하는 이유는 InfoNCE가 in-batch negative를
  쓰기 때문이다. en-ko와 en-ja가 섞인 batch의 loss는 어느 언어쌍의
  정렬 품질도 아니라 혼합 negative pool에서의 난이도가 된다.
- `Trainer`가 dict eval_dataset을 지원하므로 metric은
  `eval_massive_out_de_loss`, `eval_align_in_en-ko_loss` 형태로 자동
  생성된다.
- `eval_on_start=True`이므로 step 0에 학습 시작 전 validation loss가 기록된다.
  Retrieval/slot-F1을 측정한 B0 평가를 대신하지 않으며, resume 시에는
  로드한 checkpoint 상태의 평가라는 점을 구분한다.
- OPUS-100 config는 두 언어 코드를 알파벳 순으로 이어 붙인 이름 하나만
  존재한다. anchor를 앞에 두면 anchor보다 앞서는 언어를 놓치므로
  (`en-de`는 없고 `de-en`이 있다) `opus_config_name`으로 방향을
  해석한다.

Validation은 loss만 측정한다. Retrieval R@1과 slot-F1은 비용이 크므로
학습 루프에 넣지 않고 `evaluate.py`로 저장된 checkpoint를 지정해 수행한다.
최종 checkpoint와 중간 checkpoint 모두 지정할 수 있다. Validation 비용은
실제 split 크기, batch size, 문장 길이와 모델에 따라 달라진다.

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

Selection에는 **in-language validation만** 사용한다. 기본 rule은 저장된 task의
`<task>_in`이며, contrastive-only에서는 `align_in`이다. 미학습 언어의 validation loss로 checkpoint를 고르면
gradient에는 out-language label을 쓰지 않았더라도 모델 선택 경로로
target-language 감독이 들어가 fully-unseen transfer 주장과 충돌한다.
`align_out` 역시 out-language parallel data를 selection에 쓰는 것이므로
같은 문제가 있다. out-language 그룹을 지정하면 경고가 출력된다.

| Method | 권장 rule |
|---|---|
| transfer_only, contrastive_then_transfer, alternative | `massive_in` |
| contrastive_only, ①③④⑤ | 동일 loss·배치 조건의 `align_in` 또는 사전에 고정한 final step |
| contrastive_only, ② gap_consistency | 사전에 고정한 final step 권장; 분산 최솟값은 의미 정렬을 보장하지 않음 |

fr/de/it validation/test는 selection과 hyperparameter tuning에 사용하지
않고 최종 평가에만 쓴다.

또한 `eval_steps=2500`에서 50k 방법은 validation 후보가 21개, 100k 방법은
41개다. 모든 step의 최솟값을 고르면 긴 방법이 더 많은 시행을 갖는
selection advantage가 생긴다. Final checkpoint를 쓰거나 objective exposure
기준의 동일한 selection grid를 사전에 확정해야 한다. 스크립트가 후보 수를
출력한다.

모든 validation step을 선택 후보로 삼으려면 `eval_steps`를 `save_steps`의
배수로 맞춘다. 현재 메인 큐의 2500/1000 조합에서는 2500, 7500 등의
평가 step에 checkpoint가 없다. 실제로 저장된 step 중에서 선택하거나 실행 전에
간격을 맞춘다. 아래 Python 예시는 2500/500을 사용한다.

`save_total_limit=None`이면 모든 checkpoint를 보관한다. 보관 개수를 제한하면
오프라인에서 선택하려는 과거 checkpoint가 이미 삭제되었을 수 있다.

### 현재 수행하지 않는 부분

- Retrieval / MASSIVE generation의 학습 중 측정 (독립 평가로 수행)
- XNLI evaluator
- InfoNCE + Gap 가중합, preservation 및 anti-collapse 보조 손실

---

## 11. 로깅 및 저장 구조

W&B:

- Project: Oct_ARR
- Run name:
  model__training_type__alignment_loss__in_languages__out_languages__timestamp
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
            ├── train_samples/
            │   ├── alignment-observations.jsonl
            │   └── alignment-batches.jsonl
            ├── eval_samples/
            │   ├── step-0.json
            │   ├── step-0_batches.json
            │   ├── step-0_metrics.json
            │   ├── step-0_embeddings.pkl
            │   └── ...
            ├── checkpoint-500/
            ├── checkpoint-1000/
            └── ...

`eval_samples/step-N.json`은 `"<objective>/<언어>"`를 키로 하고, 각
샘플의 입력/정답/loss를 담는다. MASSIVE는 `utt`, `target`, `loss`,
`num_target_tokens`를, alignment는 `source_text`, `target_text`,
`loss`, `positive_cosine`을 기록한다.

고정 seed로 정해진 각 언어의 앞쪽 N개를 매 평가 round에 기록한다.
①은 기존 순차 배치를, ②③④⑤는 단일 pair 평가 배치를 사용한다.
②③④⑤의 마지막 singleton은 앞 배치에 합쳐 샘플 누락 없이 최소 2개를
보장하므로 실제 배치가 설정값보다 1개 클 수 있다.

`--train_sample_log_interval`은 alignment optimizer update 기준 상세
기록 간격(기본 1000), `--train_sample_log_limit`은 해당 update의 각
microbatch당 기록 개수(기본 8)다. 어느 쪽이든 0이면 학습 상세 기록을
끈다. 요약은 `logging_steps`마다 계속 기록한다. 학습 상세 기록은 실제
학습 forward의 텍스트/수치/토큰 해시이며 추가 forward나 임베딩 파일은
만들지 않는다. 임베딩과 `embedding_keys`는 고정 검증/독립 평가 표본에
저장한다. Dropout이 활성화된 학습 관측과 고정 검증 관측을 구분한다.

Alignment `sample_id`는 데이터셋/config/실제 split/셔플 전 원본 행 번호다.
MASSIVE는 native ID와 locale을 사용한다. 원본 fingerprint는 metadata에
기록한다. `record_id`에는 session/step/microbatch/샘플 위치가 들어가며,
`batch_id`로 전체 배치 구성원 ID 및 평균 거리를 찾을 수 있다.
Alignment 입력 토큰 수와 해시는 padding을 제외한 실제 모델 입력 기준이다.

Alignment 상세 기록의 `per_sample_loss`는 선택한 목적함수의 샘플별 값이다.
①③④⑤는 양방향 CE의 샘플별 평균, ②는 평균 거리에서의 제곱 편차다.
전체 평가 통계는 상세 표본 개수와 독립적이며 거리 scalar는 float64로 집계한다.
다음 값은 구분해야 한다.

- `selected_loss_mean`: 실제 선택한 alignment 목적함수의 평균.
- `batch_gap_loss_mean`: 배치별 **raw 거리 분산**을 실제 샘플 수로 가중 평균.
  ③④⑤를 실행해도 이 진단값이 InfoNCE loss로 바뀌지는 않는다.
- `corpus_gap_distance_variance`: 동일한 평가 모델로 전체 언어쌍의 거리
  평균을 구한 뒤 계산한 분산. 배치 사이 평균 차이까지 포함한다.
- `positive_cosine` / `positive_cosine_mean`: raw 정답 임베딩 간 cosine.
  ④의 centered cosine이나 ⑤의 gap 방향 score가 아니다.

Metadata의 `gap_reference`는 ① `null`, ②③ `microbatch_mean_distance`,
④ `microbatch_language_centroids`, ⑤ `microbatch_mean_gap_vector`다.
`gap_embedding_space` 등 거리 진단의 공간 표기와 loss별 기준을 함께 확인한다.
④⑤에서도 raw 거리 통계를 기록하므로 loss 그래프와 거리 분산 그래프를 구분한다.

`step-N_metrics.json`에는 전체 평가 통계를 저장한다. W&B에는 수치만
전달하고 원문/표본/벡터는 로컬에 저장한다. 학습 구간은 모델이 계속
변하므로 corpus 분산으로 표시하지 않는다. ②③④⑤는 단일 프로세스/GPU만
지원한다. 기존 mixed InfoNCE의 다중 프로세스 거리 진단은 rank별 local
관측으로 명시하며 전체 데이터 통계로 해석하지 않는다.

각 checkpoint에는 다음이 저장된다.

- LoRA adapter
- Adapter config
- Optimizer/Scheduler/RNG state는 `save_only_model=False`일 때만 저장
- Trainer state
- Tokenizer
- Experiment config

Base model은 model_name으로 다시 로드할 수 있으므로 4-bit base weight
전체를 매 checkpoint마다 중복 저장하지 않는다.

현재 `main.py`는 `save_only_model=True`를 설정한다. 따라서 현재 설정으로
만든 checkpoint에는 optimizer/scheduler/RNG state가 없으며, adapter 로드만으로
학습 상태 전체가 복원되지는 않는다. 완전한 resume에는 해당 상태까지 저장한
checkpoint가 필요하다.

wandb login은 실행 스크립트마다 호출하지 않고 서버 계정에서 한 번만
수행한다. API key를 shell script에 기록하지 않는다.

### Inference 샘플과 임베딩 저장

`evaluate.py`도 기본적으로 task·언어(또는 언어 pair)별 앞쪽 64개 샘플을
저장한다. `--eval_sample_log_limit N`으로 개수를 바꾸고, `0`으로 끈다.
샘플 저장 개수와 관계없이 retrieval과 MASSIVE 성능은 전체 평가 데이터로
계산한다. 예를 들어:

```bash
python evaluate.py --checkpoint_path /path/to/checkpoint \
  --split test --language_scope both --tasks alignment massive \
  --eval_sample_log_limit 64
```

기본 저장 위치는 `CHECKPOINT/evaluations/<validation|test>/<in|out>/`이다.

- `eval_samples.json`: 학습 로그와 같은 `alignment/en-ko`, `downstream/en`
  그룹별 샘플 기록. 원본 데이터, 텍스트, sample ID, `embedding_keys`를 보존한다.
- `eval_samples_embeddings.pkl`: 각 key에 대응하는 float32 NumPy 벡터.
- `alignment_metrics.json`: retrieval 및 전체 거리 진단(`distance_diagnostics`).
- `alignment_samples.jsonl`: 전체 alignment scalar 관측.
- `alignment_batches.jsonl`: 해당 관측의 전체 배치 구성원과 평균 거리.
  `--no-save_alignment_sample_metrics`로 두 JSONL을 끌 수 있다.
  상세 임베딩 제한이 0이어도 전체 평가 거리 통계는 계산한다.
  JSONL을 끈 경우에도 저장한 표본에는 `batch_sample_ids`로 배치 구성을 남긴다.
- `massive_predictions.jsonl`: 전체 생성 결과. 임베딩을 저장한 샘플에는
  JSON과 동일한 `sample_id`, `embedding_keys`도 들어간다.

Alignment는 source/target 각각의 선택 layer와 마지막 layer를 저장한다.
MASSIVE는 정답을 제외한 두 종류의 입력에서 각각 두 layer를 저장한다.

| 이름 | 입력 | 분석 용도 |
|---|---|---|
| `utt_embeddings`, `utt_last_layer_embeddings` | 발화 원문, alignment와 같은 tokenization 설정 | 언어·의미 구조와 alignment 문장 비교 |
| `prompt_embeddings`, `prompt_last_layer_embeddings` | 실제 생성 prompt의 토큰, 정답 제외 | task 지시문을 포함한 표현 분석 |

`embedding_inputs`에 입력 종류를 명시한다. 학습 validation의 `utt_embeddings`는
정답을 포함한 입력에서 추출하므로, inference의 원문 표현과 같은 조건으로
취급하면 안 된다. MASSIVE inference 샘플에는 생성 결과와 정답을 기록하며,
정답을 입력한 별도 forward나 teacher-forcing loss 계산은 하지 않는다.

MASSIVE의 저장 대상 배치에만 원문/prompt 추출용 forward를 각각 추가한다.
생성용 left padding은 유지하고, prompt 표현 추출에는 동일한 실제 토큰을
right padding으로 재배치하여 token 위치와 pooling을 맞춘다.

샘플 번호는 배치를 넘어 그룹별로 증가하며, 저장 후 JSON·임베딩 버퍼를
모두 비운다. 기존의 잘못 연결된 학습 로그는 이 수정으로 복원되지 않으므로
필요하면 해당 checkpoint에서 다시 추출해야 한다.

`--save_alignment_embeddings`는 이 샘플 저장과 별개로 **전체** retrieval
source/target 텐서를 `alignment_embeddings.pt`에 저장하는 기존 옵션이다.
시각화에서 원본 ID·intent·slot·언어·생성 결과를 사용해 색상이나 연결선을
구성할 수 있다. 앞쪽 N개는 고정 표본이며 전체 데이터의 대표성을 보장하지 않는다.

### 여러 run 순차 평가

`scripts/eval.sh`의 `run_paths` 배열에 `results/` 아래의
`모델 폴더/run 이름`을 입력하면 나열한 순서대로 평가한다. run 폴더의 최종
모델을 사용하며, 특정 step을 평가하려면 경로 뒤에 `/checkpoint-N`을 붙인다.
목록 전체의 config와 adapter 파일을 먼저 확인하고, 평가 중 오류가 나면 중단한다.

```bash
conda activate octarr
bash scripts/eval.sh --dry-run  # 경로 확인 및 실행 명령 출력
bash scripts/eval.sh            # 실제 순차 평가
```

기본값은 GPU 0, test split, in/out 언어 모두, alignment와 MASSIVE 평가,
task·언어별 샘플 64개 저장이다. 스크립트 상단에서 설정을 바꿀 수 있다.
`CUDA_VISIBLE_DEVICES=1 bash scripts/eval.sh`로 GPU를 지정할 수도 있다.
결과는 각 평가 대상 폴더의 `evaluations/<split>/<in|out>/`에 저장된다.
명령행에 경로들을 전달하면 `run_paths` 배열 대신 사용한다.

---

## 12. 실행

프로젝트 루트에서 실행한다. 직접 Python을 실행할 때는 의존성을 설치한
`octarr` 환경을 활성화한다. 두 메인 큐는 기본적으로
`/home/nlplab/anaconda3/envs/octarr/bin/python`을 사용하며, 다른 환경에서는
`PYTHON_BIN`으로 실행 파일 경로를 지정한다.

    conda activate octarr
    mkdir -p logs
    nohup bash scripts/massive_transfer_only.sh    > logs/transfer_only.log 2>&1 &      # GPU 0
    nohup bash scripts/massive_contrastive_only.sh > logs/contrastive_only.log 2>&1 &   # GPU 1

**2026-09-30 현재 활성화된 배열** 기준이다. 주석 처리된 모델/모드는 제외한다.
기존 `contrastive_only.sh` / `transfer_only.sh`는 각각 아래 `massive_*` 이름으로
변경했다. `--downstream_task massive`를 명시하며 기존 학습 설정은 유지한다.

| 스크립트 | 기본 모델 | 기본 모드 | 기본 loss | 실행 수 |
|---|---|---|---|---:|
| `massive_contrastive_only.sh` | Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct, Qwen3.5-2B | A | ① InfoNCE, ④ Centered InfoNCE | 6 |
| `massive_transfer_only.sh` | Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct, Qwen3.5-2B | A→T | ① InfoNCE, ④ Centered InfoNCE | 6 |

`massive_transfer_only.sh transfer_only`로 task-only 모드를 지정할 수 있다.
스크립트 파일명과 달리 인자 없는 기본 실행은 현재 `contrastive_then_transfer`다.
`ALIGNMENT_LOSS`는 7가지 옵션 중 하나로 제한할 때 쓴다.
InfoNCE-Alt는 `ALIGNMENT_LOSS=infonce`와 `alternative` 모드를 지정한다.
다음은 학습 없이 실제 실행 명령만 확인하는 예시다.

```bash
DRY_RUN=1 bash scripts/massive_contrastive_only.sh
DRY_RUN=1 bash scripts/massive_transfer_only.sh contrastive_then_transfer
ALIGNMENT_LOSS=infonce DRY_RUN=1 bash scripts/massive_contrastive_only.sh alternative
ALIGNMENT_LOSS=centered_infonce DRY_RUN=1 bash scripts/massive_contrastive_only.sh alternative
```

`alternative.sh`와 `contrastive_then_transfer.sh`는 별도 단일 모델용
스크립트다. 두 메인 큐와 LoRA/dtype 등의 설정이 다를 수 있으므로 같은 조건의
실험에는 위 큐의 모드 인자 또는 아래 Python 명령을 사용한다.

### Alignment 손실 선택

```bash
python main.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --training_type contrastive_only --num_steps 50000 \
  --alignment_loss centered_infonce --alignment_batching same_pair \
  --batch_size 16 --eval_batch_size 16 \
  --alignment_hidden_state_layer -1 --alignment_hidden_state_position last_token \
  --alignment_temperature 0.05 --alignment_gap_scale 1.0 \
  --learning_rate 1e-4 --lr_scheduler_type linear --warmup_ratio 0.1 \
  --peft_lora_r 16 --peft_lora_alpha 32 \
  --quantization_compute_dtype bfloat16 --training_seed 42 \
  --eval_steps 2500 --save_steps 500 \
  --train_sample_log_interval 1000 --train_sample_log_limit 8
```

CLI 기본값은 기존 실험 호환을 위해 `infonce`다. 학습 스크립트는
`ALIGNMENT_LOSS=gap_consistency bash scripts/massive_contrastive_only.sh`처럼
선택하며 run name에 손실 종류를 포함한다. 위 Python 명령의 loss를 ①~⑤ 또는
`gap_distance_detach`, `gap_distance_rms`로 바꾸면 된다. **②만 temperature를 사용하지 않으며 ③④⑤는 사용한다.**
`alignment_gap_scale`은 ③ 및 두 Gap Distance 변형에 적용한다. 현재 SH에는 scale 전용 환경변수가
없으므로 scale을 바꾸는 실험은 Python CLI에서 지정한다.
`transfer_only`는 선택한 alignment 손실로 학습하지 않고, 검증에서만
사용한다. 독립 평가는 checkpoint 설정을 읽는다. 필드가 없는 이전
checkpoint는 InfoNCE로 해석하고, 다른 목적함수로 resume하면 오류를 낸다.

② Gap-only의 분산 최솟값을 자동으로 최선의 의미 정렬로 해석하지 않는다.
최종 step 또는 사전에 정한 task in-language 기준으로 checkpoint를 선택하고,
손실 종류가 다른 실행의 alignment loss를 직접 순위 비교하지 않는다.

### Alignment 배치의 언어쌍 구성

`main.py` 실행 인자에 `--alignment_batching same_pair`를 추가하면 각
alignment 배치가 하나의 언어쌍으로만 구성된다. 기본값 `mixed`는 기존
DataLoader 경로를 사용한다. 두 메인 큐는 `ALIGNMENT_BATCHING` 환경변수를
지원하며 기본값이 `same_pair`다. ②③④⑤를 학습할 때 `mixed`는 허용하지 않는다.

- Trainer의 `objective_at(step)`을 sampler와 공유하여 네 학습 모드의
  스케줄을 동일하게 적용하고, 학습 시 실제 배치와의 일치 여부를 검사한다.
- Alignment update 기준으로 언어쌍을 균등 순환하며, 주기마다 pair 순서를
  shuffle한다. Gradient accumulation 안에서는 같은 pair를 유지한다.
- 각 pair 내부에서 shuffle 후 완전한 배치만 사용한다. 배치보다 작은
  데이터 풀은 오류로 처리하고, 마지막 불완전한 배치는 버린다.
- MASSIVE는 독립적으로 샘플링한다. 같은 seed/batch/accumulation 설정의
  `same_pair` 실행끼리는 objective별 샘플 순서를 맞출 수 있다. 기존
  `mixed` 실행과 샘플 순서까지 같은 것은 아니다.
- Validation은 objective/언어별 데이터셋을 사용하며, ②③④⑤에서는 마지막
  singleton을 앞 배치에 합치는 평가 sampler를 적용한다.
- 현재 `same_pair`는 단일 프로세스/GPU에서 지원한다. GPU별로 독립적인
  실험을 실행할 수 있으며 DDP는 지원하지 않는다.
- 샘플러는 재개 시 같은 계획을 재생하고 Trainer가 소비한 배치를 건너뛴다.
  Optimizer/scheduler/RNG 상태까지 복원하려면 별도로 해당 상태가 저장된
  checkpoint가 필요하다.

CPU 회귀 테스트: `python -m unittest discover -s tests -v`

### num_steps와 objective update 수는 다르다

1 step은 **하나의 objective**에 대한 1 optimizer update이다. 따라서
objective exposure를 맞추려면 method마다 num_steps가 달라야 한다.

| Method | num_steps | alignment | task | 스크립트 |
|---|---:|---:|---:|---|
| transfer_only | 50k | 0 | 50k | massive_transfer_only.sh |
| contrastive_only | 50k | 50k | 0 | massive_contrastive_only.sh |
| contrastive_then_transfer | **100k** | 50k | 50k | massive_transfer_only.sh |
| alternative | **100k** | 50k | 50k | massive_contrastive_only.sh |

`contrastive_then_transfer`와 `alternative`는 두 objective를 모두 쓰므로
num_steps가 두 배여야 한다. 이 값은 `planned_objective_updates`와
`objective_update_counts`가 계산하며 `run_metadata.json`에 기록되므로
실행 후 반드시 확인한다.

### Learning rate와 scheduler

`--learning_rate`, `--lr_scheduler_type`, `--warmup_ratio`로 제어한다.
현재 기본값은 **`1e-4` / `linear` / `0.1`**이며 두 메인 큐도 같은 값을 쓴다.
`main.py`는 `int(warmup_ratio * num_steps)`를 계산해 `warmup_steps`로
전달한다. 따라서 50k 실행은 5k, 100k 실행은 10k warmup step이다.
비율로 받는 이유는 방법마다
`num_steps`가 50k/100k로 달라 고정 step이 서로 다른 비율이 되기
때문이다.

과거 1e-3/no-warmup 실험이나 진단 스크립트의 값을 현재 기본값으로
해석하지 않는다. LoRA rank/alpha와 target module도 실험별 config로 확인한다.

#### Scheduler 선택이 objective 예산에 미치는 영향

전역 scheduler를 사용하므로 update 수가 같아도 objective가 경험하는 LR은
달라질 수 있다.

| 스케줄 | 각 objective가 경험하는 구간 |
|---|---|
| A / T | 하나의 objective가 warmup부터 decay 끝까지 경험 |
| A→T | A가 warmup과 전반 decay, T가 후반 decay를 경험; 전환 시 reset 없음 |
| Alt | 두 objective가 전체 warmup/decay 구간에 교대로 분포 |

현재 warmup 0.1에서 과거 no-warmup 기준 평균 LR 수치를 재사용하지 않는다.
스케줄 영향을 분리하려면 `--lr_scheduler_type constant --warmup_ratio 0`
조건을 추가로 비교할 수 있다. Stage 2 optimizer/scheduler reset은 별도
구현이 필요한 실험이며 현재 A→T에 적용되어 있지 않다.

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

① InfoNCE의 `contrastive_only`를 1e-3 / 5e-4 / 2e-4로 2000 step씩 돌린다.
현재 진단 스크립트는 Qwen2.5-3B, layer 8, FP16, warmup 0.1 조건이며
③④⑤를 순회하지 않는다. 메인 큐의 layer -1/BF16 조건과 구분한다.
동일한 alignment 개선 정도에서 LR에 따라 downstream loss 악화가 달라지는지
확인하는 용도다. 이 비교만으로 특정 목적함수가 일반적으로 task 능력을
훼손하거나 보존한다고 결론 내리지 않는다.

### Layer probe

    nohup bash scripts/layer_probe.sh > logs/layer_probe.log 2>&1 &
    python3 scripts/compare_runs.py results_layer_probe --step 0

`eval_on_start`가 첫 optimizer step 전에 평가하므로 step 0에서 사전학습
표현을 읽을 수 있다. 스크립트 자체는 설정마다 1 training step도 수행한다.
Qwen2.5-3B의 layer 8 / 18 / 27 / -1과 last_token / mean을 조합해
① InfoNCE의 step-0 `eval_align_in_*`을 비교한다. 현재 CLI/메인 큐 기본
layer는 **-1**이다. 모든 score가 같은 기준의 loss는 `ln(B)`지만, 이 값보다
크다는 사실만으로 의미 정보가 없다고 단정하지 않는다. Retrieval도 함께 보고
layer를 선택한 뒤 비교 방법들에 동일하게 적용한다.

두 스크립트 모두 `--eval_language_scope in`을 쓴다. out-language 데이터는
selection과 tuning에 쓰지 않고 최종 평가에만 사용한다.

---

## 13. 논문 결과 생성 전 체크리스트

- [ ] Pretrained / Task-only SFT / InfoNCE-only / InfoNCE then SFT / InfoNCE-Alt baseline 결과 확보
- [ ] InfoNCE-Alt와 제안 후보 ③④⑤ Alt의 조건을 맞춰 1:1 비교
- [ ] 공통 validation 기준으로 Ours 하나 선정; 나머지 방법은 ablation/분석에 배치
- [ ] Sequential baseline의 optimizer/scheduler reset 정책 확정
- [ ] 모든 baseline에 동일한 layer/pooling 적용
- [ ] 실제 로드된 alignment pair와 pair별 sample 수 확인
- [ ] Train/validation/test parallel data leakage 검사
- [x] Full-corpus candidate pool을 사용하는 retrieval 구현
- [x] A→B/B→A 양방향 retrieval 구현
- [x] MASSIVE slot micro-F1 및 language macro 구현
- [ ] XNLI accuracy evaluator 구현
- [ ] Final checkpoint 또는 validation-selected checkpoint 정책 사전 확정
- [ ] 최소 3개 seed 실행
- [ ] Exposure-matched와 compute-matched 결과 분리
- [ ] GPU-hours와 peak memory 기록
- [x] Gap 거리 분산, 단일 pair 배치, gradient 및 로깅 검증
- [x] ③ Gap Distance / ④ Centered / ⑤ Gap Direction InfoNCE와 기존 경로 회귀 검증
- [ ] Gap 분산 감소와 의미 대응/representation collapse 관계의 실험 검증

## 14. WMT25 downstream MT

`--downstream_task wmt25`는 decoder-only 번역 fine-tuning을 선택한다.
OPUS-100 alignment와 MASSIVE 경로는 기존 설정을 사용한다. WMT 실행 스크립트의
새 기본값은 **공식 recipe 전체 + 여러 목표 언어를 섞은 downstream 배치**다.

### 데이터 출처와 분할

WMT25는 하나의 HF train/test 코퍼스가 아니다.
[공식 학습 recipe](https://www2.statmt.org/wmt25/mtdata/)의
`wmt25-eng-kor`, `wmt25-eng-jpn`, `wmt25-eng-ces`에 나열된 모든 병렬
`train` 리소스를 사용한다. `mono_train`은 사용하지 않는다. TED도 이 recipe의
일부이지만 TED만 선택하지 않는다. 실제 corpus 목록과 입력 SHA-256은 manifest에 남는다.

| 용도 | 방향 | 원천과 분할 |
|---|---|---|
| Train | EN→KO/JA/CS | 각 공식 recipe의 병렬 학습 자료에서 검증·평가 원문 제외 |
| In validation | EN→KO/JA/CS | 위 학습 자료에서 언어별 500개 분리 |
| Out validation | EN→ET/RU/AR | WMT25 평가 자료에서 공통 영어 원문 20개 분리 |
| In test | EN→KO/JA/CS | WMT25 평가 문서에서 validation 원문 제외 |
| Out test | EN→ET/RU/AR | WMT25 평가 문서에서 validation 원문 제외 |

목표 언어 partition은 `training_langs=("ko", "ja", "cs")`,
`out_inference_langs=("et", "ru", "ar")`다. 입력은 항상 영어이며 EN→EN 학습은 하지 않는다.
스크립트의 OPUS alignment 언어도 같은 partition을 사용한다.

공식 평가 원문은 모든 언어의 downstream train 및 in validation에서 제외한다.
In validation 원문도 세 언어의 train 전체에서 제외한다. Out validation으로 선택한
20개 원문은 seen/unseen 전체 test에서 제거한다. 따라서 최종 test는 공식 평가 자료의
사용자 정의 holdout subset이며, 전체 공식 testset 결과라고 표기하지 않는다.
Out validation은 진단용이며 checkpoint 선택에는 사용하지 않는다.
OPUS-100과 downstream 사이의 cross-dataset overlap 제거는 별도로 구현하지 않았다.

Recipe 전체 사용은 학습 출처를 확대하는 변경이다. 학습 데이터를 문서 단위로 바꾸거나
학습·평가 도메인을 같게 만드는 변경은 아니다. In validation도 학습 코퍼스에서 분리한
데이터이므로 최종 WMT25 문서 평가와 분포가 다를 수 있다.

### 데이터 준비

준비는 학습과 별도로 실행한다. Python 3.9–3.11에서:

```bash
python -m pip install mtdata==0.4.3 PyYAML sacrebleu==2.6.0

# 작은 recipe만 읽어 대상 corpus 목록 확인. 데이터/출력 파일을 만들지 않는다.
python scripts/prepare_wmt25.py \
  --plan --corpus_profile full_recipe \
  --mtdata_dir data/wmt25_raw \
  --output_dir data/wmt25_full_recipe_et_ru_ar_seed42

# 전체 병렬 corpus 다운로드 및 준비. 대규모 저장 공간과 전처리 시간이 필요하다.
python scripts/prepare_wmt25.py \
  --download --corpus_profile full_recipe \
  --mtdata_dir data/wmt25_raw \
  --output_dir data/wmt25_full_recipe_et_ru_ar_seed42 \
  --seed 42 --validation_per_language 500 --out_validation_per_language 20
```

이미 mtdata로 추출했다면 `--download`를 빼고 기존 `--mtdata_dir`를 지정한다.
원본은 `wmt25-eng-{kor,jpn,ces}/train-parts/<corpus-id>.<lang>[.gz]` 구조다.
영어 파일은 suffix로 결정하므로 `ces-eng`와 locale suffix가 있는 리소스도 방향을 유지한다.
UTF-8, 병렬 파일 행 수, 빈 텍스트 및 NULL 표기를 검사한다. 일부 corpus 다운로드나
파일 검증이 실패하면 중단하며, TED로 대체하거나 누락 corpus를 조용히 제외하지 않는다.

출력 디렉터리는 비어 있어야 한다. 기존 TED 디렉터리/manifest를 이름만 바꾸지 않는다.
결과는 `train.{ko,ja,cs}.jsonl`, `validation.{ko,ja,cs,et,ru,ar}.jsonl`,
`test.{ko,ja,cs,et,ru,ar}.jsonl`, `manifest.json`이다.
Manifest에는 profile, 전체 recipe 목록, 선택한 corpus 목록, 입력 hash, seed,
분할별 개수 및 reference 수가 기록된다. 학습 pool 상한은 없다.

평가 입력은 `wmt-conference/wmt25-general-mt` revision
`56c0a513f64ba63500e222b25bf87ac2201cb1eb`의 `data/wmt25-genmt.jsonl`에 고정한다.
`--recipe_file` / `--eval_file`로 로컬 원본을 지정할 수도 있다.

### 혼합 배치와 샘플별 프롬프트

새 학습 설정은 `--wmt25_downstream_sampling proportional`이다.
KO/JA/CS 전체 pool의 인덱스를 함께 셔플하여 배치를 구성한다. 따라서 언어 비중은
준비된 행 수에 비례하며, 매 배치에 세 언어가 모두 포함되거나 정확히 같은 비율로
포함된다고 보장하지 않는다. MASSIVE의 전체 pool 비례 샘플링과 같은 규칙이다.

각 인덱스로부터 언어와 원문·정답을 함께 가져온 뒤 다음 프롬프트를 렌더링한다.

```text
System: Translate the following sentences from English to Korean.
User: I will meet you tomorrow.
Assistant: 내일 만나요.
```

다음 샘플의 목표 언어가 일본어라면 그 샘플의 system 언어명은 Japanese가 된다.
배치 전체에 하나의 언어명을 재사용하지 않는다. 각 행의 `tgt_lang`과 파일 언어를
검증하고, system/user 토큰과 padding은 `-100`으로 마스킹한다.
생성 평가에는 참조 번역을 프롬프트에 넣지 않는다.

`language_balanced` 옵션은 기존 비교 실험 재현용이다. 이 경우 한 optimizer update의
모든 microbatch가 하나의 목표 언어를 사용하고, 3 downstream update마다 각 언어를
한 번씩 선택한다. 이전 checkpoint에 sampling 옵션이 없으면 기존 동작으로 해석한다.

### 실행과 학습 예산

두 WMT 스크립트는 다음 기본값을 전달한다.

- `WMT25_DATA_DIR=data/wmt25_full_recipe_et_ru_ar_seed42`
- `WMT25_CORPUS_PROFILE=full_recipe`
- `WMT25_DOWNSTREAM_SAMPLING=proportional`

요청한 profile과 manifest가 다르면 모델 로딩 전에 실패한다.
Legacy TED 실험을 명시적으로 재현할 때만 `WMT25_CORPUS_PROFILE=ted`와
해당 데이터 디렉터리를 함께 지정한다. 새 run 이름에는 corpus profile과 sampler가
포함되며 `run_metadata.json`에도 실제 설정을 기록한다.

```bash
DRY_RUN=1 bash scripts/wmt25_transfer_only.sh transfer_only
DRY_RUN=1 bash scripts/wmt25_contrastive_only.sh alternative

bash scripts/wmt25_transfer_only.sh transfer_only
ALIGNMENT_LOSS=centered_infonce bash scripts/wmt25_contrastive_only.sh alternative
ALIGNMENT_LOSS=centered_infonce bash scripts/wmt25_transfer_only.sh contrastive_then_transfer
```

인자가 없으면 transfer 스크립트는 `transfer_only`와 `contrastive_then_transfer`,
contrastive 스크립트는 `contrastive_only`와 `alternative`를 실행한다.

| Mode | Alignment updates | MT updates | Total updates |
|---|---:|---:|---:|
| contrastive_only | 50,000 | 0 | 50,000 |
| transfer_only | 0 | 50,000 | 50,000 |
| alternative | 50,000 | 50,000 | 100,000 |
| contrastive_then_transfer | 50,000 | 50,000 | 100,000 |

데이터 교체가 step budget을 늘리지는 않는다. Batch 16, accumulation 1, 단일 GPU에서
50,000 MT updates는 800,000개 샘플 노출이다. 전체 recipe를 후보 pool로 사용한다는
뜻이며, 이 예산으로 모든 행을 한 번 이상 학습한다는 뜻은 아니다.
언어 혼합 여부도 처리 토큰 수나 FLOPs의 동일성을 보장하지 않는다.
Alignment의 `same_pair` 배치와 InfoNCE negative 구성은 그대로다.

### 평가

`scripts/wmt25_eval.sh`는 `results_wmt25`의 checkpoint를 대상으로
`alignment wmt25`, chrF, 생성 batch 1, 최대 신규 토큰 16384를 기본 사용한다.

```bash
bash scripts/wmt25_eval.sh --dry-run MODEL_FOLDER/RUN_NAME
bash scripts/wmt25_eval.sh MODEL_FOLDER/RUN_NAME

python evaluate.py --checkpoint_path CHECKPOINT \
  --tasks wmt25 --split test --language_scope both \
  --wmt25_batch_size 1 --wmt25_max_new_tokens 16384 --wmt25_metric chrf
```

`evaluate.py`를 직접 사용할 때 metric 기본값은 여전히 `none`이므로
점수가 필요하면 `--wmt25_metric chrf`를 지정한다. ET/RU/AR를 포함한 현재 평가
언어는 공식 `refA`를 사용한다. 시스템 출력을 reference로 대체하지 않는다.
입력은 자르지 않으며 context 초과, 출력 길이 제한 도달, 문단 수 불일치를 확인한다.
학습 중 downstream loss 평가와 생성 번역의 chrF 평가는 별도다.

`scripts/select_checkpoint.py RUN_DIR --rule wmt25_in`은 seen validation loss로
checkpoint를 선택한다. 분할 정의, 학습 corpus hash, sampling, checkpoint 선택 규칙을
논문과 함께 기록한다.

검증은 `test_prepare_wmt25.py`, `test_wmt25_dataset.py`,
`test_pair_batching.py`에서 recipe 전체 선택, 분할 간 원문 배제, profile 오류,
혼합 배치의 행별 언어 프롬프트/정답/masking, 학습 스케줄 재현을 검사한다.
작은 CPU fixture 검증은 전체 corpus 다운로드 및 실제 GPU 학습 검증과 별도다.

### Legacy WMT23 recipe profile (EN → DE/HE/JA)

이전 recipe 실험의 profile은 `accessible_parallel`이다. 이 profile의 논문 표기는
**“accessible subsets of the WMT23 parallel-training recipes”**라고 명시한다.
DE/JA는 지정 recipe의 모든 병렬 source를 사용하고, HE에서는 다음 두 source만 제외한다.

- `Neulab-tedtalks_train-1-eng-heb`: 기존 URL이 아카이브 대신 HTML을 반환.
- `ELRC-wikipedia_health-1-eng-heb`: direct ELRC 서버 인증서 만료.

나머지 OPUS ELRC source는 유지하지만 direct ELRC와 동일 파일이라고 가정하지 않는다.
이외의 다운로드 실패는 자동 제외하지 않고 오류로 처리한다. Source 수는 DE 13, HE 20,
JA 7이며, source별 원래 ID·실제 ID·제외 사유와 파일 SHA256을 manifest에 기록한다.
기존 News Commentary v16 → v18.1 변경도 manifest에 명시한다.

각 언어의 모든 유효 병렬 행을 merge한 뒤, 양쪽 문자열의 앞뒤 공백만 제거한
`(English source, target)` 쌍이 완전히 같은 중복을 제거한다. 같은 원문에 서로 다른
번역이 붙은 행은 유지한다. Corpus ID 순으로 처음 나온 행의 ID를 유지하며,
JSON으로 이스케이프한 source/target의 `LC_ALL=C` 정렬 순서를 고정한다.
대규모 중복 제거에는 메모리 제한이 있는 외부 sort를 사용한다. 그 후 seed 42로
학습 언어별 validation 500개를 추출하고, validation/test 영어 원문과 정확히 겹치는
행을 학습에서 제외한다. 공식 WMT23 test에는 dedupe나 도메인 필터를 적용하지 않는다.

WMT SFT와 OPUS-100 alignment의 학습 언어는 모두 **DE/HE/JA**다.
**ZH/RU/UK의 WMT train 및 OPUS-100 train은 adaptation에 사용하지 않는다.**
OPUS의 이 세 언어 validation/test는 평가 전용이며, checkpoint 선택에는
seen-language `wmt23_in`만 사용한다. OPUS train은 고정 revision의 로컬 파일에서
평가 원문을 제외한 뒤 언어쌍마다 10,000개를 추출한다.

```bash
python scripts/prepare_wmt23.py \
  --output_dir data/wmt23_accessible_parallel \
  --mtdata_dir data/raw/wmt23_accessible \
  --corpus_profile accessible_parallel --seed 42 --validation_per_language 500

export WMT23_MANIFEST_SHA256=PREPARATION_OUTPUT_HASH
ALIGNMENT_LOSS=centered_infonce MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct \
  CUDA_VISIBLE_DEVICES=0 bash scripts/wmt23_contrastive_only.sh alternative
```

`WMT23_DATA_DIR`로 준비 디렉터리를 지정할 수 있다. Manifest와 모든 학습·평가 파일의
해시 검증이 끝나야 모델 학습이 시작된다. Downstream은 dedupe 후 전체 DE/HE/JA pool을
비례 셔플하며, 혼합 배치에서도 각 행의 언어 프롬프트를 사용한다. 언어별 동일 노출량을
강제하지 않는다. Alternating은 alignment 50,000 + SFT 50,000 update이며,
동일한 pool·seed·batch 설정이면 contrastive-then-SFT와 objective별 샘플 순서가 같다.

### Current MT profile: ALMA + Japanese, OPUS alignment

현재 두 WMT23 학습 스크립트의 기본 profile은 `alma_ja_opus`이다.
학습 언어는 **de/cs/ja**, 미학습 평가 언어는 **zh/ru/uk**, anchor는 English다.
Mid-Align의 데이터 수집 방식을 따르되, 일본어 추가와 별도의 OPUS 정렬은 명시적인
차이이므로 원논문의 데이터 설정을 완전히 재현했다고 표기하지 않는다.

| 용도 | 데이터 |
|---|---|
| DE/CS SFT | `haoranxu/ALMA-Human-Parallel` 원본 train 전체: 14,211 / 12,076쌍 |
| JA SFT | WMT20 en-ja 1,000 + ja-en 993쌍, FLORES-200 dev 997 + devtest 1,012쌍 |
| SFT validation | DE/CS는 ALMA validation 각 1,002쌍; JA는 WMT21 en-ja 1,000쌍 |
| 번역 test | `haoranxu/WMT23-Test` 양방향 전체; cs-en은 저자 코드처럼 en-cs를 역방향 사용 |
| Alignment | OPUS-100 seen-language train에서 held-out 영어 원문을 제외한 뒤 언어쌍별 10,000개 |

SFT는 모든 병렬 쌍을 양방향으로 사용하여 **60,578개** 예제를 만든다. 원본 문자열과
행 순서를 보존하며 샘플 수 제한·도메인 필터·Unicode 정규화를 적용하지 않는다.
현재 `balanced_mixed` sampler는 **DE/CS/JA의 학습 노출 수를 균등하게** 정한다.
각 언어의 전체 양방향 pool을 셔플·순환하므로 적은 언어는 더 자주 반복하고,
어느 언어의 원본 데이터도 줄이지 않는다. 언어별 quota를 정한 뒤 전체 순서를
셔플하므로 배치마다 언어 개수를 강제하지 않는다. 각 행의 **출발/도착 언어**로 prompt를 만든다.
과거 비례 샘플링 실행은 `proportional`로 구분해 보존한다.
평가도 방향별로 분리하고 BLEU tokenizer는 도착 언어를 기준으로 선택한다.
HF WMT23-Test의 참조 번역을 그대로 사용하며 공식 저장소의 다른 ref로 바꾸지 않는다.

ALMA 원본처럼 JA의 WMT 수집 기간도 2017–2020으로 제한한다. 이 기간의 JA 자료는
WMT20에만 있다. JA validation은 WMT21, 최종 test는 WMT23으로 분리한다.
FLORES의 id와 URL을 확인하여 영어와 일본어를 대응시킨다. SFT 학습과 held-out 데이터의
동일 언어·동일 병렬 쌍 중복이 있으면 자동 삭제하지 않고 준비 단계에서 실패한다.
OPUS의 zh/ru/uk **train은 다운로드하지 않으며 adaptation에도 사용하지 않는다**.
그 언어들의 OPUS validation/test는 진단 전용이며 checkpoint 선택에 사용하지 않는다.

모델·정렬 층·pooling·loss·LoRA 설정과 기존 update 예산은 유지한다. Alternating과
contrastive-then-SFT 모두 alignment 50,000 + SFT 50,000 update이며 batch 16,
accumulation 1에서는 SFT 예제 노출이 800,000회이다. 현재 작은 SFT pool에서는
언어별 266,666 또는 266,667회이며, 800,000이 3으로 나누어떨어지지 않아
최대 1개 차이가 난다. 정렬/SFT의 RNG를 분리하여 alternating과 CTT가 각 objective에서
같은 순서를 사용한다. 언어마다 pool 크기가 달라 반복 횟수는 다르며,
이는 원논문의 최대 5 task epoch와 다른 학습 예산이다.

```bash
python scripts/prepare_wmt23.py \
  --output_dir data/wmt23_alma_ja_opus --corpus_profile alma_ja_opus --seed 42

export WMT23_MANIFEST_SHA256=PREPARATION_OUTPUT_HASH
ALIGNMENT_LOSS=centered_infonce MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct \
  CUDA_VISIBLE_DEVICES=0 bash scripts/wmt23_contrastive_only.sh alternative
```

각 원본 HF revision, WMT 아카이브 MD5, 추출 파일 SHA256, 준비 코드 hash와 최종 Arrow
체크섬을 manifest에 기록한다. 다음 실험에서는 이 로컬 저장본과 같은 manifest hash를
재사용한다. 과거 recipe 다운로드 파일은 이 profile의 학습에 필요하지 않다.

정렬 forward는 모든 token의 hidden state를 유지하면서 `logits_to_keep=1`로 사용하지
않는 전체 어휘 logits 할당을 줄인다. 표현·정렬 loss·gradient의 동일성을 검사하며,
배치 크기, 정렬 층과 pooling 설정은 바꾸지 않는다. Qwen3.5의 WMT SFT에서는
`--downstream_micro_batch_size 8`로 16개 예제를 8개씩 두 번 forward/backward하고
optimizer는 한 번만 update한다. 각 microbatch의 HF 기본 CE에
`해당 microbatch 정답 token 수 / 16개 전체 정답 token 수`를 곱하여 누적한다.
별도의 vocabulary CE chunk 구현은 사용하지 않는다. 정렬은 16개를 한 번에
비교하므로 negative 개수와 centered loss의 중심 계산이 유지된다.
전역 `batch_size=8, accumulative_steps=2`는 정렬 negative까지 줄이므로 사용하지 않는다.
Llama는 16개를 한 번에 처리한다. microbatch 분할은 dropout의 실제 난수 배치를
바꿀 수 있으며, 동일한 seed가 다른 계산 방식 사이의 bitwise 동일성을 뜻하지 않는다.
WMT 학습 스크립트는 `PYTORCH_ALLOC_CONF=expandable_segments:True`를 기본 적용하여
길이가 다른 validation과 학습 배치 사이의 CUDA 메모리 단편화를 줄인다.

`scripts/select_checkpoint.py RUN_DIR`의 기본 선택 기준은 저장된 task의 seen validation
(`wmt23_in`)이며, contrastive-only에서는 `align_in`이다. 초기 step 0과 저장되지 않은
검증 step은 후보에서 제외한다. 현재 save/eval 간격 1000/2500에서는 5000의 배수만
두 조건을 만족한다. 총 학습량을 고정하여 비교할 때는 `--rule final_step`을 사용한다.
WMT 평가 실행은 `scripts/wmt23_eval.sh RUN_DIR`을 사용한다. 기존 자동 평가 큐
`scripts/evaluation_queue.py`는 MASSIVE용 설정이므로 WMT run을 그 큐에 넣지 않는다.
일본어 SacreBLEU는 `requirements.txt`의 `sacrebleu[ja]` 의존성이 필요하다.

### Mid-Align과 같은 COMET-22 채점

[저자 평가 코드](https://github.com/dannigt/mid-align/blob/master/scripts/run_inference_eval_wmt23.py)는
HF `evaluate.load("comet")`에 원문·생성 번역·정답 번역을 전달한다.
[HF 구현](https://github.com/huggingface/evaluate/blob/main/metrics/comet/comet.py)의
`unbabel-comet >= 2` 기본 모델인 **`Unbabel/wmt22-comet-da`**를 직접 호출한다.
COMETKiwi가 아닌 reference-based 모델이다. Revision
`2760a223ac957f30acfb18c8aa649b01cf1d75f2`와 checkpoint/hparams SHA256을 검증한다.
저자가 실행했던 패키지 전체 버전은 확인되지 않아 bitwise 재현을 뜻하지는 않는다.

COMET은 Transformers 4 / NumPy 1을 요구하므로 학습 환경과 분리한다.
별도의 채점 환경은 다음과 같이 설치한다.

```bash
python -m venv .venv-comet22
# CPU 전용일 때 먼저 설치. CUDA 채점 환경을 만들 때는 이 줄을 생략한다.
.venv-comet22/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
.venv-comet22/bin/python -m pip install -r requirements-comet22.txt
```

이미 저장한 번역을 재생성 없이 채점한다. `in`/`out` 각각 실행하며, 한 디렉터리에는
같은 run·split·scope의 prediction 파일만 둔다. 첫 실행은 약 2.3 GB의 COMET
checkpoint와 XLM-R tokenizer/config를 다운로드한다.

```bash
.venv-comet22/bin/python scripts/score_comet22.py \
  --prediction_dir CHECKPOINT/evaluations/test/in --gpus 0 --batch_size 16
.venv-comet22/bin/python scripts/score_comet22.py \
  --prediction_dir CHECKPOINT/evaluations/test/out --gpus 0 --batch_size 16

# 번역 생성 + 기존 BLEU + 추가 COMET. LLM 프로세스 종료 후 COMET을 실행한다.
EVAL_COMET22=true bash scripts/wmt23_eval.sh MODEL_FOLDER/RUN_NAME

# 별도의 CUDA 지원 COMET 환경이 있을 때:
EVAL_COMET22=true COMET22_PYTHON=/path/to/comet-env/bin/python \
  COMET22_GPUS=1 bash scripts/wmt23_eval.sh MODEL_FOLDER/RUN_NAME
```

`EVAL_WMT23_METRIC=comet22`도 지원한다. 이 경우 생성은 metric `none`으로 수행하고
이후 COMET만 채점한다. `COMET22_BATCH_SIZE` 기본값은 저자 HF wrapper가 호출하는
COMET의 기본 batch size인 16이며 `COMET22_GPUS`는 0(CPU)이다.
`evaluate.py` 직접 호출도 `--wmt23_metric comet22 --comet22_python ...`로 가능하다.
이 경로에서는 LLM이 메모리에 남으므로 기본 CPU 채점을 사용하거나 위의 분리 실행을
사용한다. `--checkpoint`/`COMET22_CHECKPOINT`는 동일 공식 checkpoint의 로컬 복사본만
허용하며 `checkpoints/model.ckpt` 및 상위 디렉터리의 `hparams.yaml` 구조가 필요하다.

출력은 각 scope의 `wmt23_comet22_metrics.json`과 `wmt23_comet22_scores.jsonl`이다.
기존 `wmt23_metrics.json`의 BLEU 결과는 유지한다. 원문은 번역 방향 그대로
`src`, 생성문은 `mt`, 정답은 `ref`로 전달하며 추가 정규화·strip·문장 분할을 하지 않는다.
빈 생성문도 점수에 포함하고, 누락된 정답·중복 sample ID·비정상 점수는 오류로 처리한다.
문장별 결과는 입력 파일/행 순서를 보존한다. 저장된 생성문은 `evaluate.py`의 기존
생성 후처리가 적용된 값이며, 원논문과 생성 길이·prompt 등의 차이는 별도로 남아 있다.

- `comet22`, `mean_score`: 원래 모델 점수. `*_x100`은 논문 표처럼 100배 표시한 값.
- `by_language`: `en-de`, `de-en` 등 방향별 문장 평균.
- `macro_average`: 방향별 점수의 산술평균. 전체 문장 평균 `mean_score`와 구별한다.
- `macro_by_translation_direction`: `en_to_x`와 `x_to_en`을 분리한 방향별 macro 평균.
- 모델 revision, weights/hparams hash, 실제 패키지 버전, prediction 파일 hash를 기록한다.

COMET 채점기를 맞추어도 전체 MT 실험이 같아지는 것은 아니다. 현재 OPUS 정렬 corpus,
언어 균등 반복 sampler/seed, 학습 update 수, batch/optimizer/LoRA/양자화 설정과
모델·정렬 층·pooling 차이는 유지된다. 동일한 실험 설정 안에서 방법 간 비교에 사용한다.

### WMT 학습·평가 큐

2026-10-05 최신 요청에 따라 후순위 학습을 중단하고 BLEU·COMET 평가를 우선 실행한다.
`scripts/wmt23_pipeline.py`의 평가 전용 모드는 새 학습을 시작하지 않는다.

1. GPU 0: 완료된 모델부터 즉시 BLEU·COMET 평가. 진행 중이던 후순위 학습과 나머지
   후순위 큐는 중단했으며 자동 재시작하지 않는다.
2. GPU 1: 실행 중인 우선순위 Qwen InfoNCE alternative를 마친 뒤 남은 평가에 합류한다.
   두 GPU는 같은 큐에서 준비된 모델을 하나씩 가져가 중복 없이 평가한다.
   우선순위는 모델별 InfoNCE contrastive-only / transfer-only / contrastive-then-transfer /
   alternative와 centered InfoNCE alternative다. 각 최종 adapter로 seen/unseen 언어 양방향의
   WMT23 test 21,067개 예제에 BLEU와 COMET-22를 계산한다. 생성 batch 1, 신규 토큰 제한
   16,384를 유지하고, 생성 프로세스 종료 후 COMET GPU batch 16으로 채점한다.

현재 평가 계획과 상태는 `logs/wmt23_bleu_comet_20261005/`에 저장한다.
완료된 학습은 config·최종 step·완료 metadata·adapter를 검증해 재사용한다.
실행 중인 학습을 인계할 때 PID와 시작 시각을 함께 확인한다. 실행 전에 학습 명령이
저장된 계획과 같은지 검사한다. 평가 실패는 상태와 로그에 기록하고 이후 평가를 차단한다.
재실행 시 완료된 생성/COMET 결과는 예제 수와 prediction hash 검증 후 재사용한다.

```bash
python scripts/wmt23_pipeline.py \
  --manifest logs/wmt23_bleu_comet_20261005/manifest.json \
  --state_dir logs/wmt23_bleu_comet_20261005 --dry-run

# 각 큐의 상태/로그: state.json / controller.log
# 평가 결과: RUN/evaluations/priority_mt_20261004/test/{in,out}/

# 평가가 끝날 때 자동 집계하며, 아래 명령으로 수동 갱신할 수도 있다.
python scripts/summarize_wmt23.py \
  --manifest logs/wmt23_split_20261005/previous_pipeline/manifest.json \
  --state_dirs logs/wmt23_split_20261005/gpu0_later logs/wmt23_bleu_comet_20261005 \
  --output_dir reports/wmt23_20261005
```

큐의 중복 실행은 controller lock으로 막고, GPU별 기존 학습 lock도 공유한다.
이 큐는 GPU를 지원하는 COMET 환경을 manifest의 `comet_python`으로 지정해야 한다.
`summary.md`와 `summary.csv`에는 BLEU·COMET만 기록한다. 학습 중 validation loss는
별도 `validation.md` / `validation.csv`에 보관한다.
`test_wmt23_pipeline.py`는 단계 전환, 평가 전용 모드의 학습 차단, GPU 간 중복 없는
작업 배정, 완료 판정 및 오래된 prediction/COMET 결과의 재사용 방지를 검사한다.
