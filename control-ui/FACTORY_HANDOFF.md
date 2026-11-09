# 공장 배경 main 통합 인계

2026-11-09 기준, **현재 HQ 회색 압축 캔 베일 버전을 main 통합용 고정본으로 유지한다.**
일반 갈색 골판지 상자는 SSH Mac mini에서 진행하는 별도 실험이다. 현재 고정본에
병합하지 않으며, 아래 자산이나 기본 배경 설정을 실험 결과로 덮어쓰지 않는다.

갈색 종이박스 별도 작업본: SSH `macmini`의
`/Users/jeff/project/YC_compiled_Hackathon-factory-boxes/`.
이 디렉터리의 `README-PAPER-BOXES.md`를 참고한다. 무늬 없는 상자 6개와 별도 제거
영역으로 팔레트를 보존하며, 빌드·미터 좌표 변환 검사 및 500k 배경 화면 검증을 마쳤다.
이 변경은 현재 로컬 실행 코드나 HQ 원본 자산에 적용하지 않았다.

**고정본 식별**

- 자산 디렉터리: `public/worlds/factory-conveyor-hq/` 전체.
- World ID: `55ff7d06-eaef-4460-9197-8db675bd1293`.
- 모델: `marble-1.1`; 원본 SPZ **1,920,000 splats**.
- 패키지: 약 **79 MiB**; 현재 파일 합계 **82,056,554 bytes**.
- 기준 파일: `public/worlds/factory-conveyor-hq/world.json`.
- 기본 경로 및 `?factory=hq`는 HQ 고정본이다. `?factory=baseline`은 별도 구버전
  `public/worlds/factory-conveyor/`을 선택한다. A/B 비교가 필요할 때만 구버전도 함께 옮긴다.

**옮길 자산과 코드**

아래 경로는 모두 `control-ui/` 기준이다. HQ 자산은 일부 파일만 고르지 말고 디렉터리
전체를 그대로 옮긴다.

| 구분 | 실제 경로 / 파일 |
| --- | --- |
| 배경 설정·SPZ | `public/worlds/factory-conveyor-hq/world.json`, `factory-100k.spz`, `factory-500k.spz`, `factory-full.spz` |
| 이미지·조명 | 같은 디렉터리의 `source-reference.png`, `source-reference.jpg`, `panorama.jpg`, `lighting-pano.jpg`, `thumbnail.jpg` |
| 형상·검증 | 같은 디렉터리의 `collider.glb`, `calibration-report.json`, `visible-floor-report.json`, `source-reference-report.json` |
| 출처·프롬프트·라이선스 | 같은 디렉터리의 `reference-prompts.json`, `ATTRIBUTION.txt`, `SPARK_LICENSE.txt` |
| 배경 로더 | `src/sim/FactoryBackdrop.ts` — 경로 선택, 원본 SPZ 로딩, 행렬 적용, 조명, 품질·스케일 제어 |
| 두 뷰어 | `src/sim/FactoryScene.tsx`, `src/sim/NativeScene.tsx` — offline / GPU native 연결 및 표시 |
| 표시·의존 모듈 | `src/sim/ScaleReference.ts`, `src/sim/NativeMaterials.ts`, `src/sim/FrankaAnalyticalIK.ts`, `src/protocol/types.ts` |
| 연결·스타일 | `src/App.tsx`의 두 scene 연결부, `src/styles.css`의 `factory-background-*` 설정 UI |
| 패키지 | `package.json`, `package-lock.json` — **Spark 2.4.0 / Three.js 0.181.2** 잠금 파일 유지 |
| offline 모델 | `public/models/panda/` 전체 — `factory.xml`, `panda.xml`, manifest, meshes, 라이선스 포함 |
| 캔 외형 | `public/models/libero-alphabet-soup/` 전체 — XML, texture, visual mesh, `LIBERO_LICENSE` 포함 |

`App.tsx`에는 다른 작업의 GPU 연결·조작·기록 흐름도 있다. main의 기존 작업을 보존하면서
scene import와 props 연결부를 병합한다. `NativeMaterials.ts`와 캔 외형 자산도 함께
참고해야 현재 캔 텍스처 표시가 유지된다. 자산 URL은 사이트 루트의 `/worlds/...`,
`/models/...`에서 제공된다.

**스케일·표시 설정**

- 시뮬레이터 좌표는 **미터 단위 Z-up**이다. `world.json`의
  `calibration.rawToWorld`에는 API 미터 환산, OpenCV→Z-up 회전, 바닥 경사·높이 및
  원본 SPZ의 보이는 바닥 보정이 이미 들어 있다. 이 행렬을 **한 번만** 적용한다.
  그 뒤 API metric factor나 추가 `Rx` 회전을 다시 적용하면 크기·축이 틀어진다.
- 현재 API factor는 **1.8628582954406738 m/raw unit**. 수동 보정은
  `scaleCorrection=1`, `groundCorrectionM=0`, `yawDegrees=0`, `translationM=[0,0,0]`이다.
- Offline은 **작업대 상판 z=0**, 공장 바닥 **z=-0.78 m**이다. 작업대 높이 **0.78 m**,
  크기 **1.65×1.95 m**, 벨트 길이 **1.40 m**를 유지한다. 노란 눈금은 **0.50 m**,
  별도 기준 큐브는 **1.00 m**이다. 배경 스케일 조절은 로봇·벨트·기준 큐브를 키우지 않는다.
- Native는 서버가 전달한 수평 바닥 plane의 실제 높이를 읽고, 없을 때 0을 사용한다.
  Offline의 -0.78 m를 native 로봇이나 테이블에 적용하지 않는다. 받은 pose, 물리,
  정책 카메라와 관측은 그대로 유지한다.
- `NativeScene.tsx`의 **배경 layer → depth clear → native geometry → 조작 overlay**
  렌더 순서를 유지한다. 배경 splat이 native 로봇·작업물을 가리는 문제를 방지하는 설정이다.
  Offline의 조작 overlay 순서도 현재 구현을 유지한다.
- 두 뷰어 모두 **DPR 최대 2**, `antialias:false`이다. 기본은 500k를 먼저 띄운 뒤
  원본 full SPZ로 교체한다. **Balanced(500k)** / **Performance(100k)** 품질 제어가
  구현되어 있지만, 현재 operator 화면은 `operator-layout.css`에서 배경 설정 배지를
  숨기고 있다. 느린 장치용 품질 선택이 필요하면 main 화면에 이 제어를 연결한다.
  품질 전환·조회에는 API 크레딧이 들지 않는다.
- 바닥 보정 보고서의 median **1.69 mm**, P95 **4.45 mm**는 수치적 바닥 fitting 잔차다.
  생성된 건물의 절대 치수를 실측한 값은 아니다.

**통합·실행 경계**

현재 로컬 브랜치는 `codex/control-ui`이며, `control-ui/` 등을 포함한 작업이 **untracked**
상태다. 이 배경 변경이 이미 커밋·push되었다고 가정하거나, 존재하지 않는 커밋을
cherry-pick하지 않는다. main은 위 파일을 명시적으로 비교·통합하고 기존 미커밋 작업을
보존한다.

`.env`, `.marble/`, `node_modules/`, `dist/`는 전달 묶음에서 제외한다. 특히 API 키와
signed URL 캐시는 필요 없다. 고정본 열기에는 **새 World Labs 생성·결제가 필요 없다**.
런타임 자산이 이미 완성되어 있으므로 통합 과정에서 `world:generate`나
`world:calibrate`를 다시 실행하지 않는다.

```sh
cd control-ui
npm ci
npm run build
# 필요할 때 개발 서버 실행:
npm run dev
```

`dist/`는 main 환경에서 새로 빌드한다. 검증·재현 코드를 보관하려면 선택적으로
`scripts/generate-factory-standard.py`, `scripts/build-hq-panorama.py`,
`scripts/calibrate-factory-world.py`, `scripts/calibrate-visible-floor.py`와
`FACTORY_BACKGROUND.md`, `THIRD_PARTY_NOTICES.md`도 함께 참고한다.

**main에 전달할 문구**

```text
현재 HQ 회색 압축 캔 베일 배경을 그대로 통합해 주세요.
기준 문서는 control-ui/FACTORY_HANDOFF.md입니다.
고정 자산은 control-ui/public/worlds/factory-conveyor-hq/ 전체이며,
World ID는 55ff7d06-eaef-4460-9197-8db675bd1293입니다.
FactoryBackdrop / FactoryScene / NativeScene / ScaleReference 및 연결·모델·패키지
의존성을 문서대로 병합하고, rawToWorld는 추가 scale/축 변환 없이 한 번 적용해 주세요.
현재 로컬 변경은 untracked이므로 커밋·push 완료로 가정하지 말고 파일을 직접 비교해 주세요.
API 키·.marble 캐시·node_modules·dist는 복사하지 말고 npm ci와 build를 사용해 주세요.
갈색 일반 골판지 상자 버전은 SSH Mac mini의 별도 실험이며 현재 고정본에 합치지 마세요.
기존 main의 GPU 제어·물리·정책 관측 흐름을 보존하고 새 유료 생성은 실행하지 마세요.
```
