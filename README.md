# 공간지능 및 멀티모달 AI 기반 다중 사회기반시설 진단·유지관리 기술 개발

**AI-Driven Spatial Intelligence for Integrated Infrastructure Health Monitoring and Maintenance**

> **상태: 1차년도(2026) 결과물 공개 · 지속 업데이트 중**
> 과제 기간(2026–2028) 동안 연차별 데이터셋과 모델을 이 저장소에 계속 추가합니다.

지표투과레이더(GPR) 영상에 기상·위치 같은 환경 정보를 함께 넣어 사회기반시설의 이상을 진단하고, 진단 결과를 유지관리 지침과 연결하는 기술을 개발하는 국제공동연구 과제의 공개 저장소입니다.

---

## 1. 과제 정보

| 항목 | 내용 |
|---|---|
| 사업명 | 협력거점형 국토교통 국제협력 연구개발사업 (선도기술탐색형 국제공동연구) |
| 중앙행정기관 / 전문기관 | 국토교통부 / 국토교통과학기술진흥원(KAIA) |
| 과제번호 | RS-2026-25523576 |
| 연구기간 | 2026.04.01 – 2028.12.31 |
| 주관연구개발기관 | 성균관대학교 Data eXperience Laboratory (연구책임자 박은일) |
| 공동연구개발기관 | ㈜셀파이엔씨 |
| 위탁연구개발기관 | University of Minnesota |
| 해외연구개발기관 | Universitat Jaume I |

| 기관 | 역할 |
|---|---|
| 성균관대학교 | 멀티모달 진단 모델 개발, 유지관리 전략 생성 구조 설계 |
| ㈜셀파이엔씨 | GPR 현장 자료 선별·라벨링, 데이터셋 구축·검수 |
| University of Minnesota | 공간 메타데이터·공간정보 품질 기준 수립 |
| Universitat Jaume I | 기능별 에이전트 및 오케스트레이션 구조 설계 |

## 2. 저장소 구성

| 폴더 | 내용 | 주요 수치 |
|---|---|---|
| [`1-year/1.dataset`](1-year/1.dataset) | **GPR–기상 멀티모달 데이터셋** (비식별화본). 서울 도로 하부 공동(cavity) GPR B-scan 영상과, 영상마다 라벨·기상 정보를 담은 XML | 7,376장 · 서울 25개 자치구 308개 지점 · 환경 변수 103개 |
| [`1-year/2.model`](1-year/2.model) | **GPR–기상–위치 융합 진단 모델**. YOLOv8s 영상 특징을 기상·위치 정보로 보정(FiLM)해 결함을 검출하고 이미지를 정상/불량으로 판정. 코드·설정·가중치·데이터 분할 포함 | 시험 F1 88.33% (학습에 쓰지 않은 구조물 8곳, 3,000장) |

> **두 결과물은 서로 다른 데이터로 만들었습니다.**
> 데이터셋은 도로 하부 공동 영상이고, 모델은 건축·교량·옹벽·터널 시설물 GPR 영상 20,000장으로 학습·평가했습니다.
> 2·3차년도에 두 데이터를 공통 구조로 묶어 통합 학습할 예정입니다.

각 폴더의 README에 데이터 형식, 실행 방법, 결과가 자세히 정리되어 있습니다.

```
infra_anomaly_detection/
├── README.md              이 문서
└── 1-year/                1차년도(2026) 결과물
    ├── 1.dataset/         GPR–기상 멀티모달 데이터셋 (images/, xml/, README.md)
    └── 2.model/           멀티모달 진단 모델 (code/, weights/, data_split/, README.md)
```

## 3. 연차별 계획

| 연차 | 목표 | 이 저장소에 추가할 결과물 |
|---|---|---|
| **1차년도 (2026)** | 공간지능 및 멀티모달 진단 기술 개발 · 유지관리 기술 설계 | GPR–기상 데이터셋, 멀티모달 진단 모델 **(공개 완료)** |
| 2차년도 (2027) | 유지관리·검색 기술 개발 · 에이전트 오케스트레이션 기반 통합 관리 기술 설계 | 유지관리 시나리오 데이터셋, 지침 검색·전략 제안 모델 |
| 3차년도 (2028) | 유지관리 전략 최적화 · 통합 관리 기술 고도화 | 통합 데이터셋, 에이전트 기반 통합 관리 및 경량화 모델 |

## 4. 인용

이 저장소의 데이터셋이나 모델을 사용했다면 아래와 같이 인용해 주세요.

```bibtex
@misc{dxlab2026infra,
  title        = {AI-Driven Spatial Intelligence for Integrated Infrastructure Health Monitoring and Maintenance: Year-1 Dataset and Model},
  author       = {{Data eXperience Laboratory, Sungkyunkwan University}},
  year         = {2026},
  howpublished = {\url{https://github.com/dxlabskku/infra_anomaly_detection}},
  note         = {Supported by KAIA grant funded by the Ministry of Land, Infrastructure and Transport (RS-2026-25523576)}
}
```

## 5. 사사

본 연구는 국토교통부/국토교통과학기술진흥원의 지원으로 수행되었습니다 (과제번호 RS-2026-25523576).

This study was supported by the Korea Agency for Infrastructure Technology Advancement (KAIA) grant funded by the Ministry of Land, Infrastructure, and Transport (RS-2026-25523576).

> 이 연구개발내용을 대외적으로 발표할 때에는 반드시 국토교통부(국토교통과학기술진흥원)에서 시행한 국토교통과학기술 연구개발사업의 결과임을 밝혀야 한다.

## 6. 문의

- 질문이나 오류 제보는 이 저장소의 [Issues](https://github.com/dxlabskku/infra_anomaly_detection/issues)에 남겨 주세요.
- 성균관대학교 Data eXperience Laboratory

## 7. 업데이트 기록

| 날짜 | 내용 |
|---|---|
| 2026.11 | 1차년도 데이터셋(`1-year/1.dataset`)과 진단 모델(`1-year/2.model`) 공개 |
