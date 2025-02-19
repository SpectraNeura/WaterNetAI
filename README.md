-----------------------개요-------------------------


[배경]

본 경진대회는 상수도 관망의 이상 시점과 누수 발생 구간을 정확하게 탐지할 수 있는 범용 AI 알고리즘 개발을 목표로 하고 있습니다. 
대회 참가자들에게 다양한 상수도 관망의 실제 데이터를 제공하여, 이를 기반으로 실시간으로 이상을 감지하고 효율적으로 의사결정을 지원할 수 있는 기술을 개발하도록 지원합니다. 
개발된 알고리즘은 향후 상수관망 디지털트윈 및 Water-Net 등 사내 시스템에 내재화하여 상수도 관리의 효율성과 정확성을 높이고자 합니다.


[주제]

상수도 관망 이상 감지 AI 알고리즘 개발


[설명]

다양한 상수도 관망의 실시간 이상을 감지하는 AI 모델을 개발해야 합니다. 
학습 데이터는 A와 B 구조를 가진 상수도 관망 데이터로, 분 단위의 시간 정보가 모두 공개되어 있습니다. 
반면, 평가 데이터는 C와 D 구조를 가진 상수도 관망 데이터로 제공되며, 현재 시점 T를 기준으로 시간이 비식별화되어 있습니다.
모델은 평가 데이터에서 최대 1주일 분량의 분 단위 입력 데이터를 바탕으로 T+1분 시점의 이상 여부를 감지해야 하며, 관망 구조 내 존재하는 각각의 압력계(P)에 대해 이상을 감지할 수 있어야 합니다.



-----------------------규칙-------------------------


1. 리더보드

	* 평가 산식: 관망 구조 내 압력계별 변형 F1 Score [코드]
		실제 정상 샘플에 대한 계산 방식 (*정상 샘플은 샘플 내 모든 압력계(P)가 정상(0)인 경우를 뜻함.)
			1) 예측에서 False Positive가 발생하면 해당 샘플의 점수는 0점으로 처리
			2) 실제 정상 샘플을 정확히 정상이라고 예측한 경우에는 평가 산식 계산에서 제외

	* 실제 비정상 샘플에 대한 계산 방식 (*비정상 샘플은 샘플 내 압력계(P) 중 하나라도 이상(1)인 경우를 뜻함.)
			1) Precision (정밀도)
			![image](https://github.com/user-attachments/assets/e0fa4218-6937-4dc5-a45f-48de807afca6)
   

    				* Matched Abnormal Weights : 예측과 실제 모두 이상(1)인 압력계들의 가중치 합
				* Predicted Abnormal Weights : 예측값이 이상(1)인 압력계들의 가중치 합
				* False Positives Weights : 가중치가 0인 압력계에서 예측값이 이상(1)인 경우
				* *가중치: 가중치는 실제 압력계 별 정상(0)/이상(1)의 여부와 달리 각 압력계에 대한 예측 배점 가중치를 뜻함. 가중치는 0 ~ 1의 수치를 가질 수 있음. 관망 내 압력계 별 배점 가중치는 내부 평가용 자료로 공개되지 않습니다. 단, 실제 답(GT)과 크게 다르지 않으며 정답에 근접한 예측에 대해 가산점을 주기 위한 용도.
   
			2) Recall (재현율)
			![image](https://github.com/user-attachments/assets/f935d87a-07f2-4e75-a9a1-82d848193cd7)
   
			
				* Total Abnormal Weights: 실제 이상(1)인 압력계들의 가중치 합.
   
			3) F1 Score (가중치 기반)
			![image](https://github.com/user-attachments/assets/d723091b-13e6-4a57-8c0e-e0fe21876225)

				* Precision 또는 Recall이 정의되지 않으면 F1 = 0.
		* 리더보드 Score: 샘플 별로 계산된 구조 내 압력계별 변형 F1 Score의 평균
				![image](https://github.com/user-attachments/assets/7c33321a-4d61-418f-a3e3-84e9525f2cf1)
    
							
				* 유효 샘플 수: 정상 샘플에서 False Positive가 없는 경우를 제외한 샘플의 개수
* Public Score: 전체 테스트 샘플 중 '관망 구조 C' 샘플
* Private Score: 전체 테스트 샘플 100%


2. 평가 방식

* 1차 평가: 리더보드 Private Score
* 2차 평가: Private Score 상위 10팀 코드 및 PPT 제출 후 코드 검증


3. 개인 또는 팀 참여 규칙

* 개인 또는 팀을 이루어 참여할 수 있습니다.
* 팀을 이루어 참여하는 경우, 팀원 모두 참가 자격에 부합하는 상태여야합니다.
* 개인 참가 방법 : 팀 신청 없이, 자유롭게 제출탭에서 제출 가능
* 팀 참가 방법 : 팀 탭에서 가능, 상세 내용은 팀 탭에서 팀 병합 정책 확인
* 팀 구성 방법: 팀 페이지에서 팀 구성 안내 확인
* 팀 최대 인원: 5 명
* 동일인이 개인 또는 복수팀에 중복하여 등록 불가
  

4. 외부 데이터 및 사전 학습 모델

* 대회 제공 데이터 이외의 외부 데이터 사용 불가능
* 사용에 법적 제한이 없으며 논문으로 공개된 베이스의 사전 학습 모델(Pre-trained Model) 사용 가능


5. [중요] 평가 데이터 관련 Data Leakage 규칙

* 본 경진대회는 실시간 분 단위 탐지를 목표로 합니다. 
* 따라서 각 평가 데이터 샘플들은 분 단위 탐지를 위해 사전에 개별 파일(csv)로 분할되어 있습니다.
* 평가 데이터 샘플 내 시점 T는 비식별화 되어 있으며, 평가 데이터 샘플 ID(또는 파일명)는 시간의 흐름과는 전혀 인과관계가 없으며 이를 악용하거나 연관성을 유추하려는 시도는 금지됩니다.
	규칙 1) Lookback 기간
		* 분할된 평가 데이터 샘플들은 각 현재 시점 T를 가지고 있으며, 최대 1주일 기간의 T 시점 이전의 데이터로 구성되어 있습니다. (최대 활용 가능한 Lookback 기간: 1분 단위의 1주일)
	규칙 2) 평가 데이터 샘플 간 독립성 유지
		* 분할된 평가 데이터 샘플들은 각각 독립적으로 예측이 진행되어야 합니다.
		* 다른 평가 데이터 샘플을 활용하여 예측에 활용할 수 없으며, 다른 평가 데이터 샘플의 예측 결과도 활용할 수 없습니다.
	규칙 3) 이상 감지 Threshold 설정
		* Threshold 설정 독립성: 이상 감지에 활용되는 Threshold는 다른 평가 데이터 샘플의 예측 결과 또는 Anomaly Score 등의 정보로부터 설정할 수 없습니다.
		* Threshold 설정 데이터 제한: Threshold 설정은 '학습 데이터' 또는 '현재 추론 중인 평가 데이터' 내에서만 이루어질 수 있습니다. 다른 평가 데이터 샘플에서 계산된 값을 활용하는 것은 불가능합니다.
	규칙 4) 평가 데이터 샘플 내 통계 정보 활용
		* 분할된 평가 데이터 샘플 내 Input 데이터의 통계 정보는 해당 샘플의 추론 프로세스 내에서만 활용할 수 있습니다. (단, 해당 샘플의 통계 정보는 모델 학습에는 활용할 수 없습니다.)
	규칙 5) 모델 학습 데이터와 평가 데이터 분리
		* 평가 데이터 샘플들은 모델 학습에 활용될 수 없으며, Pseudo Labeling 기법도 사용 불가능합니다. 또한 이미지로 제공되는 '관망 구조 정보'도 마찬가지로 적용됩니다.
	규칙 6) 전처리 및 후처리 일관성
		* 평가 데이터 샘플들에 적용되는 전처리, 후처리 과정은 규칙 2와 규칙 3을 준수하는 범위 내에서 모든 평가 데이터 샘플에 일관되게 적용되어야 합니다. 특히, 특정 평가 데이터 샘플이나 관망 구조에 유리한 맞춤형 처리는 금지됩니다.

※ 본 대회는 시계열, 이상 감지, 분 단위 실시간 예측 등 Data Leakage 관련하여 중요한 사항이 많기 때문에, 반드시 숙지 후 진행해야합니다.
※ 본인이 진행하려는 방법론이 해당 규칙에 위반하는 지 판단이 불가능한 경우, 데이콘 공식 메일(dacon@dacon.io) 또는 토크 게시판에 반드시 문의 후 진행 부탁드립니다.
※ 대회 기간 중 또는 코드 검증 과정에서 규칙 위반 사실이 확인되는 경우 대회 실격에 해당할 수 있습니다.



6. 코드 및 PPT 제출 규칙

* 대회 종료 후 2차 평가 대상자는 아래의 양식에 맞추어 코드와 PPT를 dacon@dacon.io 메일로 기한 내에 제출
* 제출한 코드는 Private Score 복원이 가능해야 함
 	o  코드에 ‘/data’ 데이터 입/출력 경로 포함
	o  코드 파일 확장자: .R, .rmd, .py, .ipynb
	o  코드와 주석 인코딩: UTF-8
  	o  모든 코드는 오류 없이 실행되어야 함(라이브러리 로딩 코드 포함)
	o  개발 환경(OS) 및 라이브러리 버전 기재

* 솔루션 PPT 자료
   	o 자유 양식으로 작성

* 제출 파일 목록
   	o Private Score 복원이 가능한 코드 파일
   	o Private Score 복원이 가능한 모델 weight 파일
   	o 솔루션 PPT 자료
