# CV-09조 Data Centric Wrap-up Report

# 1. 프로젝트 개요

- 프로젝트 주제
    - 학습 데이터 추가 및 수정을 통한 이미지 속 글자 검출 성능 개선
- 프로젝트 목표
    - streamlit을 사용한 분석을 통해 다양한 아이디어 적용
- 데이터셋
    - 진료비 영수증 데이터셋 100장. 캠퍼들이 라벨링한 추가 데이터 200장

# 2. 프로젝트 팀 구성 및 역할

- 김지범: 추가 데이터셋을 이용한 학습, 최종 데이터셋 제작
- 이경봉: 추가 데이터셋 학습, model deploy, 성능 개선 아이디어 설정
- 이윤석: 베이스라인 코드 수정, 배경 데이터셋 추가, ensemble
- 양경훈: 추가 데이터셋을 이용한 학습 및 Augmentation
- 정현석: 추가 데이터셋 학습 및 Android serving

# 3. 프로젝트 수행 절차 및 방법

1. EDA
2. Stremlit
3. 성능 개선 아이디어 
    - Train set 분리
    - Augmentation
    - 추가 데이터셋
    - Ensemble
4. 기타
    - 학습 속도 개선
    - 모바일 적용

# 4. 프로젝트 결과

- 결과
    
    Pulblic f1-score : 0.9766 (7/19등)
    
    Private f1-score : 0.9737 (9/19등)
    

✅: 성능 향상에 도움이 되었던 아이디어

1. EDA
    - ✅ 이미지당 글자 박스의 개수가 300~800개에 분포하는 것을 확인.
        
        → 1000개 이상의 박스를 예측했을 경우 underfitting으로 판단.
        
2. Streamlit
    - ✅ Model deploy
        
        모델의 Output을 모델과 사진을 넣으면 누구나 보기 쉽도록 하여 팀원들의 분석에 도움을 주고자 함.
        
    - ✅ MLops의 Feedback/Retrain 연습
        
        각 모델의 예측결과를 확인하며 오판하는 경우를 따로 수집하도록 함.
        
        이 데이터를 분석해 문제점을 보완해 나가는 방향으로 데이터 추가 후 모델 학습.
        
        → 개선 아이디어를 쉽게 찾을 수 있었고 모델향상에도 많은 도움이 됨.
        
3. 성능 개선 아이디어 결과
    - ✅ Train set 분리
        - 기존 베이스라인 코드는 validation set이 없어 모델의 overfitting을 판단하기 어려움.
            
            → train set에서 랜덤하게 분리 후, validation set에는 detect파일 전처리 수행.
            
    - Augmentation
        - ✅ 영수증 데이터 특성상 filp을 사용한 다양한 데이터가 들어오지 않음.
            
            → flip이나 큰 각도의 rotate는 적용하지 않음.
            
        - 영수증의 글자가 모두 검은색이며, 도장은 인식하지 않아야하므로 color를 변경하는
            
            augmentation은 성능하락을 발생시킨다고 판단.  
            
            → brightness, contrast만 조정. 성능 하락
            
        - ✅ 흑백으로 구성된 배경을 글자로 인식하는 문제 발생.
            
            → color,brightness, contrast 모두 조정하는 것이 배경과 도장을 구분하는데 도움을 줌
            
    - 추가 데이터셋
        - ✅ 캠퍼들이 라벨링한 데이터셋 추가. 문서 하단에 글자가 많아 박스 예측이 위 보다는 아래쪽에 많은 양이 분포.
            
            → 상하좌우 어느쪽에도 치우지지 않고 유사한 양식을 가진 AI HUB 금융 데이터셋 추가
            
        - ✅ 배경과 줄을 글자로 인식하는 문제. 모델이 글자를 인식하는 것이 아닌 검은색 패턴을
            
            인식. 문자가 아닌 글자를 판단하기 위한 비슷한 양식과 다른 글씨체를 가진 데이터 필요. 
            
            → AI HUB 다양한 공공행정 문서 데이터셋 추가. 
            
            → 손글씨 데이터셋은 성능향상은 있었지만 손글씨에만 box가 있어 사용하지 않음.
            
        - ✅ 여러 데이터셋을 넣어도 몇몇 특정 데이터에서 배경과 글씨를 오판하는 경우가 생김.
            
            → 격자무늬 또는 검은 패턴을 가진 배경 데이터셋 추가. 
            
        - ✅ 각각의 데이터셋의 크기가 달라 한쪽으로 치우치지 않도록 함.
            
            → 최종적으로 기본 100장, 직접 라벨링 데이터 100장, 금융 데이터 100장, 
            
                공공기관 100장, 배경 사진 20장을 사용.
            
    - Ensemble
        - ✅ nms, wbf 등 object detection에 사용한 방법을 사용하려 했으나, label과 score의 정보가 없기 때문에 bbox만을 가지고 ensemble하는 방법을 고려.
            
            → 가장 많은 박스를 만들어낸 예측을 기준으로 나머지 예측 박스와 IoU를 계산해 
            
                0.5이상의 박스들로 평균을 구함.
            
            → 큰 성능향상을 보였으나 하나의 박스를 기준으로 잡고 나머지를 맞추다보니 기준되는 
            
                예측에 없는 박스는 ensemble하지 못한 단점. 그로인해 precision보다 recall이 작음.
            
4. 기타
    - 학습속도 개선
        - Mixed Precision 을 통한 학습속도 개선 시도
            
            → 적용을 해보았지만 1epoch에서 loss가 NaN으로 발산하였고 내부적인 모델구현체 
            
            안에서 log 0 이 수행되는것을 확인
            
            → 모델 수정이 금지된 대회 룰이 있었고 실제 학습속도도 크게 개선되지 않았기에 
            
            최종적으로 적용하지 못했음.
            
        - ✅ Data Pickle
            
            앞서 Mixed Precision을 통해 DataLoader에 시간이 오래걸리는 것을 확인함. 데이터를 
            
            Pickle로 저장하여 geo map, roi mask 연산 과정을 없앰.
            
            → Random하게 적용되는 augmentation때문에 학습에 영향을 끼치지 않은 validation
            
                set에만 적용. evaluate 시간 10배 단축
            
    - Android serving
        - 안드로이드 기기에서 모델을 serving 하기 위해 baseline 코드를 제작.
        - pretrained 된 object detection tflite 모델을 사용하였고, bounding box 및 클래스와 confidence score를 표시할 수 있도록 구현.
        - 실제 serving을 위하여 pytorch로 작성된 custom 모델을 pth → onnx → pb → tflite 의 과정을 거쳐 변환.
        - tflite 모듈에서는 **[batch x height x width x channels] & batch = 1** 형식으로 맞추도록 호환성이 설정되어 있습니다. 그러나 pytorch에서는 **[batch x channels x height x width]** 로 입력받도록 모델을 설계하였기 때문에 모델 변환이 효율적인 방식으로 이루어지지는 않음.