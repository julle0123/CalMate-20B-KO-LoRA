# 📌 CalMate-20B-KO-LoRA

## 1. 개요
CalMate-20B-KO-LoRA는 **gpt-oss-20b** 기반 한국어 일정 파서 모델입니다.  
LoRA(Low-Rank Adaptation) 기법을 활용해 **개인 비서용 일정 JSON 추출** 태스크에 맞춰 튜닝했습니다.

---

## 2. 모델 정보
- **Base Model**: [`unsloth/gpt-oss-20b-unsloth-bnb-4bit`](https://huggingface.co/unsloth/gpt-oss-20b-unsloth-bnb-4bit)  
- **LoRA Adapter**: [`Seonghaa/CalMate-20B-KO-LoRA`](https://huggingface.co/Seonghaa/CalMate-20B-KO-LoRA)  
- **Task**: 한국어 자연어 → 일정 JSON 변환  
- **출력 스키마**
```json
{
  "title": "string",
  "start": "YYYY-MM-DD HH:MM",
  "end": "YYYY-MM-DD HH:MM",
  "location": "string",
  "priority": "high|normal|low"
}
```

---

## 3. 데이터셋
- 형식: `test.jsonl`, `train.jsonl` (JSON Lines)
- 각 라인은 `{"user": "...", "gold": {...}}` 형식
- `gold`는 정답 JSON 스키마에 맞춰 작성됨

---

## 4. 학습 환경
- **Framework**: PyTorch + Transformers + PEFT(LoRA)
- **Quantization**: 4-bit (nf4, double quantization)
- **환경**
  - GPU: A100 / T4
  - Python 3.10
  - CUDA 12.x
  - Transformers 4.43+
  - bitsandbytes 0.43+

---

## 5. 학습 스크립트 예시
```python
from unsloth import FastLanguageModel
from peft import LoraConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
        task_type="CAUSAL_LM",
    )
)

# 이후 학습 진행
```

---

## 6. 평가 방법
모델과 LoRA 어댑터를 로드해 `test.jsonl`에 대해 평가:
```bash
python evaluate_calmate.py     --test ./test.jsonl     --adapter-id Seonghaa/CalMate-20B-KO-LoRA     --base-id unsloth/gpt-oss-20b-unsloth-bnb-4bit     --max-new 512     --temperature 0.2     --top-p 0.9     --batch-size 1
```

---

## 7. 사용 예시
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("unsloth/gpt-oss-20b-unsloth-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    device_map="auto",
    load_in_4bit=True
)
model = PeftModel.from_pretrained(model, "Seonghaa/CalMate-20B-KO-LoRA")

prompt = "내일 오후 3시에 강남역에서 민수랑 회의"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 8. 라이선스
- Base Model 및 LoRA 모두 Apache-2.0 라이선스 준수
- 데이터셋은 내부 제작 데이터 + 공개 일정 예시 데이터 혼합
