import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Noise 제거
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
data = data.drop_duplicates(["ID"], keep = False)

# Back translation - T5
model_ckpt = "KETI-AIR/ke-t5-small"
max_token_length = 64

kor2eng_model_dir = "./kor2eng"
kor2eng_tokenizer = AutoTokenizer.from_pretrained(kor2eng_model_dir)
kor2eng_model = AutoModelForSeq2SeqLM.from_pretrained(kor2eng_model_dir)

input_text = data["input_text"].tolist()
inputs = kor2eng_tokenizer(input_text, 
                           padding=True, 
                           max_length=max_token_length, 
                           truncation=True, 
                           return_tensors="pt")

english = kor2eng_model.generate(
    **inputs,
    max_length=max_token_length,
    num_beams=5,
)

eng_t5 = kor2eng_tokenizer.batch_decode(english, skip_special_tokens=True)

eng2kor_model_dir = "./eng2kor"
eng2kor_tokenizer = AutoTokenizer.from_pretrained(eng2kor_model_dir)
eng2kor_model = AutoModelForSeq2SeqLM.from_pretrained(eng2kor_model_dir)

input_text = eng_t5
inputs = eng2kor_tokenizer(input_text, 
                           padding=True, 
                           max_length=max_token_length, 
                           truncation=True, 
                           return_tensors="pt")


korean = eng2kor_model.generate(
    **inputs,
    max_length=max_token_length,
    num_beams=5,
)

kor_t5 = eng2kor_tokenizer.batch_decode(korean, skip_special_tokens=True )
data["input_text"]=kor_t5
data.to_csv("t5-aug.csv", index=False)