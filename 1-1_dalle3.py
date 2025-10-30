import io
import base64
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
import os
import time

# .env 파일 경로 지정 
load_dotenv(override=True)

# Open AI API 키 설정하기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key = OPENAI_API_KEY
)

# 프롬프트 예시: 장화신은 고양이가 우주복을 입고 우주를 걷는 모습
# 프롬프트 예시: Puss in Boots is a cat wearing a spacesuit and walking through space.
prompt= "Puss in Boots is a cat wearing a spacesuit and walking through space."
response = client.images.generate(
        model="dall-e-3", # 모델은 DALLE 버전3 (현 최신 버전)
        prompt=prompt, # 사용자의 프롬프트
        size="1024x1024", # 이미지의 크기
        quality="standard", # 이미지 퀄리티는 '표준'
        response_format='b64_json', # 이때 Base64 형태의 이미지를 전달한다.
        n=1,
    )

response = response.data[0].b64_json # DALLE로부터 Base64 형태의 이미지를 얻음.
image_data = base64.b64decode(response) # Base64로 쓰여진 데이터를 이미지 형태로 변환
image = Image.open(io.BytesIO(image_data)) # '파일처럼' 만들어진 이미지 데이터를 컴퓨터에서 볼 수 있도록 Open
# 이미지 저장하기, 폴더 없으면 생성
os.makedirs('output_img', exist_ok=True)
# 현재 시간을 기반으로 고유한 파일 이름 생성
timestamp = int(time.time())
image.save(f"output_img/dalle_image_{timestamp}.png")

