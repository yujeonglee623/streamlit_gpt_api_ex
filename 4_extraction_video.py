import streamlit as st
import os
import openai
from dotenv import load_dotenv
from moviepy import VideoFileClip
import requests
import json
import csv
import io
import re
from collections import Counter

# 화자 분리 라이브러리 (선택)
try:
    from pyannote.audio import Pipeline
    _HAS_PYANNOTE = True
except Exception:
    _HAS_PYANNOTE = False


# ==================== 오디오 추출 & 전사 ====================
def extract_audio(video_path, audio_path):
    """비디오에서 오디오를 추출하여 MP3 파일로 저장"""
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, codec='mp3')
        return True
    except Exception as e:
        st.error(f"오디오 추출 중 오류: {e}")
        return False


def transcribe_audio(audio_path, client):
    """기본 음성→텍스트 변환"""
    try:
        with open(audio_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        st.error(f"음성 변환 중 오류: {e}")
        return None


def transcribe_audio_with_timestamps(audio_path, client):
    """타임스탬프 포함 전사 (세그먼트 정보)"""
    try:
        with open(audio_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        return transcript
    except Exception as e:
        st.error(f"타임스탬프 전사 중 오류: {e}")
        return None


# ==================== 타임스탬프 포맷팅 ====================
def format_timestamp(seconds):
    """초를 HH:MM:SS 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_transcript_with_timestamps(transcript_data):
    """타임스탬프가 포함된 전사 데이터를 읽기 쉬운 형식으로 변환"""
    formatted_text = ""
    for segment in transcript_data.segments:
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        text = segment.text.strip()
        formatted_text += f"[{start_time} --> {end_time}]\n{text}\n\n"
    return formatted_text


# ==================== 번역 ====================
def translate_text(client, text_to_translate, target_language="English"):
    """OpenAI로 텍스트 번역"""
    if not text_to_translate:
        return None
    
    system_prompt = "You are a professional translator. Translate the given text accurately and naturally."
    user_prompt = f"Translate the following text to {target_language}:\n\n{text_to_translate}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"번역 중 오류: {e}")
        return None


# ==================== AI 이미지 생성 ====================
def summarize_script_for_image(client, script_text):
    """스크립트를 이미지 생성용 프롬프트로 요약"""
    if not script_text:
        return "A blank canvas."
    
    summary_prompt = (
        f"Summarize the following video transcript into a concise, descriptive sentence "
        f"that would be ideal for generating an illustrative image. "
        f"Focus on the main subject, action, and setting. Keep it under 20 words.\n\n"
        f"Transcript: {script_text[:1000]}"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text for image generation."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"이미지 요약 중 오류: {e}")
        return "An abstract representation of a video."


def generate_image_from_text(client, prompt_text, image_path):
    """DALL-E로 이미지 생성"""
    try:
        st.info(f"🎨 AI 이미지 생성 프롬프트: '{prompt_text}'")
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt_text,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        
        img_data = requests.get(image_url).content
        with open(image_path, 'wb') as handler:
            handler.write(img_data)
        
        return image_path
    except Exception as e:
        st.error(f"이미지 생성 중 오류: {e}")
        return None


# ==================== 키워드 추출 ====================
def extract_keywords(text, client, top_k=10, language="ko"):
    """
    LLM을 활용해 핵심 키워드를 JSON 리스트로 반환.
    환경변수 KEYWORD_MODEL (기본: gpt-4o-mini)
    실패 시 keyword_fallback으로 대체.
    """
    model = os.getenv("KEYWORD_MODEL", "gpt-4o-mini")
    prompt = (
        "당신은 텍스트에서 핵심 개념을 뽑아내는 분석가입니다. "
        "아래 스크립트의 핵심 키워드를 {} 언어로 상위 {}개만 엄밀하게 추출하세요. "
        "출력은 반드시 JSON 배열(list) 형식으로만, 각 요소는 string으로만 반환하세요. "
        "해시태그, 주석, 설명문 없이 순수 JSON만 출력합니다."
    ).format("한국어" if language.startswith("ko") else language, top_k)

    # 입력이 너무 길 경우 앞/뒤 일부만 결합
    max_chars = 12000
    if len(text) > max_chars:
        head = text[:6000]
        tail = text[-6000:]
        text_for_llm = head + "\n...\n" + tail
    else:
        text_for_llm = text

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt + "\n\n[스크립트]\n" + text_for_llm}
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        # JSON만 추출 시도
        json_str = extract_json_block(raw)
        keywords = json.loads(json_str)
        # 문자열 리스트 보장
        keywords = [str(k).strip() for k in keywords if isinstance(k, (str, int, float))]
        # 고유화 및 상위 k 제한
        seen = set()
        uniq = []
        for k in keywords:
            if k not in seen and k != "":
                uniq.append(k)
                seen.add(k)
        return uniq[:top_k] if top_k else uniq
    except Exception:
        # 실패 시 백업 로직
        return keyword_fallback(text, top_k=top_k, language=language)


def extract_json_block(s: str) -> str:
    """응답에서 JSON 배열 부분만 안전하게 추출."""
    s = s.strip()
    # code fence 안의 JSON 우선 탐색
    fence = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", s, flags=re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    # 전체에서 대괄호 블록 탐색
    arr = re.search(r"\[[\s\S]*\]", s)
    if arr:
        return arr.group(0).strip()
    # 마지막 시도: 그대로 반환 (파싱 시 에러 나면 fallback)
    return s


def keyword_fallback(text, top_k=10, language="ko"):
    """
    간단한 빈도 기반 키워드 추출 (형태소 분석기 없이 대략적).
    - 영문/숫자/한글 단어 토큰화
    - 길이 2 이상 & 흔한 불용어 제거
    """
    # 토큰화: 한글/영문/숫자
    tokens = re.findall(r"[A-Za-z0-9가-힣]+", text)
    # 소문자
    tokens = [t.lower() for t in tokens]

    stop_ko = set("""그리고 그러나 그래서 또한 또한은 또는 또는은 및 등이 이런 그런 저런 매우 너무 정말 그냥 바로 이제 또한 즉 다시 또한의 대한 대한의 
        경우 이런저런 하는 하는데 하는데요 하는걸 하는거 하는것 이런것 저런것 그런것 이것 저것 그것 그런 이 그 저 뭐 뭐지 뭐냐 왜 어떻게 누구 어디 언제 
        있다 없다 되다 하다 이다 아니다 많은 많은데 같은 같은데 해당 해당의 해당됨 포함 포함한 포함하여 포함해서 대한 대해 대해선 대해선요 대한거
        등 등등 등의 로 로서 으로 으로서 으로써 에 에서 에게 와 과 의 는 은 가 이 을 를 도 만 로부터 부터 까지""".split())
    stop_en = set("""
        the a an and or of for in on to with from as is are was were be been being that this these those it its by at if then
        so such very just also more most much many any some no not but into over under out up down off about can could should would
    """.split())

    filtered = [t for t in tokens if len(t) >= 2 and t not in stop_ko and t not in stop_en]
    freq = Counter(filtered)
    # 숫자/단순 숫자열 제외
    items = [(w, c) for w, c in freq.most_common() if not w.isdigit()]
    top = [w for w, _ in items[:top_k]]
    return top


def keywords_to_csv_bytes(keywords):
    """키워드를 CSV 바이트로 변환"""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["rank", "keyword"])
    for i, kw in enumerate(keywords, 1):
        writer.writerow([i, kw])
    return buf.getvalue().encode("utf-8-sig")


def keywords_to_json_bytes(keywords):
    """키워드를 JSON 바이트로 변환"""
    return json.dumps({"keywords": keywords}, ensure_ascii=False, indent=2).encode("utf-8")

# ==================== 파일 저장 ====================
def save_file(content, filename):
    """텍스트 파일 저장"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


# ==================== MAIN ====================
def main():
    load_dotenv(override=True)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return
    
    client = openai.OpenAI(api_key=api_key)
    
    st.title("🎥 통합 비디오 AI 처리 시스템")
    st.markdown("*비디오 → 오디오 추출 → 전사 → 번역 → 이미지 생성 → 키워드 추출*")
    
    # 옵션 선택
    col1, col2, col3 = st.columns(3)
    with col1:
        include_timestamps = st.checkbox("⏱️ 타임스탬프 포함", value=False)
    with col2:
        enable_translation = st.checkbox("🌐 영어 번역", value=True)
    with col3:
        enable_image_gen = st.checkbox("🖼️ AI 이미지 생성", value=False)
    
    uploaded_file = st.file_uploader("비디오 파일 업로드", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)
        
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        audio_path = os.path.join(temp_dir, "extracted_audio.mp3")
        script_path = os.path.join(temp_dir, "transcribed_script.txt")
        translated_script_path = os.path.join(temp_dir, "translated_script.txt")
        video_thumbnail_path = os.path.join(temp_dir, "video_thumbnail.png")
        
        # === 1. 오디오 추출 ===
        with st.spinner("1️⃣ 오디오 추출 중..."):
            audio_extracted = extract_audio(video_path, audio_path)
        
        if not audio_extracted:
            st.error("❌ 오디오 추출 실패")
            return
        
        st.success("✅ 오디오 추출 완료")
        st.audio(audio_path, format='audio/mp3')
        
        # === 2. 음성 → 텍스트 ===
        with st.spinner("2️⃣ 음성 변환 중..."):
            if include_timestamps:
                transcript_data = transcribe_audio_with_timestamps(audio_path, client)
                script_text = format_transcript_with_timestamps(transcript_data) if transcript_data else None
            else:
                script_text = transcribe_audio(audio_path, client)
        
        if not script_text:
            st.error("❌ 음성 변환 실패")
            return
        
        st.success("✅ 음성 변환 완료")
        save_file(script_text, script_path)
        
        # === 3. 번역 ===
        translated_text = None
        if enable_translation:
            with st.spinner("3️⃣ 영어 번역 중..."):
                translated_text = translate_text(client, script_text)
            if translated_text:
                st.success("✅ 번역 완료")
                save_file(translated_text, translated_script_path)
        
        # === 4. AI 이미지 생성 ===
        generated_image_path = None
        if enable_image_gen:
            st.markdown("---")
            with st.spinner("4️⃣ 비디오 대표 이미지 생성 중..."):
                image_prompt = summarize_script_for_image(client, script_text)
                generated_image_path = generate_image_from_text(client, image_prompt, video_thumbnail_path)
            
            if generated_image_path:
                st.success("✅ 이미지 생성 완료")
                st.image(generated_image_path, caption="AI 생성 대표 이미지", use_column_width=True)
        
        # === 5. 키워드 추출 ===
        st.markdown("---")
        st.subheader("🧠 핵심 키워드 추출")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            top_k = st.number_input("키워드 개수", min_value=3, max_value=30, value=10, step=1)
        with col2:
            lang = st.selectbox("언어", ["ko", "en"], index=0)
        with col3:
            use_llm = st.checkbox("LLM 사용(정교, 유료)", value=True, help="체크 해제 시 간단 빈도 기반 추출")
        
        if st.button("키워드 추출"):
            with st.spinner("키워드 분석 중..."):
                if use_llm:
                    keywords = extract_keywords(script_text, client, top_k=int(top_k), language=lang)
                else:
                    keywords = keyword_fallback(script_text, top_k=int(top_k), language=lang)
            
            if keywords:
                st.success("✅ 키워드 추출 완료")
                
                # 태그처럼 표시
                st.markdown("**Top Keywords:** " + " · ".join([f"`{k}`" for k in keywords]))
                
                # 테이블 표시
                st.write({"rank": list(range(1, len(keywords)+1)), "keyword": keywords})
                
                # 다운로드
                csv_bytes = keywords_to_csv_bytes(keywords)
                json_bytes = keywords_to_json_bytes(keywords)
                
                col_k1, col_k2 = st.columns(2)
                with col_k1:
                    st.download_button("📥 키워드 CSV", csv_bytes, "keywords.csv", mime="text/csv")
                with col_k2:
                    st.download_button("📥 키워드 JSON", json_bytes, "keywords.json", mime="application/json")
            else:
                st.warning("키워드를 찾지 못했습니다. 스크립트 내용을 확인해주세요.")
        
        # === 결과 표시 ===
        st.markdown("---")
        st.subheader("📝 원본 스크립트")
        st.text_area("korean_script", script_text, height=200, label_visibility="collapsed")
        
        if translated_text:
            st.subheader("🌐 번역 스크립트 (English)")
            st.text_area("english_script", translated_text, height=200, label_visibility="collapsed")
        
        # === 다운로드 섹션 ===
        st.markdown("---")
        st.subheader("📥 다운로드")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with open(audio_path, 'rb') as f_audio:
                st.download_button("🎵 오디오", f_audio, "audio.mp3", mime="audio/mp3")
        
        with col2:
            st.download_button("📄 원본 스크립트", script_text, "script_kr.txt", mime="text/plain")
        
        if translated_text:
            with col3:
                st.download_button("🌐 번역 스크립트", translated_text, "script_en.txt", mime="text/plain")
        
        # === 다양한 포맷 내보내기 ===
        st.markdown("---")
        st.subheader("📤 다양한 포맷으로 내보내기")
        
        export_type = st.selectbox("포맷 선택", ["Text (.txt)", "Markdown (.md)", "CSV (.csv)"])
        
        if st.button("선택한 포맷으로 다운로드"):
            if export_type == "Text (.txt)":
                st.download_button("📥 TXT 다운로드", script_text, "transcript.txt", mime="text/plain")
            elif export_type == "Markdown (.md)":
                md_content = f"# Video Transcript\n\n{script_text}"
                st.download_button("📥 MD 다운로드", md_content, "transcript.md", mime="text/markdown")
            elif export_type == "CSV (.csv)":
                csv_content = 'timestamp,text\n0:00:00,"' + script_text.replace('"', '""') + '"'
                st.download_button("📥 CSV 다운로드", csv_content, "transcript.csv", mime="text/csv")


if __name__ == "__main__":
    main()