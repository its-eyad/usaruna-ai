import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# أضفنا allow_credentials و حددنا الرابط بشكل أوضح
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",    # رابط Vite المحلي
        "http://127.0.0.1:5173",
        "https://usaruna-ai.onrender.com" # رابط السيرفر نفسه
    ],
    allow_credentials=True,
    allow_methods=["*"],           # يسمح بـ POST, GET, OPTIONS, إلخ
    allow_headers=["*"],           # يسمح بكل أنواع الـ Headers
)
# تفعيل CORS عشان الفرونت إند (Vite) يقدر يتصل بالسيرفر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في الإنتاج، يفضل تحديد رابط موقعك على Render
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- إعدادات Hugging Face API ---
# نصيحة: حط التوكن في الـ Environment Variables في Render باسم HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/v1/chat/completions"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_hf_api(messages, params):
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": messages,
        "max_tokens": params.get("max_new_tokens", 150),
        "temperature": params.get("temperature", 0.5),
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    return result['choices'][0]['message']['content']

# --- دالة تلخيص المراجعات ---
def get_summary(reviews: List[str], user_lang: str = "en"):
    params = {"max_new_tokens": 100, "temperature": 0.2}

    if user_lang == "ar":
        lang_instruction = "Always respond in Arabic. Start with 'بشكل عام، يرى العملاء...'"
        start_phrase = "بشكل عام، يرى العملاء"
    else:
        lang_instruction = "Always respond in English. Start with 'Overall, customers...'"
        start_phrase = "Overall, customers"

    messages = [
        {"role": "system", "content": f"You are a professional analyst. {lang_instruction} Summarize strictly in ONE sentence."},
        {"role": "user", "content": f"Summarize these reviews starting with '{start_phrase}':\n\n{' . '.join(reviews)}"}
    ]

    return query_hf_api(messages, params)

# --- دالة تحسين الوصف ---
def enhance_description(raw_text: str):
    params = {"max_new_tokens": 250, "temperature": 0.5}

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional marketing writer. REFORMAT and ENHANCE the description."
                "\n1. Same language as input. 2. Warm tone. 3. Bullet points and emojis. 4. No new facts."
            )
        },
        {"role": "user", "content": f"Enhance this product description:\n\n{raw_text}"}
    ]

    return query_hf_api(messages, params).replace("\\n", "\n")

# --- دالة الرد الذكي ---
class ReviewRequest(BaseModel):
    product_name: str
    product_description: str
    product_details: str
    customer_name: str
    review_text: str

def generate_reply(data: ReviewRequest):
    params = {"max_new_tokens": 150, "temperature": 0.3}

    messages = [
        {
            "role": "system",
            "content": (
                f"You are customer support for 'Osruna'. INFO: Name: {data.product_name}, Desc: {data.product_description}, Details: {data.product_details}."
                f"\nGreet {data.customer_name}. Respond ONLY in the customer's language. Be short. No follow-up questions."
            )
        },
        {"role": "user", "content": data.review_text}
    ]

    return query_hf_api(messages, params).strip()

# --- Endpoints ---

@app.get("/")
async def root():
    return {"status": "success", "message": "Osruna AI is Live!"}
    
@app.post("/smart-reply")
async def smart_reply_endpoint(data: ReviewRequest):
    return {"suggested_reply": generate_reply(data)}

class ProductDesc(BaseModel):
    description: str

@app.post("/enhance")
async def enhance_endpoint(data: ProductDesc):
    return {"enhanced_description": enhance_description(data.description)}

class ReviewData(BaseModel):
    reviews: List[str]
    lang: str = "en"

@app.post("/summarize")
async def summarize_endpoint(data: ReviewData):
    return {"summary": get_summary(data.reviews, data.lang)}
