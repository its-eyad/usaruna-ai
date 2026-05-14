from transformers import pipeline
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct", device_map="auto", torch_dtype="float16")


def get_summary(reviews: List[str], user_lang: str = "en"):

    sum_args = {
    "max_new_tokens": 70,
    "temperature": 0.2,
    "do_sample": True,
    "repetition_penalty": 1.1
    }

    if user_lang == "ar":
        lang_instruction = "Always respond in Arabic. Start with 'بشكل عام، يرى العملاء...'"
        start_phrase = "بشكل عام، يرى العملاء"
    else:
        lang_instruction = "Always respond in English. Start with 'Overall, customers...'"
        start_phrase = "Overall, customers"


    messages = [
        {
            "role": "system", 
            "content": (
                "You are a professional e-commerce marketplace for family businesses analyst. Your task is to summarize reviews with a focus on 'Consensus' and 'Overall meaning' of the reviews."
                f"\nCRITICAL RULE: {lang_instruction}" 
                "\n1. Identify the majority opinion and lead with it."
                "\n2. If most reviews are positive, keep the tone encouraging and highlight the strengths."
                "\n3. Briefly mention any minority concerns (if any) at the end of the sentence as a minor note."
                "\n4. Ensure the output is a single, consistent, sophisticated, and professional sentence."
                "\n5. Avoid repetition and redundant phrases."

            )
        },
        {
            "role": "user", 
            "content": f"Summarize these reviews starting with '{start_phrase}':\n\n{' . '.join(reviews)}"
        }
    ]

    result = pipe(messages, **sum_args)

    return result[0]['generated_text'][-1]['content'] 

def enhance_description(raw_text: str):

    messages = [
        {
            "role": "system", 
            "content": (
                "You are a professional marketing content writer specializing in the e-commerce market for home-based businesses."                
                "\nYour task is to only REFORMAT and ENHANCE descriptions to be attractive and professional."
                "\nCRITICAL RULES:" 
                "\n1. Respond in the same language throughout the whole text (arabic or english)"
                "\n2. Do not add new information."
                "\nSTRICT RULES:"
                "\n1. Ensure the tone is warm and brief."
                "\n2. Ensure the output is sophisticated and professional sentence."
            )
        },
        {
            "role": "user", 
            "content": f"enhance this product description:\n\n{raw_text}"
        }
    ]

    enhance_args = {
    "max_new_tokens": 150,
    "temperature": 0.2,
    "do_sample": True,
    "top_p": 0.9,
    }
    
    result = pipe(messages, **enhance_args)
    final_text = result[0]['generated_text'][-1]['content'].strip()
    
    final_text = final_text.replace("\\n", "\n")
    print(final_text)
    return final_text

class ReviewRequest(BaseModel):
    product_name: str
    product_description: str
    product_details: str  
    customer_name: str
    review_text: str

def generate_reply(data: ReviewRequest):
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a professional Customer Support Assistant specializing in the e-commerce market for home-based businesses."
                f"\nPRODUCT INFO:"
                f"\n- Name: {data.product_name}"
                f"\n- Description: {data.product_description}"
                f"\n- Specific Details: {data.product_details}" 
                f"\n- Customer name:{data.customer_name}"
                "\nCRITICAL RULE: Always respond in the same review language throughout the whole text ONLY"
                "\nINSTRUCTIONS:"
                "\n1. Use the PRODUCT INFO above to answer questions."
                f"\n2. Greet customer usnig {data.customer_name} and keep it nice and short."
                "\n3. Understand then respond to the question/review without follow-up questions"
            )
        },
        {
            "role": "user", 
            "content": f"Customer Review/Question: {data.review_text}"
        }
    ]

    gen_args = {
        "max_new_tokens": 100,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True
    }

    result = pipe(messages, **gen_args)
    return result[0]['generated_text'][-1]['content'].strip()

@app.post("/smart-reply")
async def smart_reply_endpoint(data: ReviewRequest):
    reply = generate_reply(data)
    return {"suggested_reply": reply}

class ProductDesc(BaseModel):
    description: str

@app.post("/enhance")
async def enhance_endpoint(data: ProductDesc):
    enhanced_text = enhance_description(data.description)
    return {"enhanced_description": enhanced_text}


class ReviewData(BaseModel):
    reviews: List[str]
    lang: str = "en"

@app.post("/summarize")
async def summarize_endpoint(data: ReviewData):
    summary = get_summary(data.reviews, data.lang)
    return {"summary": summary}

# uvicorn ai:app --reload       
# cd "C:\Users\eyadx\المستندات\cpit499"
# http://127.0.0.1:8000/docs