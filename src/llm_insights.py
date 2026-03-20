import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_insight(churn_prob, top_reasons, model_used):

    reasons_text = "\n".join([f"- {r}" for r in top_reasons])

    prompt = f"""
    You are a customer retention analyst at a telecom company.
    
    A customer has been flagged by our ML system:
    
    Churn probability: {churn_prob:.0%}
    Model used: {model_used}
    Key risk factors:
    {reasons_text}
    
    The model is uncertain about this customer (borderline case).
    
    Provide:
    1. A 2-sentence plain English explanation of WHY this customer might churn
    2. ONE specific retention action to take immediately
    
    Be specific and concise. No fluff.
    Format: [Explanation]. [Action].
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()