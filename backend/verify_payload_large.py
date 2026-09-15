import asyncio
import httpx
import os
import time
from dotenv import load_dotenv

async def run():
    load_dotenv('.env')
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"
    
    system_prompt = (
        "You are the Vision Module for SatQuery AI, an expert geospatial intelligence system. "
        "Analyze the provided satellite imagery with a highly professional, technical, and remote-sensing specific tone. "
        "Provide detailed, comprehensive observations about land use, infrastructure, geography, and anomalies. "
        "Do not be brief. Expand on your findings and provide a rich, multi-sentence analysis.\n\n"
        f"User Query: describe the image"
    )
    
    parts = [{"text": system_prompt}]
    
    # 1MB fake image (to simulate satellite imagery payload)
    img_data = "A" * 1024 * 1024
    
    parts.append({
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": img_data
        }
    })
    
    payload = {
        "contents": [
            {
                "parts": parts
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 1200
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("Sending request...")
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            print("Status:", resp.status_code)
            print("Body:", resp.text)
    except Exception as e:
        print("Error:", e)
    finally:
        print("Elapsed:", time.time() - t0)

asyncio.run(run())
