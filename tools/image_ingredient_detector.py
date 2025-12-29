import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

# 正确获取环境变量
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


def image_to_ingredients(image_path: str) -> list:
    """
    使用 Qwen-VL-Max 识别图片中的食材
    返回：英文食材列表，如 ["egg", "tomato", "pepper"]
    """
    if not DASHSCOPE_API_KEY:
        raise ValueError("❌ DASHSCOPE_API_KEY 未设置，请在 .env 文件中配置")

    # 读取图片并 base64 编码
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 使用正确的 API 格式
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen-vl-max",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": f"data:image/jpeg;base64,{img_b64}"
                        },
                        {
                            "text": "Please identify all the food ingredients in this image. Return ONLY the ingredient names in English, separated by commas. For example: egg, tomato, onion, broccoli"
                        }
                    ]
                }
            ]
        },
        "parameters": {
            "result_format": "message"
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)

        if response.status_code != 200:
            raise Exception(f"Qwen-VL API 错误: {response.status_code} - {response.text}")

        result = response.json()

        # 解析响应
        output = result.get("output", {})
        choices = output.get("choices", [])

        if not choices:
            raise Exception("API 返回为空")

        message = choices[0].get("message", {})
        content = message.get("content", [])

        # 提取文本内容
        text = ""
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                text = item.get("text")
                break

        if not text:
            raise Exception("未能从响应中提取文本")

        # 清理和分割食材
        # 移除可能的前缀（如 "Ingredients:", "答："等）
        text = text.replace("Ingredients:", "").replace("答:", "").strip()

        # 分割并清理
        ingredients = [item.strip() for item in text.split(",") if item.strip()]

        # 过滤掉非食材词汇
        filtered = []
        skip_words = ["and", "or", "the", "a", "an", "some", "等"]

        for ing in ingredients:
            # 移除括号内容
            ing = ing.split("(")[0].strip()
            # 只保留字母和空格
            ing = ''.join(c for c in ing if c.isalpha() or c.isspace()).strip()
            # 跳过空白和停用词
            if ing and ing.lower() not in skip_words and len(ing) > 1:
                filtered.append(ing.lower())

        return filtered[:10]  # 最多返回10种食材

    except Exception as e:
        raise Exception(f"图像识别失败: {str(e)}")