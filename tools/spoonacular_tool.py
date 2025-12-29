import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# ✅ 正确获取环境变量
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")


def search_recipes_by_ingredients(ingredients: List[str]) -> List[Dict]:
    """
    调用 Spoonacular API 搜索食谱
    返回：包含菜谱名称、ID、步骤、营养信息的列表
    """
    if not SPOONACULAR_API_KEY:
        raise ValueError("❌ SPOONACULAR_API_KEY 未设置，请在 .env 文件中配置")

    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "ingredients": ",".join(ingredients),
        "number": 3,
        "ranking": 1,  # 优先匹配最多食材
        "apiKey": SPOONACULAR_API_KEY
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise Exception(f"Spoonacular API 错误: {resp.status_code} - {resp.text}")

    recipes = []
    for item in resp.json():
        # 获取详细信息（含步骤和营养）
        detail = get_recipe_detail(item["id"])
        recipes.append(detail)
    return recipes


def get_recipe_detail(recipe_id: int) -> Dict:
    """获取单个菜谱的完整信息"""
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/information"
    params = {
        "includeNutrition": True,
        "apiKey": SPOONACULAR_API_KEY
    }
    resp = requests.get(url, params=params)
    data = resp.json()

    # 提取关键字段
    nutrition = data.get("nutrition", {}).get("nutrients", [])
    nutri_map = {n["name"]: n["amount"] for n in nutrition}

    return {
        "name": data["title"],
        "steps": [step["step"] for step in data.get("analyzedInstructions", [{}])[0].get("steps", [])],
        "calories": round(nutri_map.get("Calories", 0)),
        "protein_g": round(nutri_map.get("Protein", 0)),
        "carbs_g": round(nutri_map.get("Carbohydrates", 0)),
        "fat_g": round(nutri_map.get("Fat", 0))
    }