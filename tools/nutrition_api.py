# nutrition_api.py
import os
import requests
from typing import Dict, Any


def get_nutrition_info(recipe_name: str) -> Dict[str, Any]:
    """
    使用 USDA FoodData Central API 获取菜谱的营养信息
    """
    api_key = os.getenv("USDA_API_KEY")
    if not api_key:
        raise ValueError("USDA_API_KEY 未在 .env 中设置")

    # 第一步：通过菜谱名称搜索食品
    search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "api_key": api_key,
        "query": recipe_name.strip(),
        "pageSize": 1,  # 只取最匹配的一个
        "dataType": ["Foundation", "SR Legacy"]  # 优先使用权威数据
    }

    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get("foods"):
            # 如果没找到，返回默认值
            return {"calories": "N/A", "protein": "N/A"}

        # 获取第一个结果的 FDC ID
        fdc_id = data["foods"][0]["fdcId"]

        # 第二步：用 FDC ID 获取详细营养信息
        detail_url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
        detail_params = {"api_key": api_key}
        detail_response = requests.get(detail_url, params=detail_params, timeout=10)
        detail_response.raise_for_status()
        food_detail = detail_response.json()

        # 提取营养数据
        nutrients = food_detail.get("foodNutrients", [])
        calories = "N/A"
        protein = "N/A"

        for nutrient in nutrients:
            name = nutrient.get("nutrient", {}).get("name", "").lower()
            value = nutrient.get("amount", 0)

            if "energy" in name and "calorie" in name:
                calories = round(value, 1)
            elif "protein" in name:
                protein = round(value, 1)

        return {
            "calories": calories,
            "protein": protein
        }

    except Exception as e:
        print(f"⚠️ USDA API 调用失败: {str(e)}")
        return {"calories": "N/A", "protein": "N/A"}