import os
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, List, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
import operator
from tools.image_ingredient_detector import image_to_ingredients as detect_ingredients
import requests
import json

load_dotenv()

# 页面配置
st.set_page_config(
    page_title="SmartChef Agent",
    page_icon="🍳",
    layout="wide"
)


# ============================================================================
# 1. 定义工具集 (Tools)
# ============================================================================

@tool
def image_recognition_tool(image_path: str) -> str:
    """
    识别图片中的食材。

    Args:
        image_path: 图片文件路径

    Returns:
        str: 识别到的食材列表（逗号分隔）
    """
    try:
        ingredients = detect_ingredients(image_path)
        result = ", ".join(ingredients)
        return f"✅ 识别到的食材: {result}"
    except Exception as e:
        return f"❌ 图片识别失败: {str(e)}"


@tool
def recipe_search_tool(ingredients: str) -> str:
    """
    根据食材搜索菜谱。

    Args:
        ingredients: 食材列表，用逗号分隔（英文）。例如: "chicken, tomato, onion"

    Returns:
        str: 搜索到的菜谱信息（JSON 格式）
    """
    try:
        api_key = os.getenv("SPOONACULAR_API_KEY")

        # 清理输入
        ing_list = [ing.strip() for ing in ingredients.split(",") if ing.strip()]

        # 调用 API
        url = "https://api.spoonacular.com/recipes/findByIngredients"
        params = {
            "ingredients": ",".join(ing_list),
            "number": 3,
            "ranking": 1,
            "apiKey": api_key
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return f"❌ API 错误: {response.status_code}"

        recipes = response.json()

        if not recipes:
            return "❌ 未找到匹配的菜谱"

        # 格式化结果并返回 JSON
        recipe_data = []
        for recipe in recipes[:3]:
            recipe_data.append({
                "id": recipe['id'],
                "title": recipe['title'],
                "used_ingredients": len(recipe.get('usedIngredients', [])),
                "missed_ingredients": len(recipe.get('missedIngredients', []))
            })

        result = f"✅ 找到 {len(recipes)} 个菜谱:\n\n"
        for i, r in enumerate(recipe_data, 1):
            result += f"{i}. {r['title']} (ID: {r['id']})\n"
            result += f"   - 已有食材: {r['used_ingredients']} 种\n"
            result += f"   - 缺失食材: {r['missed_ingredients']} 种\n"

        result += f"\n📋 菜谱 ID 列表: {json.dumps([r['id'] for r in recipe_data])}\n"
        result += f"⚠️ 重要: 你必须为每个菜谱 ID 调用 recipe_detail_tool 获取详情"

        return result

    except Exception as e:
        return f"❌ 搜索失败: {str(e)}"


@tool
def recipe_detail_tool(recipe_id: int) -> str:
    """
    获取菜谱的详细信息（步骤和营养）。

    Args:
        recipe_id: 菜谱 ID

    Returns:
        str: 菜谱详细信息
    """
    try:
        api_key = os.getenv("SPOONACULAR_API_KEY")

        url = f"https://api.spoonacular.com/recipes/{recipe_id}/information"
        params = {
            "includeNutrition": True,
            "apiKey": api_key
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        # 提取步骤
        steps = data.get("analyzedInstructions", [{}])[0].get("steps", [])
        steps_text = "\n".join([f"  {i}. {s['step']}" for i, s in enumerate(steps[:5], 1)])

        # 提取营养
        nutrition = data.get("nutrition", {}).get("nutrients", [])
        nutri_map = {n["name"]: n["amount"] for n in nutrition}

        result = f"📖 {data['title']}\n\n"
        result += f"📋 烹饪步骤:\n{steps_text}\n"
        if len(steps) > 5:
            result += f"  ...(还有 {len(steps) - 5} 步)\n"

        result += f"\n🥗 营养成分:\n"
        result += f"  - 热量: {nutri_map.get('Calories', 0):.0f} kcal\n"
        result += f"  - 蛋白质: {nutri_map.get('Protein', 0):.0f}g\n"
        result += f"  - 碳水: {nutri_map.get('Carbohydrates', 0):.0f}g\n"
        result += f"  - 脂肪: {nutri_map.get('Fat', 0):.0f}g\n"

        return result

    except Exception as e:
        return f"❌ 获取详情失败: {str(e)}"


# 工具列表
tools = [image_recognition_tool, recipe_search_tool, recipe_detail_tool]


# ============================================================================
# 2. 定义 Agent 状态
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    pending_recipe_ids: List[int]  # 待获取的菜谱 ID
    fetched_recipe_ids: List[int]  # 已获取的菜谱 ID


# ============================================================================
# 3. 初始化 LLM（大脑）
# ============================================================================

llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.3
)

# 绑定工具到 LLM
llm_with_tools = llm.bind_tools(tools)


# ============================================================================
# 4. 定义 Agent 节点
# ============================================================================

def agent_node(state: AgentState) -> AgentState:
    """Agent 决策节点：LLM 决定下一步做什么"""

    # 系统提示
    system_prompt = """你是一个智能厨师助手 Agent。你的任务是：
1. 理解用户需求
2. 智能地选择和调用工具
3. 根据工具返回的结果继续决策
4. 最终为用户提供完整的菜谱推荐

可用工具:
- image_recognition_tool: 识别图片中的食材
- recipe_search_tool: 根据食材搜索菜谱
- recipe_detail_tool: 获取菜谱详细信息

【重要】工作流程要求:
1. 如果用户上传了图片，先调用 image_recognition_tool
2. 拿到食材后，调用 recipe_search_tool 搜索菜谱
3. **必须**为 recipe_search_tool 返回的**所有菜谱**（通常是3个）逐一调用 recipe_detail_tool 获取详情
4. 只有当获取了所有菜谱的详细信息后，才能整理结果并给用户完整的推荐
5. 不要只获取一个菜谱就停止，用户需要看到所有推荐的菜谱详情

请根据当前对话历史，智能地决定下一步行动。"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    # LLM 决策
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


def should_continue(state: AgentState):
    """判断是否继续执行"""
    last_message = state["messages"][-1]

    # 如果有工具调用，继续执行
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # 否则结束
    return "end"


# ============================================================================
# 5. 构建 Agent 图
# ============================================================================

@st.cache_resource
def create_agent_graph():
    """创建 Agent 工作流图"""

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    # 添加边
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ============================================================================
# 6. Streamlit UI
# ============================================================================

def main():
    st.title("🍳 SmartChef Agent")
    st.markdown("### 基于 LLM 的智能厨师助手")

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统架构")
        st.markdown("""
        **🧠 大脑 (Controller)**
        - Qwen-Plus LLM
        - 动态决策工具调用

        **🔧 工具集 (Tools)**
        1. 图片识别工具
        2. 菜谱搜索工具
        3. 菜谱详情工具

        **📄 编排 (LangGraph)**
        - 智能工作流
        - 自主决策循环
        """)

        st.divider()

        st.header("📊 API 状态")
        if os.getenv('DASHSCOPE_API_KEY'):
            st.success("✅ DashScope (LLM + Vision)")
        else:
            st.error("❌ DashScope 未配置")

        if os.getenv('SPOONACULAR_API_KEY'):
            st.success("✅ Spoonacular (Recipe)")
        else:
            st.error("❌ Spoonacular 未配置")

        st.divider()

        st.header("📖 使用说明")
        st.markdown("""
        ### 输入方式：

        **📝 文本**
        - 描述你的需求
        - 例如：我有鸡肉和土豆，推荐菜谱

        **📸 图片**
        - 上传食材照片
        - AI 自动识别并推荐

        ### Agent 特点：
        - ✅ 自主决策调用工具
        - ✅ 完整的思考链日志
        - ✅ 动态规划任务流程
        """)

    # 主界面
    st.divider()

    # 输入区域
    tab1, tab2 = st.tabs(["📝 文本输入", "📸 图片上传"])

    user_input = None
    image_path = None

    with tab1:
        text_input = st.text_area(
            "描述你的需求",
            placeholder="例如：我有鸡肉、番茄和洋葱，推荐几道菜",
            height=100
        )

        # 快速示例
        st.markdown("**💡 快速示例：**")
        col1, col2, col3 = st.columns(3)

        if col1.button("🍗 鸡肉 + 土豆", use_container_width=True):
            st.session_state.quick_input = "我有鸡肉和土豆，请推荐菜谱"
            st.rerun()

        if col2.button("🥚 鸡蛋 + 番茄", use_container_width=True):
            st.session_state.quick_input = "我有鸡蛋、番茄和洋葱，推荐菜谱"
            st.rerun()

        if col3.button("🥩 牛肉 + 蔬菜", use_container_width=True):
            st.session_state.quick_input = "我有牛肉、胡萝卜和芹菜，推荐菜谱"
            st.rerun()

        # 使用快速输入
        if 'quick_input' in st.session_state:
            text_input = st.session_state.quick_input
            del st.session_state.quick_input

        if text_input:
            user_input = text_input

    with tab2:
        uploaded_file = st.file_uploader(
            "上传食材图片",
            type=['png', 'jpg', 'jpeg']
        )

        if uploaded_file:
            st.image(uploaded_file, caption="上传的图片", use_column_width=True)

            # 保存临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                image_path = tmp_file.name

            user_input = "请识别这张图片中的食材，并推荐菜谱"

    # 运行按钮
    st.divider()

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        run_button = st.button("🚀 启动 Agent", type="primary", use_container_width=True)

    if run_button:
        if not user_input:
            st.error("⚠️ 请输入需求或上传图片")
            return

        st.divider()
        st.header("🤖 Agent 工作日志")

        # 准备初始消息
        if image_path:
            initial_message = HumanMessage(
                content=f"{user_input}\n图片路径: {image_path}"
            )
        else:
            initial_message = HumanMessage(content=user_input)

        # 初始化状态
        initial_state = {
            "messages": [initial_message],
            "pending_recipe_ids": [],
            "fetched_recipe_ids": []
        }

        # 创建 Agent
        agent = create_agent_graph()

        # 运行 Agent
        with st.spinner("Agent 正在思考和执行..."):
            try:
                step_count = 0
                max_steps = 20  # 增加最大步数

                log_container = st.container()

                for step in agent.stream(initial_state, {"recursion_limit": max_steps}):
                    step_count += 1

                    with log_container:
                        st.write(f"**--- 步骤 {step_count} ---**")

                        for node_name, node_output in step.items():
                            if node_name == "agent":
                                message = node_output["messages"][0]

                                # 显示工具调用
                                if hasattr(message, 'tool_calls') and message.tool_calls:
                                    for tool_call in message.tool_calls:
                                        st.success(f"🔧 **调用工具**: `{tool_call['name']}`")
                                        with st.expander("查看参数", expanded=False):
                                            st.json(tool_call['args'])

                            elif node_name == "tools":
                                for msg in node_output["messages"]:
                                    with st.expander("📊 **工具返回**（点击展开）", expanded=True):
                                        st.text(msg.content)

                        st.divider()

                # 显示最终结果
                st.header("✅ 最终推荐")
                final_message = list(step.values())[0]["messages"][-1]

                if hasattr(final_message, 'content'):
                    st.markdown(final_message.content)

            except Exception as e:
                st.error(f"❌ Agent 执行错误: {str(e)}")
                import traceback
                with st.expander("查看错误详情"):
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()