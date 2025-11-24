import os
import json
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from openai import OpenAI
from tqdm import tqdm
import io

# ================= 配置区域 =================
API_KEY = "sk-gfrwjhmclmjqofdvkhqkgkbpfcwvmygvaeuxpkansvtnowuf"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "moonshotai/Kimi-K2-Instruct-0905"

# Matplotlib 字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_json_or_jsonl(file_path):
    """
    智能加载：兼容标准 JSON (列表/对象) 和 JSON Lines
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            return []

        try:
            # 尝试直接加载 (标准 JSON)
            parsed = json.loads(content)
            if isinstance(parsed, list):
                data = parsed
            elif isinstance(parsed, dict):
                data = [parsed]
        except json.JSONDecodeError:
            # 失败则尝试按行加载 (JSONL)
            f.seek(0)
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
    return data

def llm_judge(question, gold, pred):
    """
    调用 LLM 判断正确性
    """
    gold_str = str(gold).strip()
    pred_str = str(pred).strip()

    # 1. 快速通道：完全匹配
    if gold_str == pred_str:
        return True, "Exact Match"

    # 2. LLM 裁判通道
    prompt = f"""
    你是一个客观的阅卷老师。请判断【考生答案】是否符合【标准答案】的意思。

    题目: {question}
    标准答案: {gold_str}
    考生答案: {pred_str}

    判断标准：
    1. 只要考生答案的核心含义与标准答案一致，即为 True。
    2. 忽略标点、大小写或无关的废话（如"答案是..."）。
    3. 如果数值、实体或关键时间点错误，即为 False。

    请仅回复 "True" 或 "False"。
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a judge. Reply only True or False."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.01,
        )
        content = response.choices[0].message.content.strip().lower()
        is_correct = "true" in content
        return is_correct, "LLM Judged"
    except Exception as e:
        return False, f"Error: {str(e)}"

def evaluate_single_file(file_path, progress=gr.Progress()):
    """
    评估单个上传的文件
    """
    filename = os.path.basename(file_path)
    items = load_json_or_jsonl(file_path)

    if not items:
        return f"文件 {filename} 为空或格式错误", None, None

    correct_count = 0
    total_count = 0
    details = []

    progress(0, desc=f"正在评估 {filename}...")

    for idx, item in enumerate(items):
        q = item.get('question', '')
        gold = item.get('golden_answer', '')
        pred = item.get('agent_result', '')

        is_correct, reason = llm_judge(q, gold, pred)

        icon = "✅" if is_correct else "❌"
        detail = {
            "题号": idx + 1,
            "结果": icon,
            "问题": q[:50] + "..." if len(q) > 50 else q,
            "标准答案": str(gold)[:30] + "..." if len(str(gold)) > 30 else str(gold),
            "预测答案": str(pred)[:30] + "..." if len(str(pred)) > 30 else str(pred),
            "判断方式": reason
        }
        details.append(detail)

        if is_correct:
            correct_count += 1
        total_count += 1

        progress((idx + 1) / len(items), desc=f"正在评估 {filename}... ({idx + 1}/{len(items)})")

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    summary = f"""
    📊 **评估结果摘要**

    文件名: {filename}
    总题数: {total_count}
    正确数: {correct_count}
    错误数: {total_count - correct_count}
    正确率: {accuracy:.2f}%
    """

    return summary, accuracy, details, filename

def evaluate_multiple_files(files, progress=gr.Progress()):
    """
    评估多个上传的文件并生成对比图表
    """
    if not files:
        return "请上传至少一个文件", None

    stats = {}
    all_details = []

    for idx, file in enumerate(files):
        progress(idx / len(files), desc=f"处理文件 {idx + 1}/{len(files)}")
        summary, accuracy, details, filename = evaluate_single_file(file.name, progress)

        # 提取文件名（去除扩展名）
        basename = os.path.splitext(filename)[0]
        stats[basename] = accuracy

        all_details.append(f"\n### {filename}\n{summary}\n")

    # 生成图表
    fig = create_comparison_chart(stats)

    combined_summary = "\n".join(all_details)

    return combined_summary, fig

def create_comparison_chart(stats):
    """
    创建对比图表
    """
    if not stats:
        return None

    filenames = list(stats.keys())
    accuracies = list(stats.values())

    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    bars = ax.bar(filenames, accuracies, color='#4CAF50', edgecolor='black', alpha=0.8)

    ax.set_title('测评正确性', fontsize=16, fontweight='bold')
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('正确率 (%)', fontsize=12)
    ax.set_ylim(0, 115)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.tight_layout()

    return fig

# 创建 Gradio 界面
with gr.Blocks(title="测评正确性分析工具", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 📊 测评正确性分析工具

    上传 JSON 或 JSONL 格式的测评结果文件，系统将自动评估正确性并生成可视化报告。

    ### 使用说明：
    1. 选择一个或多个 `.json` 或 `.jsonl` 文件
    2. 点击"开始评估"按钮
    3. 查看评估结果和可视化图表

    ### 文件格式要求：
    每条记录需包含以下字段：
    - `question`: 问题
    - `golden_answer`: 标准答案
    - `agent_result`: 预测答案
    """)

    with gr.Tab("多文件对比评估"):
        with gr.Row():
            file_input_multi = gr.File(
                label="上传文件（支持多选）",
                file_count="multiple",
                file_types=[".json", ".jsonl"]
            )

        eval_btn_multi = gr.Button("🚀 开始评估", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                result_output_multi = gr.Markdown(label="评估结果")
            with gr.Column(scale=1):
                chart_output_multi = gr.Plot(label="测评正确性")

    with gr.Tab("单文件详细评估"):
        with gr.Row():
            file_input_single = gr.File(
                label="上传单个文件",
                file_count="single",
                file_types=[".json", ".jsonl"]
            )

        eval_btn_single = gr.Button("🚀 开始评估", variant="primary", size="lg")

        result_output_single = gr.Markdown(label="评估结果摘要")

    # 绑定事件
    eval_btn_multi.click(
        fn=evaluate_multiple_files,
        inputs=[file_input_multi],
        outputs=[result_output_multi, chart_output_multi]
    )

    eval_btn_single.click(
        fn=lambda file: evaluate_single_file(file.name)[0] if file else "请上传文件",
        inputs=[file_input_single],
        outputs=[result_output_single]
    )

if __name__ == "__main__":
    print("🚀 正在启动 Gradio 测评分析工具...")
    print("📍 请在浏览器中访问: http://127.0.0.1:7860")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True  # 自动打开浏览器
    )
