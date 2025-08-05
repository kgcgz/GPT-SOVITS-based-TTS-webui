import gradio as gr
import webui
import GPT_SoVITS.inference_webui as infui
import translate
import os
import subprocess

# 引入 webui 和 inference_webui 中的所需变量和函数
css = webui.css
js = webui.js
i18n = webui.i18n
translater = translate.translate

# 以下在 inference_webui.py 中定义
html_center = infui.html_center
custom_sort_key = infui.custom_sort_key
GPT_names = infui.GPT_names
SoVITS_names = infui.SoVITS_names
gpt_path = infui.gpt_path
sovits_path = infui.sovits_path
change_choices = infui.change_choices
change_sovits_weights = infui.change_sovits_weights
change_gpt_weights = infui.change_gpt_weights
get_tts_wav = infui.get_tts_wav
dict_language = infui.dict_language
v3v4set = infui.v3v4set
model_version = infui.model_version
dict_language_translate = translate.dict_language_translate


from tools import my_utils
from tools.my_utils import check_details, check_for_existance
from tools.asr.config import asr_dict


p_asr = None  # 全局 ASR 进程
import sys
python_exec = sys.executable or "python"
from webui import open_asr

def open_asr_text(
    asr_inp_dir,
    asr_opt_dir="output/asr_opt",
    asr_model="达摩 ASR (中文)",        # 默认使用达摩 ASR
    asr_model_size="large",
    asr_lang="zh",
    asr_precision="float32",
):
    """
    1) 通过子进程调用 funasr_asr.py
       - 单文件模式下：脚本会直接 print(text)
       - 文件夹模式下：脚本会生成 .list 并 print(list_path)
    2) 读取 .list（如果存在），否则直接返回 stdout
    """
    # 构造命令
    cmd = (
        f'"{python_exec}" '
        f'tools/asr/{asr_dict[asr_model]["path"]} '
        f'-i "{asr_inp_dir}" '
        f'-o "{asr_opt_dir}" '
        f'-s {asr_model_size} '
        f'-l {asr_lang} '
        f'-p {asr_precision}'
    )
    print(f"[ASR] 正在运行: {cmd}")

    # 启动子进程，捕获整个 stdout/stderr
    completed = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if completed.returncode != 0:
        # 打印错误日志，方便调试
        print("[ASR 子进程出错]", completed.stderr)
        # 你也可以选择在这里抛出异常
        return ""

    stdout = completed.stdout.strip()
    print("[ASR 子进程原始输出]", stdout)


    # 试着找到 .list 文件
    base = os.path.basename(asr_inp_dir.rstrip(os.sep))
    list_path = os.path.abspath(
        os.path.join(asr_opt_dir or "output/asr_opt", f"{base}.list")
    )

    if os.path.isfile(list_path):
        # 批量模式，读取 .list 并拼接所有行末尾的文本字段
        texts = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                # 最后一段就是 ASR 文本
                texts.append(parts[-1])
        return "\n".join(texts)

    # 单文件模式，funasr_asr.py print 的就是转写文本
    # 只返回 stdout 的最后一行
    print("[ASR] 单文件模式，翻译完成")
    lines = stdout.splitlines()
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return ""

import base64

def main():
    # 读取本地背景图片并转换为 Base64
    img_path = "background_image/【哲风壁纸】原神-报纸墙-水神.png"  # 替换为你的图片文件名
    # 自动识别文件扩展名设定对应的 MIME 类型
    ext = os.path.splitext(img_path)[1].lower().lstrip('.')
    if ext == 'png':
        mime = 'image/png'
    elif ext in ('jpg', 'jpeg'):
        mime = 'image/jpeg'
    else:
        mime = f'image/{ext}'


    # 读取并编码图片
    with open(img_path, 'rb') as img_file:
        img_bytes = img_file.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    # 用更全面的选择器，并加上 !important
    # 定义统一背景样式，作用于 body 和 Gradio 容器
    background_css = f"""
    html, body, .gradio-container {{
        background-image: url("data:{mime};base64,{img_b64}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        background-color: transparent !important;
    }}
    """


    with gr.Blocks(title="Multipied-Language TTS WebUI", analytics_enabled=False, js=js, css=background_css) as app:

        # 添加页面顶部标题和副标题
        gr.Markdown(
            "<h1 style='text-align:center;font-size:2.5em;'>多语言TTS系统</h1>",
            elem_id="main-title"
        )
        gr.Markdown(
            "<p style='text-align:center;font-size:1.2em;color:#666;'>基于GPT-SOVITS框架</p>",
            elem_id="subtitle"
        )
        # 上半部分：输入与翻译展示区域
        gr.Markdown(html_center(i18n("文本翻译"), "h3"))
        with gr.Row():
        # 左半部分：上下两部分
            with gr.Column(scale=7):
                # 上半部分：源文本输入框
                input_box = gr.Textbox(
                    label=i18n("需要翻译的文本"),
                    placeholder="在此输入要翻译的内容……",
                    lines=5
                )
                # 下半部分：一行内左右布局
                with gr.Row():
                    # 左：语言选择 Radio
                    lang_radio = gr.Radio(
                        choices=list(dict_language_translate.keys()),
                        value="英语",  # 默认选项
                        label=i18n("目标语言")
                    )
                    # 右：开始翻译按钮
                    translate_button = gr.Button(
                        i18n("开始翻译"),
                        variant="primary"
                    )

            # 右半部分：翻译结果输出框
            with gr.Column(scale=6):
                output_box = gr.Textbox(
                    label=i18n("翻译后的文本"),
                    lines=10,
                    interactive=False
                )
        # 将按钮和函数绑定
        translate_button.click(
        fn=translater, 
        inputs=[input_box, lang_radio], 
        outputs=output_box
    )

        # 下半部分：标签页切换
        with gr.Tabs():
            # 第一个 Tab：1-GPT-SoVITS-TTS
            with gr.TabItem(i18n("1-GPT-SoVITS-TTS")):
                
                with gr.Group():
                    gr.Markdown(html_center(i18n("模型切换"), "h3"))
                    with gr.Row():
                        GPT_dropdown = gr.Dropdown(
                            label=i18n("GPT模型列表"),
                            choices=sorted(GPT_names, key=custom_sort_key),
                            value=gpt_path,
                            interactive=True,
                            scale=14,
                        )
                        SoVITS_dropdown = gr.Dropdown(
                            label=i18n("SoVITS模型列表"),
                            choices=sorted(SoVITS_names, key=custom_sort_key),
                            value=sovits_path,
                            interactive=True,
                            scale=14,
                        )
                        refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary", scale=14)
                        refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])

                    gr.Markdown(html_center(i18n("*请上传并填写参考信息"), "h3"))
                    with gr.Row():
                        inp_ref = gr.Audio(label=i18n("请上传3~10秒内参考音频，超过会报错！"), type="filepath", scale=13)
                        with gr.Column(scale=13):
                            ref_text_free = gr.Checkbox(
                                label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。")
                                + i18n("v3暂不支持该模式，使用了会报错。"),
                                value=False,
                                interactive=True if model_version not in v3v4set else False,
                                show_label=True,
                                scale=1,
                            )

                            gr.Markdown(
                                html_center(
                                    i18n("可以选择开启ASR自动语音识别")
                                )
                            )
                            # —— 新增按钮 —— 
                            asr_button = gr.Button(
                                value=i18n("开始语音识别"),
                                scale=1,
                                variant="primary"
                            )



                            gr.Markdown(
                                html_center(
                                    i18n("使用无参考文本模式时建议使用微调的GPT") + "<br>"
                                    + i18n("听不清参考音频说的啥(不晓得写啥)可以开。开启后无视填写的参考文本。"),
                                    "p"
                                )
                            )
                            prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value="", lines=5, max_lines=5, scale=1)

                            

                        with gr.Column(scale=14):
                            prompt_language = gr.Dropdown(
                                label=i18n("参考音频的语种"),
                                choices=list(dict_language.keys()),
                                value=i18n("中文"),
                            )
                            inp_refs = gr.File(
                                label=i18n(
                                    "可选项：通过拖拽多个文件上传多个参考音频（建议同性），平均融合他们的音色。如不填写此项，音色由左侧单个参考音频控制。如是微调模型，建议参考音频全部在微调训练集音色内，底模不用管。"
                                ),
                                file_count="multiple",
                            )
                            sample_steps = gr.Radio(
                                label=i18n("采样步数,如果觉得电,提高试试,如果觉得慢,降低试试"),
                                choices=[4, 8, 16, 32, 64, 128] if model_version == "v3" else [4, 8, 16, 32],
                                value=32 if model_version == "v3" else 8,
                                visible=True,
                            )
                            if_sr_Checkbox = gr.Checkbox(
                                label=i18n("v3输出如果觉得闷可以试试开超分"),
                                value=False,
                                interactive=True,
                                show_label=True,
                                visible=False if model_version != "v3" else True,
                            )
                    
                    # —— 绑定按钮点击事件 —— 
                    asr_button.click(
                        fn=open_asr_text,      # 同步版 ASR 调用函数
                        inputs=[inp_ref],       # 只需要上传的音频路径
                        outputs=[prompt_text]   # 将识别结果写到 prompt_text
                    )
                    
                    
                    gr.Markdown(html_center(i18n("*请填写需要合成的目标文本和语种模式"), "h3"))
                    with gr.Row():
                        with gr.Column(scale=13):
                            text = gr.Textbox(label=i18n("需要合成的文本"), value="", lines=26, max_lines=26)
                        with gr.Column(scale=7):
                            text_language = gr.Dropdown(
                                label=i18n("需要合成的语种") + i18n(".限制范围越小判别效果越好。"),
                                choices=list(dict_language.keys()),
                                value=i18n("中文"),
                                scale=1,
                            )
                            how_to_cut = gr.Dropdown(
                                label=i18n("怎么切"),
                                choices=[
                                    i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"),
                                    i18n("按英文句号.切"), i18n("按标点符号切"),
                                ],
                                value=i18n("凑四句一切"),
                                interactive=True,
                                scale=1,
                            )
                            gr.Markdown(html_center(i18n("语速调整，高为更快")))
                            if_freeze = gr.Checkbox(
                                label=i18n("是否直接对上次合成结果调整语速和音色。防止随机性。"),
                                value=False,
                                interactive=True,
                                show_label=True,
                                scale=1,
                            )
                            with gr.Row():
                                speed = gr.Slider(
                                    minimum=0.6, maximum=1.65, step=0.05,
                                    label=i18n("语速"), value=1, interactive=True, scale=1
                                )
                                pause_second_slider = gr.Slider(
                                    minimum=0.1, maximum=0.5, step=0.01,
                                    label=i18n("句间停顿秒数"), value=0.3, interactive=True, scale=1
                                )
                            gr.Markdown(html_center(i18n("GPT采样参数(无参考文本时不要太低。不懂就用默认)：")))
                            top_k = gr.Slider(
                                minimum=1, maximum=100, step=1,
                                label=i18n("top_k"), value=15, interactive=True, scale=1
                            )
                            top_p = gr.Slider(
                                minimum=0, maximum=1, step=0.05,
                                label=i18n("top_p"), value=1, interactive=True, scale=1
                            )
                            temperature = gr.Slider(
                                minimum=0, maximum=1, step=0.05,
                                label=i18n("temperature"), value=1, interactive=True, scale=1
                            )

                    with gr.Row():
                        inference_button = gr.Button(value=i18n("合成语音"), variant="primary", size="lg", scale=25)
                        output = gr.Audio(label=i18n("输出的语音"), scale=14)

                    inference_button.click(
                        get_tts_wav,
                        [
                            inp_ref, prompt_text, prompt_language, text, text_language,
                            how_to_cut, top_k, top_p, temperature, ref_text_free,
                            speed, if_freeze, inp_refs, sample_steps, if_sr_Checkbox,
                            pause_second_slider,
                        ],
                        [output],
                    )
                    SoVITS_dropdown.change(
                        change_sovits_weights,
                        [SoVITS_dropdown, prompt_language, text_language],
                        [
                            prompt_language, text_language, prompt_text, prompt_language,
                            text, text_language, sample_steps, inp_refs, ref_text_free,
                            if_sr_Checkbox, inference_button,
                        ],
                    )
                    GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])

            # 第二个 Tab：2-cosyvoice-TTS （占位）
            with gr.TabItem(i18n("2-chat-TTS")):
                gr.Markdown(i18n("施工中，敬请期待！"))

        app.launch()

if __name__ == "__main__":
    main()
