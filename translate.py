from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,pipeline

# 支持的目标语言及其 FLORES-200 代码
LANG_CODE_MAP = {
    "ja": "jpn_Jpan",  # Japanese
    "ko": "kor_Hang",  # Korean
    "fr": "fra_Latn",  # French
    "de": "deu_Latn",  # German
    "en": "eng_Latn",  # English
}

dict_language_translate = {
    "日语":"ja",
    "韩语":"ko",
    "英语":"en",
    "德语":"de",
    "法语":"fr"
}

CACHE_DIR = "./translate_models"
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# 全局懒加载实例
_tokenizer = None
_model = None


def translate(text: str, target: str = '英语', src_lang: str = 'zho_Hans') -> str:
    """
    使用 NLLB-200 将源语言文本翻译为指定目标语言。

    参数:
        text: 待翻译的文本（默认简体中文）
        target: 目标语言键（"ja","ko","fr","de","en"）
        src_lang: 源语言 FLORES-200 代码（默认 "zho_Hans"）

    返回:
        翻译结果字符串
    """
    global _tokenizer, _model

    target = dict_language_translate[target]  # 获取目标语言代码
    
    # 验证目标语言并获取编码
    if target not in LANG_CODE_MAP:
        raise ValueError(f"不支持的目标语言: {target}")
    tgt_lang = LANG_CODE_MAP[target]

    # 初始化 tokenizer 和 model（仅首次调用时）
    if _tokenizer is None or _model is None:
        print("首次加载模型和分词器...")
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            local_files_only=True    # ← 这一行
        )
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            local_files_only=True    # ← 这一行
        )

   # 用 pipeline 包装
    pipe = pipeline(
        "translation",
        model=_model,
        tokenizer=_tokenizer,
        device=0                    # 如果有 GPU 就设成 0，否则省略
    )
    # 解码并返回
    print(f"翻译中: {text} -> {target}")
    result = pipe(
        text,
        src_lang=src_lang,
        tgt_lang=tgt_lang      
    )
    print(f"翻译完成: {result[0]['translation_text']}")
    return result[0]['translation_text']

if __name__ == "__main__":
    # 测试翻译功能
    text = "全民制作人，你们好。我是蔡徐坤。"
    target_lang = "日语"
    print("正在翻译...")
    translation = translate(text, target=target_lang)
    print(f"原文: {text}")
    print(f"翻译 ({target_lang}): {translation}")
