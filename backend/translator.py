"""
Prompt translator for Japanese to English conversion.
SDXL-Lightning is trained on English prompts, so Japanese prompts need translation.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_japanese(text: str) -> bool:
    """Check if text contains Japanese characters."""
    # Japanese Unicode ranges: Hiragana, Katakana, CJK
    japanese_pattern = re.compile(
        r'[\u3040-\u309F]|'  # Hiragana
        r'[\u30A0-\u30FF]|'  # Katakana
        r'[\u4E00-\u9FAF]|'  # CJK (Kanji)
        r'[\u3000-\u303F]'   # CJK punctuation
    )
    return bool(japanese_pattern.search(text))


def translate_prompt(prompt: str, for_3d: bool = True) -> str:
    """
    Translate Japanese prompt to English for SDXL-Lightning.

    Args:
        prompt: Input prompt (Japanese or English)
        for_3d: If True, add prefix for better 3D object generation

    Returns:
        English prompt for image generation
    """
    if not is_japanese(prompt):
        logger.debug(f"Prompt is already in English: {prompt[:50]}...")
        translated = prompt
    else:
        logger.info(f"Translating Japanese prompt: {prompt[:50]}...")
        translated = _do_translation(prompt)

    # Add 3D-optimized prefix if requested
    if for_3d:
        # Check if prompt already has "single" or similar terms
        lower = translated.lower()
        if not any(word in lower for word in ['single', 'one ', 'a ', 'an ']):
            translated = f"a single {translated}, centered, white background, studio lighting, high quality, 3D render style"
        else:
            translated = f"{translated}, centered, white background, studio lighting, high quality"

    logger.info(f"Final prompt: {translated[:80]}...")
    return translated


def _do_translation(prompt: str) -> str:
    """Internal translation function."""

    try:
        # Try using deep-translator (Google Translate)
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='ja', target='en')
        translated = translator.translate(prompt)
        logger.info(f"Translated to: {translated[:50]}...")
        return translated
    except ImportError:
        logger.warning("deep-translator not installed, trying fallback...")
    except Exception as e:
        logger.warning(f"Google Translator failed: {e}, trying fallback...")

    try:
        # Fallback: Try using argostranslate (offline)
        import argostranslate.translate
        translated = argostranslate.translate.translate(prompt, 'ja', 'en')
        if translated:
            logger.info(f"Translated (argos) to: {translated[:50]}...")
            return translated
    except ImportError:
        logger.warning("argostranslate not installed")
    except Exception as e:
        logger.warning(f"Argos translate failed: {e}")

    # Final fallback: Use a simple dictionary for common terms
    translated = fallback_translate(prompt)
    logger.info(f"Fallback translation: {translated[:50]}...")
    return translated


# Common Japanese to English mappings for 3D/image generation
TRANSLATION_DICT = {
    # Objects
    "車": "car",
    "スポーツカー": "sports car",
    "ロボット": "robot",
    "椅子": "chair",
    "花瓶": "vase",
    "マグ": "mug",
    "カップ": "cup",
    "コーヒー": "coffee",
    "スニーカー": "sneaker",
    "靴": "shoe",
    "時計": "watch",
    "ポケット": "pocket",
    # Materials
    "金属": "metallic",
    "メタリック": "metallic",
    "光沢": "glossy",
    "マット": "matte",
    "木製": "wooden",
    "木": "wood",
    "クリスタル": "crystal",
    "ガラス": "glass",
    "セラミック": "ceramic",
    "陶器": "ceramic",
    "ホログラフィック": "holographic",
    "真鍮": "brass",
    "銅": "copper",
    # Colors
    "赤": "red",
    "青": "blue",
    "緑": "green",
    "黄色": "yellow",
    "白": "white",
    "黒": "black",
    "金": "gold",
    "銀": "silver",
    # Styles
    "モダン": "modern",
    "ミニマリスト": "minimalist",
    "未来的": "futuristic",
    "エレガント": "elegant",
    "かわいい": "cute",
    "フレンドリー": "friendly",
    "スチームパンク": "steampunk",
    # Lighting/Studio
    "スタジオ照明": "studio lighting",
    "スタジオ": "studio",
    "照明": "lighting",
    "反射": "reflective",
    # Design
    "デザイン": "design",
    "コンセプト": "concept",
    "形状": "shape",
    "表面": "surface",
    "素材": "material",
    "詳細": "detailed",
    "精巧": "intricate",
    # Adjectives
    "の": " ",
    "な": " ",
    "、": ", ",
}


def fallback_translate(prompt: str) -> str:
    """
    Simple dictionary-based fallback translation.
    Used when translation services are unavailable.
    """
    result = prompt

    # Sort by length (longest first) to handle compound words first
    sorted_items = sorted(TRANSLATION_DICT.items(), key=lambda x: len(x[0]), reverse=True)

    for jp, en in sorted_items:
        result = result.replace(jp, en)

    # Clean up any remaining Japanese characters by removing them
    # (better than having garbled output)
    result = re.sub(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3000-\u303F]+', ' ', result)

    # Clean up whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    return result if result else "3D object, studio lighting, high quality"
