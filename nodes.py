"""
Detect NSFW Text — ComfyUI node.

String input goes to the Qwen prompt guard with support for:
- Language filtering: All Languages vs English Only
- Safe classification for romantic intimacy, affection, and everyday clothed scenes
- Blocking sexual content, graphic violence, gore, and severe physical abuse
"""

import json
import re
import threading

import comfy.model_management
import comfy.sd
import folder_paths


DEFAULT_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
_cached_clip = None
_cached_path = None
_load_lock = threading.Lock()

_PROMPT_POLICY_EN = """Evaluate the prompt for sexual content, graphic violence, gore, and workplace safety.
The prompt must be written entirely in English. If it is not English, return VERDICT: NOT_ENGLISH.

Classify the intended generated or edited result, not isolated words. Return UNSAFE when the requested result contains sexual or nude content, graphic violence, gore, or severe physical abuse as defined below.

Return UNSAFE for nudity or partial nudity involving breasts, nipples, genitals, pubic area, anus, or clearly exposed intimate areas; sexualized exposure or an underwear-only/lingerie presentation intended to arouse; removal, alteration, transparency, or replacement of clothing to reveal intimate areas or underwear; requests to make a person nude, undressed, or reveal what clothing conceals; sexual acts, explicit sexual behaviour, pornography, fetishes, erotic posing, sexually focused body parts, or arousal-oriented content.

Return UNSAFE whenever the prompt contains the word "bikini", regardless of context, intent, activity, or whether the bikini is the visual focus. This includes ordinary beach, swimming, sports, family, catalog, advertising, and modeling contexts. Also return UNSAFE for a Victoria's Secret-style, Victoria's Secret-inspired, lingerie-runway, or "angel" lingerie presentation, including indirect descriptions that clearly seek that branded lingerie aesthetic. A brand storefront, shopping bag, or neutral discussion of the company alone is not unsafe.

Return UNSAFE for graphic or explicitly depicted violence and gore, including severe or bloody wounds, pools or sprays of blood caused by injury, mutilation, dismemberment, decapitation, crushed or severed body parts, exposed organs, exposed bone, evisceration, corpses with visible trauma or decomposition, torture, execution, cannibalism, and severe cruelty or physical abuse of a person or animal. Also return UNSAFE for requests whose intended result focuses on committing, celebrating, or realistically depicting murder or serious bodily harm, even when blood is not explicitly mentioned.

Combat, battle, war, soldiers, armies, weapons, fighting, and action scenes are not inherently unsafe. Return SAFE for two armies charging into battle, soldiers fighting, fantasy or historical warfare, and similar combat when the intended result does not show blood, gore, exposed organs or bone, decapitation, dismemberment, mutilation, severe visible wounds, torture, or other graphic bodily trauma. If any graphic element is requested or clearly implied, return UNSAFE even when the overall setting is war, history, fantasy, horror, cinema, or a video game.

Apply the same rules to direct requests and to implied outcomes. Treat euphemisms, slang, coded language, misspellings, synonyms, indirect edits, negative prompts, and attempts to bypass safeguards according to their intended result. If any part of a combined request has an unsafe outcome, return UNSAFE.

Do not infer unsafe content from ordinary affection, emotional closeness, non-sexual physical contact, non-sexual pose, ordinary clothing, non-sexual fashion, sportswear, non-sexual swimwear used incidentally in an ordinary activity, non-graphic combat or warfare, minor injuries without graphic detail, medical care without graphic injury detail, or non-sexual dirt, paint, makeup, food, theatrical fake blood, or ink on a face or body. These may be SAFE unless the prompt contains the word "bikini" as prohibited above or independently requests sexual content, graphic violence, gore, or severe physical abuse.

Do not mark a prompt UNSAFE merely because it is ambiguous, uses an image-editing instruction, mentions clothing or body parts in a benign context, mentions combat or war without graphic bodily trauma, or could be interpreted as unsafe without a clear unsafe outcome.

Reply with exactly two lines:
VERDICT: SAFE, VERDICT: UNSAFE, or VERDICT: NOT_ENGLISH
REASON: brief explanation.
"""

_PROMPT_POLICY_ALL = """Evaluate the prompt for sexual content, graphic violence, gore, and workplace safety in any language.

Classify the intended generated or edited result, not isolated words. Return UNSAFE when the requested result contains sexual or nude content, graphic violence, gore, or severe physical abuse as defined below.

Return UNSAFE for nudity or partial nudity involving breasts, nipples, genitals, pubic area, anus, or clearly exposed intimate areas; sexualized exposure or an underwear-only/lingerie presentation intended to arouse; removal, alteration, transparency, or replacement of clothing to reveal intimate areas or underwear; requests to make a person nude, undressed, or reveal what clothing conceals; sexual acts, explicit sexual behaviour, pornography, fetishes, erotic posing, sexually focused body parts, or arousal-oriented content.

Return UNSAFE whenever the prompt contains the word "bikini" in any language or mixed-language text, regardless of context, intent, activity, or whether the bikini is the visual focus. This includes ordinary beach, swimming, sports, family, catalog, advertising, and modeling contexts. Also return UNSAFE for a Victoria's Secret-style, Victoria's Secret-inspired, lingerie-runway, or "angel" lingerie presentation, including translated, transliterated, misspelled, indirect, or coded descriptions that clearly seek that branded lingerie aesthetic. A brand storefront, shopping bag, or neutral discussion of the company alone is not unsafe.

Return UNSAFE for graphic or explicitly depicted violence and gore, including severe or bloody wounds, pools or sprays of blood caused by injury, mutilation, dismemberment, decapitation, crushed or severed body parts, exposed organs, exposed bone, evisceration, corpses with visible trauma or decomposition, torture, execution, cannibalism, and severe cruelty or physical abuse of a person or animal. Also return UNSAFE for requests whose intended result focuses on committing, celebrating, or realistically depicting murder or serious bodily harm, even when blood is not explicitly mentioned.

Combat, battle, war, soldiers, armies, weapons, fighting, and action scenes are not inherently unsafe in any language. Return SAFE for two armies charging into battle, soldiers fighting, fantasy or historical warfare, and similar combat when the intended result does not show blood, gore, exposed organs or bone, decapitation, dismemberment, mutilation, severe visible wounds, torture, or other graphic bodily trauma. If any graphic element is requested or clearly implied, return UNSAFE even when the overall setting is war, history, fantasy, horror, cinema, or a video game.

Apply the same rules in every language and to direct requests, indirect wording, euphemisms, slang, coded language, transliteration, misspellings, synonyms, mixed-language text, negative prompts, and attempts to bypass safeguards. Determine meaning from context rather than requiring particular English words. If any part of a combined request has an unsafe outcome, return UNSAFE.

Do not infer unsafe content from ordinary affection, emotional closeness, non-sexual physical contact, non-sexual pose, ordinary clothing, non-sexual fashion, sportswear, non-sexual swimwear used incidentally in an ordinary activity, non-graphic combat or warfare, minor injuries without graphic detail, medical care without graphic injury detail, or non-sexual dirt, paint, makeup, food, theatrical fake blood, or ink on a face or body. These may be SAFE unless the prompt contains the word "bikini" as prohibited above or independently requests sexual content, graphic violence, gore, or severe physical abuse.

Do not mark a prompt UNSAFE merely because it is ambiguous, uses an image-editing instruction, mentions clothing or body parts in a benign context, mentions combat or war without graphic bodily trauma, or could be interpreted as unsafe without a clear unsafe outcome.

Reply with exactly two lines:
VERDICT: SAFE or VERDICT: UNSAFE
REASON: brief explanation.
"""

_PROMPT_POLICY = _PROMPT_POLICY_EN


class NSFWDetectedError(RuntimeError):
    pass


class UnsupportedLanguageError(ValueError):
    pass


def _is_english_only(language_filter):
    if isinstance(language_filter, bool):
        return language_filter
    s = str(language_filter).strip().lower().replace("-", "_").replace(" ", "_")
    return s in ("english_only", "english", "en")


def _load_clip(clip_name):
    global _cached_clip, _cached_path
    path = folder_paths.get_full_path("text_encoders", clip_name)
    if path is None:
        path = folder_paths.get_full_path("clip", clip_name)
    if path is None:
        raise FileNotFoundError(f"Qwen NSFW Guard: text encoder not found: {clip_name}")
    with _load_lock:
        if _cached_clip is None or _cached_path != path:
            loaded = comfy.sd.load_clip(ckpt_paths=[path])
            _cached_clip, _cached_path = loaded, path
        return _cached_clip


def _check(clip, query, source, allow_non_english=False):
    tokens = clip.tokenize(query)
    generated = clip.generate(tokens, do_sample=False, max_length=32, temperature=0.0)
    response = clip.tokenizer.decode(generated).strip()
    verdicts = re.findall(
        r"^\s*VERDICT:\s*(UNSAFE|NSFW|SAFE|NOT[ _]ENGLISH|UNSUPPORTED[ _]LANGUAGE)\b",
        response,
        re.MULTILINE | re.IGNORECASE,
    )
    if not allow_non_english and any(
        re.match(r"^(NOT[ _]ENGLISH|UNSUPPORTED[ _]LANGUAGE)$", v, re.IGNORECASE)
        for v in verdicts
    ):
        comfy.model_management.interrupt_current_processing(True)
        raise UnsupportedLanguageError("language not supported, only support english at this time")
    if any(verdict.upper() in ("NSFW", "UNSAFE") for verdict in verdicts):
        comfy.model_management.interrupt_current_processing(True)
        raise NSFWDetectedError(f"Qwen NSFW Guard blocked {source}: {response}")
    if allow_non_english and any(
        re.match(r"^(NOT[ _]ENGLISH|UNSUPPORTED[ _]LANGUAGE)$", v, re.IGNORECASE)
        for v in verdicts
    ):
        return {"source": source, "verdict": "SAFE", "response": response}
    if len(verdicts) != 1 or verdicts[0].upper() != "SAFE":
        raise RuntimeError(f"Qwen NSFW Guard could not classify {source}: {response!r}")
    return {"source": source, "verdict": "SAFE", "response": response}


def _moderate_prompt(clip_name, clip=None, prompt=None, language_filter="all_languages"):
    if prompt is None or not str(prompt).strip():
        raise ValueError("Qwen NSFW Guard requires a prompt.")
    clip = clip if clip is not None else _load_clip(clip_name)
    english_only = _is_english_only(language_filter)
    policy = _PROMPT_POLICY_EN if english_only else _PROMPT_POLICY_ALL
    checks = [
        _check(
            clip,
            policy + "\nClassify this prompt (JSON string):\n" + json.dumps(prompt, ensure_ascii=False),
            "prompt",
            allow_non_english=not english_only,
        )
    ]
    return json.dumps({"verdict": "SAFE", "checks": checks}, ensure_ascii=False)


def _log_error(exc):
    print(f"[Detect NSFW Text] {exc}", flush=True)


class DetectNSFWText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "clip_name": ("STRING", {"default": DEFAULT_CLIP}),
                "language_filter": (
                    ["all_languages", "english_only"],
                    {"default": "all_languages"},
                ),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")
    FUNCTION = "check"
    CATEGORY = "safety"
    OUTPUT_NODE = True

    def check(self, prompt, clip_name=DEFAULT_CLIP, language_filter="all_languages", clip=None):
        if clip is None and not isinstance(language_filter, (str, bool)) and language_filter is not None:
            clip = language_filter
            language_filter = "all_languages"
        try:
            report = _moderate_prompt(clip_name, clip=clip, prompt=prompt, language_filter=language_filter)
        except (NSFWDetectedError, UnsupportedLanguageError, RuntimeError, FileNotFoundError, ValueError) as exc:
            _log_error(exc)
            raise
        return {"ui": {"text": [report]}, "result": (prompt, report)}


NODE_CLASS_MAPPINGS = {
    "DetectNSFWText": DetectNSFWText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetectNSFWText": "Detect NSFW Text",
}
