"""Strict, deterministic cleaning for the selected Wikisource editions."""

import re

from corpus_pipeline.catalogue import (
    DAODEJING,
    JOURNEY_TO_THE_WEST,
    CorpusSpec,
)
from corpus_pipeline.mediawiki import validate_page_provenance
from corpus_pipeline.models import (
    CorpusValidationError,
    SourcePage,
    TextSection,
)


CLEANER_VERSION = "wikisource-wikitext-v4"

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_CHAPTER_HEADING = re.compile(
    r"(?m)^[ \t]*===[ \t]*"
    r"([〇零一二三四五六七八九十百兩]+)"
    r"章[ \t]*===[ \t]*$"
)
_ANY_HEADING = re.compile(
    r"(?m)^[ \t]*={2,6}.*?={2,6}[ \t]*$"
)
_CATEGORY_LINK = re.compile(
    r"\[\[(?:Category|分類):[^\[\]]*\]\]",
    re.IGNORECASE,
)
_WIKILINK = re.compile(r"\[\[([^\[\]]+)\]\]")
_EXTERNAL_LINK = re.compile(r"\[(?:https?|ftp)://[^\]]+\]")
_LANGUAGE_CONVERSION = re.compile(r"-\{(.*?)}-", re.DOTALL)
_TAG = re.compile(r"<\s*(/?)\s*([A-Za-z0-9]+)(?:\s[^>]*)?>")
_MAGIC_WORD = re.compile(r"__[A-Z][A-Z0-9_]*__")
_HORIZONTAL_RULE = re.compile(r"(?m)^[ \t]*----+[ \t]*$")
_INDENT = re.compile(r"(?m)^[ \t]*:+[ \t]?")
_TRAILING_SPACE = re.compile(r"(?m)[ \t]+$")
_EXCESS_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")

_FIRST_ARGUMENT_TEMPLATES = frozenset({
    "另",
    "参",
    "參",
})
_DISCARD_TEMPLATES = frozenset({
    "alsosee",
    "cjk-new-char",
    "col-begin",
    "col-break",
    "col-end",
    "footer",
    "header",
    "pd-old",
    "textquality",
    "wikipedia",
    "檢索",
})
_ALLOWED_CONTAINER_TAGS = frozenset({
    "onlyinclude",
    "poem",
})
_INTERLANGUAGE_NAMESPACES = frozenset({
    "cs",
    "de",
    "en",
    "es",
    "fi",
    "fr",
    "it",
    "ko",
    "pt",
    "vi",
})

# These selected Traditional pages contain a few isolated variant or
# simplified forms. Normalize only the exact, manually audited phrases:
# character-wide conversion could corrupt a legitimate future occurrence of
# 台, 膻, or another historically distinct form. Broad OpenCC conversion
# would also damage legitimate classical forms such as 后, 云, 里, 凶, 采,
# 几, 帘, and 价.
_DAODEJING_ORTHOGRAPHY = (
    ("春登台", "春登臺"),
    ("其事好还", "其事好還"),
    ("九層之台", "九層之臺"),
)
_JOURNEY_ORTHOGRAPHY = (
    ("攛將上来", "攛將上來"),
    ("獅驼王", "獅駝王"),
    ("腥膻", "腥羶"),
    ("上腭子", "上齶子"),
    ("跨了刬馬", "跨了剗馬"),
    ("跨著刬馬", "跨著剗馬"),
    ("醃了晒乾", "醃了曬乾"),
    ("晾晒衣服", "晾曬衣服"),
    ("要晒。", "要曬。"),
    ("晒衣的索子", "曬衣的索子"),
    ("理著晒哩", "理著曬哩"),
    ("晒乾巴子", "曬乾巴子"),
    ("自收自晒的", "自收自曬的"),
    ("晒刷了馬匹", "曬刷了馬匹"),
    ("晒晒關文", "曬曬關文"),
    ("妻晒網圍", "妻曬網圍"),
    ("掛晒繒", "掛曬繒"),
    ("醃著，晒乾了", "醃著，曬乾了"),
    ("原來晒乾疤", "原來曬乾疤"),
    ("日晒花心", "日曬花心"),
)
_JOURNEY_CLOSING_COLOPHON = "《西遊記》至此終。"


def _normalize_audited_phrases(text, substitutions):
    for source, replacement in substitutions:
        text = text.replace(source, replacement)
    return text


class WikitextCleaningError(CorpusValidationError):
    """Source markup changed in a way the strict cleaner cannot interpret."""


class UnsupportedTemplateError(WikitextCleaningError):
    """A possibly content-bearing template has no explicit cleaning rule."""


def _normalise_template_name(name):
    return " ".join(name.replace("_", " ").split()).casefold()


def _balanced_template_end(text, start):
    if not text.startswith("{{", start):
        raise ValueError("Template start expected.")
    depth = 1
    cursor = start + 2
    while cursor < len(text):
        if text.startswith("{{", cursor):
            depth += 1
            cursor += 2
            continue
        if text.startswith("}}", cursor):
            depth -= 1
            cursor += 2
            if depth == 0:
                return cursor
            continue
        cursor += 1
    raise WikitextCleaningError(
        "Source contains an unterminated MediaWiki template.")


def _split_template_parts(inner):
    parts = []
    start = 0
    template_depth = 0
    link_depth = 0
    cursor = 0
    while cursor < len(inner):
        if inner.startswith("{{", cursor):
            template_depth += 1
            cursor += 2
            continue
        if inner.startswith("}}", cursor):
            template_depth -= 1
            if template_depth < 0:
                raise WikitextCleaningError(
                    "Template has unbalanced nested braces.")
            cursor += 2
            continue
        if inner.startswith("[[", cursor):
            link_depth += 1
            cursor += 2
            continue
        if inner.startswith("]]", cursor):
            link_depth -= 1
            if link_depth < 0:
                raise WikitextCleaningError(
                    "Template has an unbalanced nested link.")
            cursor += 2
            continue
        if (
                inner[cursor] == "|"
                and template_depth == 0
                and link_depth == 0):
            parts.append(inner[start:cursor])
            start = cursor + 1
        cursor += 1
    if template_depth or link_depth:
        raise WikitextCleaningError(
            "Template has unbalanced nested markup.")
    parts.append(inner[start:])
    return parts


def _template_replacement(inner):
    parts = _split_template_parts(inner)
    name = _normalise_template_name(parts[0])
    arguments = parts[1:]
    if name in _DISCARD_TEMPLATES:
        return ""
    if name in _FIRST_ARGUMENT_TEMPLATES:
        if not arguments or not arguments[0].strip():
            raise WikitextCleaningError(
                f"Template {parts[0].strip()!r} has no preferred reading.")
        return _expand_templates(arguments[0])
    if any(argument.strip() for argument in arguments):
        raise UnsupportedTemplateError(
            "Unsupported content-bearing MediaWiki template: "
            f"{parts[0].strip()!r}.")
    raise UnsupportedTemplateError(
        f"Unsupported MediaWiki template: {parts[0].strip()!r}.")


def _expand_templates(text):
    output = []
    cursor = 0
    while cursor < len(text):
        start = text.find("{{", cursor)
        if start < 0:
            output.append(text[cursor:])
            break
        output.append(text[cursor:start])
        end = _balanced_template_end(text, start)
        output.append(_template_replacement(text[start + 2:end - 2]))
        cursor = end
    return "".join(output)


def _replace_language_conversion(match):
    content = match.group(1).strip()
    variants = {}
    for item in content.split(";"):
        if ":" not in item:
            if len(content.split(";")) == 1:
                return item
            raise WikitextCleaningError(
                "Unsupported MediaWiki language-conversion markup.")
        language, value = item.split(":", 1)
        variants[language.strip().casefold()] = value
    try:
        return variants["zh-hant"]
    except KeyError as exc:
        raise WikitextCleaningError(
            "Language-conversion markup has no Traditional Chinese value."
        ) from exc


def _replace_tags(text):
    cursor = 0
    output = []
    for match in _TAG.finditer(text):
        output.append(text[cursor:match.start()])
        closing, raw_name = match.groups()
        name = raw_name.casefold()
        if name == "br" and not closing:
            output.append("\n")
        elif name in _ALLOWED_CONTAINER_TAGS:
            pass
        else:
            raise WikitextCleaningError(
                f"Unsupported HTML/XML tag in source: <{raw_name}>.")
        cursor = match.end()
    output.append(text[cursor:])
    return "".join(output)


def _replace_wikilink(match):
    content = match.group(1)
    parts = content.split("|")
    target = parts[0].strip()
    display = parts[-1].strip()
    if not target or not display:
        raise WikitextCleaningError("Source contains an empty wiki link.")
    if ":" in target:
        namespace = target.split(":", 1)[0].casefold()
        if namespace in _INTERLANGUAGE_NAMESPACES and len(parts) == 1:
            return ""
        if namespace != "w":
            raise WikitextCleaningError(
                f"Unsupported wiki-link namespace: {namespace!r}.")
        if len(parts) < 2:
            raise WikitextCleaningError(
                "Interwiki links must provide visible source text.")
    return display


def _replace_links(text):
    text = _CATEGORY_LINK.sub("", text)
    text = _WIKILINK.sub(_replace_wikilink, text)
    if "[[" in text or "]]" in text:
        raise WikitextCleaningError(
            "Source contains an unsupported or unbalanced wiki link.")
    external = _EXTERNAL_LINK.search(text)
    if external:
        raise WikitextCleaningError(
            f"Unsupported external link in source: {external.group(0)!r}.")
    return text


def clean_wikitext(raw_wikitext, *, remove_headings=True):
    """Convert the understood source subset to clean, readable plain text."""
    if not isinstance(raw_wikitext, str):
        raise TypeError("Wikitext must be a string.")
    text = raw_wikitext.replace("\r\n", "\n").replace("\r", "\n")
    text = _COMMENT.sub("", text)
    text = _expand_templates(text)
    text = _LANGUAGE_CONVERSION.sub(
        _replace_language_conversion,
        text,
    )
    if "-{" in text or "}-" in text:
        raise WikitextCleaningError(
            "Source contains unsupported language-conversion markup.")
    text = _replace_tags(text)
    text = _replace_links(text)
    if "<" in text or ">" in text:
        raise WikitextCleaningError(
            "Source contains unsupported angle-bracket markup.")
    if remove_headings:
        text = _ANY_HEADING.sub("", text)
    text = _MAGIC_WORD.sub("", text)
    text = _HORIZONTAL_RULE.sub("", text)
    text = _INDENT.sub("", text)
    text = text.replace("'''", "").replace("''", "")
    text = _TRAILING_SPACE.sub("", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    text = text.strip()
    if "{{" in text or "}}" in text:
        raise WikitextCleaningError(
            "Source contains unprocessed template braces.")
    return text


def _chinese_number(value):
    digits = {
        "〇": 0,
        "零": 0,
        "一": 1,
        "二": 2,
        "兩": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {
        "十": 10,
        "百": 100,
    }
    total = 0
    current = 0
    for character in value:
        if character in digits:
            current = digits[character]
            continue
        try:
            unit = units[character]
        except KeyError as exc:
            raise WikitextCleaningError(
                f"Unsupported Chinese chapter number: {value!r}.") from exc
        total += (current or 1) * unit
        current = 0
    return total + current


def clean_daodejing(page: SourcePage):
    """Split and clean exactly 81 chapters from 老子 (匯校版)."""
    if page.title != DAODEJING.content_titles[0]:
        raise CorpusValidationError(
            "The Dao De Jing cleaner only accepts 老子 (匯校版).")
    validate_page_provenance(page)
    matches = tuple(_CHAPTER_HEADING.finditer(page.raw_wikitext))
    numbers = tuple(
        _chinese_number(match.group(1))
        for match in matches
    )
    expected = tuple(range(1, DAODEJING.expected_section_count + 1))
    if numbers != expected:
        raise CorpusValidationError(
            "老子 (匯校版) must contain chapter headings 1 through 81 "
            "exactly once and in order.")

    sections = []
    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(page.raw_wikitext)
        )
        text = _normalize_audited_phrases(
            clean_wikitext(page.raw_wikitext[match.end():end]),
            _DAODEJING_ORTHOGRAPHY)
        if not text:
            raise CorpusValidationError(
                f"Dao De Jing chapter {index + 1} is empty after cleaning.")
        sections.append(TextSection(
            section_id=(
                f"{DAODEJING.key}:chapter:{index + 1:03d}"
            ),
            order=index + 1,
            title=f"第{match.group(1)}章",
            source_page_key=page.page_key,
            text=text,
        ))
    return tuple(sections)


def validate_journey_index(index_page: SourcePage):
    """Return and validate the 100 chapter links in the Wikisource index."""
    if index_page.title != JOURNEY_TO_THE_WEST.index_title:
        raise CorpusValidationError(
            "The Journey to the West index must be the 西遊記 page.")
    validate_page_provenance(index_page)
    chapter_numbers = tuple(
        int(value)
        for value in re.findall(
            r"\[\[/第([0-9]{3})回(?:\||]])",
            index_page.raw_wikitext,
        )
    )
    expected_numbers = tuple(range(1, 101))
    if chapter_numbers != expected_numbers:
        raise CorpusValidationError(
            "The 西遊記 index must link chapters 001 through 100 exactly "
            "once and in order.")
    return tuple(
        f"西遊記/第{number:03d}回"
        for number in chapter_numbers
    )


def _top_level_templates(text):
    cursor = 0
    while cursor < len(text):
        start = text.find("{{", cursor)
        if start < 0:
            return
        end = _balanced_template_end(text, start)
        yield _split_template_parts(text[start + 2:end - 2])
        cursor = end


def _header_section_title(raw_wikitext, chapter_number):
    for parts in _top_level_templates(raw_wikitext):
        if _normalise_template_name(parts[0]) != "header":
            continue
        named = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            named[key.strip().casefold()] = value
        section = named.get("section", "").strip()
        if not section:
            continue
        title = clean_wikitext(section, remove_headings=False)
        title = " ".join(title.split())
        ordinal = re.match(
            r"^第([〇零一二三四五六七八九十百兩]+)回(?:\s+|$)",
            title,
        )
        if ordinal is None or _chinese_number(
                ordinal.group(1)) != chapter_number:
            raise CorpusValidationError(
                f"西遊記 chapter {chapter_number:03d} has a mismatched "
                "header section.")
        return title
    raise CorpusValidationError(
        f"西遊記 chapter {chapter_number:03d} has no usable header section.")


def clean_journey_chapters(chapter_pages):
    """Clean the exact 100 ordered 西遊記 chapter source pages."""
    pages = tuple(chapter_pages)
    expected_titles = JOURNEY_TO_THE_WEST.content_titles
    if tuple(page.title for page in pages) != expected_titles:
        raise CorpusValidationError(
            "西遊記 chapter pages must be 001 through 100 exactly once "
            "and in order.")

    sections = []
    for chapter_number, page in enumerate(pages, start=1):
        validate_page_provenance(page)
        title = _header_section_title(
            page.raw_wikitext,
            chapter_number,
        )
        text = _normalize_audited_phrases(
            clean_wikitext(page.raw_wikitext),
            _JOURNEY_ORTHOGRAPHY)
        if chapter_number == 100:
            if not text.endswith(_JOURNEY_CLOSING_COLOPHON):
                raise CorpusValidationError(
                    "西遊記 chapter 100 no longer has the expected closing "
                    "colophon.")
            text = text[:-len(_JOURNEY_CLOSING_COLOPHON)].rstrip()
        elif _JOURNEY_CLOSING_COLOPHON in text:
            raise CorpusValidationError(
                "The 西遊記 closing colophon appeared before chapter 100.")
        if not text:
            raise CorpusValidationError(
                f"西遊記 chapter {chapter_number:03d} is empty after cleaning.")
        sections.append(TextSection(
            section_id=(
                f"{JOURNEY_TO_THE_WEST.key}:chapter:{chapter_number:03d}"
            ),
            order=chapter_number,
            title=title,
            source_page_key=page.page_key,
            text=text,
        ))
    return tuple(sections)


def clean_corpus_pages(spec: CorpusSpec, pages):
    """Validate and clean pages fetched through ``fetch_corpus_pages``."""
    source_pages = tuple(pages)
    expected_titles = spec.requested_titles
    if tuple(page.title for page in source_pages) != expected_titles:
        raise CorpusValidationError(
            f"Fetched pages do not match corpus catalogue {spec.key!r}.")
    if spec.key == DAODEJING.key:
        if len(source_pages) != 1:
            raise CorpusValidationError(
                "The Dao De Jing source requires exactly one page.")
        return clean_daodejing(source_pages[0])
    if spec.key == JOURNEY_TO_THE_WEST.key:
        index_page, *chapter_pages = source_pages
        linked_titles = validate_journey_index(index_page)
        if linked_titles != tuple(
                page.title for page in chapter_pages):
            raise CorpusValidationError(
                "西遊記 index links do not match fetched chapter pages.")
        return clean_journey_chapters(chapter_pages)
    raise KeyError(f"No cleaner is registered for corpus {spec.key!r}.")
