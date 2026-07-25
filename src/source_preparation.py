"""Prepare user-supplied PDF/text sources for the corpus vocabulary pipeline.

This module is intentionally local-only.  It extracts text, records the
original-file hash, invokes the selected historical tokenizer, and publishes
the same audited build format used by the built-in Wikisource corpora.  It
never imports OpenAI or Anki modules.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import marshal
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

from corpus_pipeline.audit import audit_build, audit_snapshot
from corpus_pipeline.contexts import assemble_sections
from corpus_pipeline.models import (
    BuildConfig,
    CORPUS_PROCESSING_VERSION,
    CorpusSnapshot,
    CorpusValidationError,
    SourcePage,
    TextSection,
    TokenSpan,
    TokenizerIdentity,
)
from corpus_pipeline.processing import build_vocabulary
from corpus_pipeline.refiners import (
    JiebaLongSpanRefiner,
    refine_build_long_spans,
)
from corpus_pipeline.resources import (
    ensure_jieba_traditional_dictionary,
)
from corpus_pipeline.storage import (
    read_snapshot,
    resolve_corpus_base,
    snapshot_id,
    write_build,
    write_snapshot,
)
from corpus_pipeline.tokenizers import (
    CkipHanTokenizer,
    HistoricalEnglishTokenizer,
    recommended_hardware_batch_size,
)
import runtime_paths


CUSTOM_SOURCE_SCHEMA_VERSION = 1
CUSTOM_SOURCE_CLEANER_VERSION = "local-document-extraction-v3"
LOCAL_DOCUMENT_REVISION_TIMESTAMP = "1970-01-01T00:00:00+00:00"
TOKENIZATION_CHECKPOINT_SCHEMA_VERSION = 1
TOKENIZATION_CHECKPOINT_IMPLEMENTATION = "section-token-spans-v1"
SUPPORTED_SOURCE_LANGUAGES = (
    "classical_chinese_han",
    "classical_chinese_wang_bi",
    "classical_chinese_warring_states",
    "classical_chinese_ming",
    "middle_english",
    "old_english",
)
_AUTOMATIC_REFINER = object()
_STORAGE_KEY_CHARACTERS = re.compile(r"[^a-z0-9]+")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_MEDIAWIKI_DIRECTIVE = re.compile(r"__[A-Z]+__")
_SKEAT_CANTERBURY_OPENING = (
    "HERE BIGINNETH THE BOOK OF THE TALES OF CAUNTERBURY.")
_SKEAT_CANTERBURY_CLOSING = (
    "=Here is ended the book of the Tales of Caunterbury, compiled by "
    "Geffrey")
_SKEAT_PAGE_LINE = re.compile(r"^\[(?:\d+\s*:|T\.)")
_SKEAT_INLINE_PAGE = re.compile(r"\[\d+(?::[^\]]*)?\]")
_SKEAT_INLINE_REFERENCE = re.compile(
    r"\s*\[(?:T\.|See\s+p\.)[^\]]*(?:\]|$)",
    re.IGNORECASE)
_SKEAT_TRAILING_LINE_NUMBER = re.compile(
    r"[ \t]{2,}(?:"
    r"\(\s*(?:\d+[a-z]?|no)\s*\)"
    r"|\d+(?:\s*[a-z])?"
    r")\s*$",
    re.IGNORECASE)
_SKEAT_SECTION_NUMBER = re.compile(r"^\s*§\s*\d+\.\s*")
_SKEAT_PROSE_LINE_NUMBER = re.compile(r"/\d+")
_SKEAT_ROMAN_ORDINAL = re.compile(
    r"(?<!\w)[ivxlcdm]+º(?!\w)",
    re.IGNORECASE)
_SKEAT_ASCII_NUMBER = re.compile(r"(?<!\w)\d+[º°]?(?!\w)")
_SKEAT_ALL_CAPS = re.compile(
    r"^[\[\]()]*[A-Z][A-Z\s.’’‘“”\-—&,;:()\[\]0-9º]+[.\])]*$")
_SKEAT_EDITORIAL_TEXT = re.compile(
    r"^(?:"
    r"\[?\s*(?:The|This|For)\b.*"
    r"(?:follows|Prologue|Appendix|see\s+p\.)"
    r"|\(?\s*(?:NERO\b|For\s+T\.|Numbered\s+in\s+continuation"
    r"|The\s+Second\s+Fit)"
    r"|T\.\s*\d"
    r"|_?Explicit\b"
    r"|_?(?:Iamque\s+domos|Ier\.\s*6|Radix\s+malorum\s+est\s+Cupiditas"
    r"|Interpretado\s+nominis|Domine,?\s+dominus\s+noster)"
    r"|GROUP\s+[A-I]\."
    r"|THE\s+.+(?:TALE|PROLOGUE|EPILOGUE)\.?$"
    r")",
    re.IGNORECASE)
_SKEAT_PUBLISHED_ERRATA_CORRECTIONS = (
    (
        "This maistow understonde and seen at eye.",
        "This maistow understonde and seen at yë.",
    ),
    (
        "Ne breed ne ale, til he cam to the celle",
        "Ne breed ne ale, til he cam to the selle",
    ),
    (
        "Thise noble wyves and thise loveres eek.",
        "Thise noble wyves and thise loveres eke.",
    ),
    (
        "Who-so that wol his large volume seek",
        "Who-so that wol his large volume seke",
    ),
    (
        "Thy selve neighebour wol thee despyse;",
        "‘Thy selve neighebour wol thee despyse;’",
    ),
    (
        "If thou be povre, thy brother hateth thee,\n"
        "And alle thy freendes fleen fro thee, alas!",
        "‘If thou be povre, thy brother hateth thee,\n"
        "And alle thy freendes fleen fro thee, alas!’",
    ),
    (
        "In al that lond no cristen durste route,",
        "In al that lond no Cristen durste route,",
    ),
    (
        "Alle cristen folk ben fled fro that contree",
        "Alle Cristen folk ben fled fro that contree",
    ),
    (
        "To Walis fled the cristianitee",
        "To Walis fled the Cristianitee",
    ),
    (
        "But yet nere cristen Britons so exyled",
        "But yet nere Cristen Britons so exyled",
    ),
    (
        "And royal spicerye;",
        "And royal spicerye",
    ),
    (
        "yevynge",
        "yevinge",
    ),
    (
        "I ne owe nat usen thy\nconseil",
        "I ne ow nat usen thy\nconseil",
    ),
    (
        "I se wel that the word of Salomon is sooth",
        "I see wel that the word of Salomon is sooth",
    ),
    (
        "Iurisdicctioun",
        "Iurisdiccioun",
    ),
    (
        "nothing certeyne;” for as lightly is oon hurt with a spere "
        "as another.",
        "nothing certeyne; for as lightly is oon hurt with a spere "
        "as another.”",
    ),
    (
        "Was wel my lorn",
        "Was wel ny lorn",
    ),
    (
        "So penible in the warre, and curteis eke,",
        "So penible in the werre, and curteis eke,",
    ),
    (
        "A povre widwe, somdel stope in age,",
        "A povre widwe, somdel stape in age,",
    ),
    (
        "The Friday for to chide, as diden ye?",
        "The Friday for to chyde, as diden ye?",
    ),
    (
        "commune opinoun",
        "commune opinioun",
    ),
    (
        "Thay shul be shryned",
        "They shul be shryned",
    ),
    (
        "But if I telle tales two or thre",
        "But-if I telle tales two or thre",
    ),
    (
        "All was this land fulfild of fayerye.",
        "Al was this land fulfild of fayerye.",
    ),
    (
        "Chese now,’ quod she",
        "Chees now,’ quod she",
    ),
    (
        "Now chese your-selven",
        "Now chees your-selven",
    ),
    (
        "But if it be to hevy or to hoot.",
        "But-if it be to hevy or to hoot.",
    ),
    (
        "Commending now the markis gouernaunce.—",
        "Commending now the markis governaunce.—",
    ),
    (
        "After thy good, and hath don many a day.",
        "After thy good, and hath don many a day.’",
    ),
    (
        "Ful lightly maystow been a cokewold.’",
        "Ful lightly maystow been a cokewold.",
    ),
    (
        "Saue o thing priketh in my conscience,",
        "Save o thing priketh in my conscience,",
    ),
    (
        "Lyk to the scorpion so deceivable,",
        "Lyk to the scorpioun so deceivable,",
    ),
    (
        "God bless us and his moder Seinte Marie!",
        "God blesse us and his moder Seinte Marie!",
    ),
    (
        "Pitous and Iust, and ever-more y-liche",
        "And piëtous and Iust, alwey y-liche.",
    ),
    (
        "Whan that this Tartre king, this Cambynskan,",
        "Whan that this Tartre king, this Cambinskan,",
    ),
    (
        "First wol I telle yow of Cambynskan,",
        "First wol I telle yow of Cambinskan,",
    ),
    (
        "Ye sle me with your sorwe, verraily;",
        "Ye slee me with your sorwe, verraily;",
    ),
    (
        "The cristen folk, which that aboute hir were,",
        "The Cristen folk, which that aboute hir were,",
    ),
    (
        "‘Sir,’ quod the preest, ‘it shall be doon, y-wis.’",
        "‘Sir,’ quod the preest, ‘it shal be doon, y-wis.’",
    ),
    (
        "Til he had torned him, coude he not blinne.",
        "Til he had terved him, coude he not blinne.",
    ),
    (
        "Him torne, I pray to god, for his falshede;",
        "Him terve, I pray to god, for his falshede;",
    ),
    (
        "wolde have hept hir fayn;",
        "wolde have kept hir fayn;",
    ),
    (
        "holier than Daniel,",
        "holier than David,",
    ),
)
_SKEAT_AUDITED_TRANSCRIPTION_CORRECTIONS = (
    (
        "And over his he’ed ther shynen two figures",
        "And over his heed ther shynen two figures",
    ),
    (
        "Er we were bom, knew al our freletee;",
        "Er we were born, knew al our freletee;",
    ),
)


def _looks_like_skeat_canterbury(text):
    """Recognize the retained body of Project Gutenberg ebook 22120."""
    stripped = text.strip()
    return (
        stripped.startswith(_SKEAT_CANTERBURY_OPENING)
        and _SKEAT_CANTERBURY_CLOSING in stripped[-1000:]
        and "HEADING. _From_ E." in stripped[:5000]
        and "THE KNIGHTES TALE." in stripped)


def _source_blocks(text):
    blocks = []
    current = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(tuple(current))
            current = []
    if current:
        blocks.append(tuple(current))
    return tuple(blocks)


def _skeat_editorial_text(text):
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith(("=", "***", "[_")):
        return True
    comparison = re.sub(r"[_=]+", "", stripped)
    comparison = " ".join(comparison.split())
    return (
        bool(_SKEAT_ALL_CAPS.fullmatch(comparison))
        or bool(_SKEAT_EDITORIAL_TEXT.match(comparison)))


def _keep_skeat_block(block):
    """Separate selected text from Skeat's indented critical apparatus."""
    minimum_indent = min(
        len(line) - len(line.lstrip(" "))
        for line in block)
    if minimum_indent > 4:
        return False
    if minimum_indent == 4:
        # The apparatus is indented four spaces throughout. The only
        # selected-text blocks with that minimum indentation are isolated
        # one-line speeches, all marked by an opening quotation mark.
        return (
            len(block) == 1
            and block[0].lstrip().startswith(("‘", "“")))
    return not _skeat_editorial_text(
        " ".join(line.strip() for line in block))


def _clean_skeat_line(line):
    text = line.strip()
    if (
            not text
            or _SKEAT_PAGE_LINE.match(text)
            or _skeat_editorial_text(text)
            or re.match(r"^\[\d+\]\s*=", text)
            or re.fullmatch(r"p\.\s*\d+\.\)?", text, re.IGNORECASE)):
        return ""

    # Marginal speaker labels and all line/page references are Skeat's,
    # while bracketed supplied readings such as "[that]" are retained as
    # ordinary text without the editorial brackets.
    text = re.sub(r"^_Auctor_\.\s*", "", text)
    text = re.sub(r"\s*=.*?=\s*$", "", text)
    text = _SKEAT_INLINE_REFERENCE.sub("", text)
    text = _SKEAT_INLINE_PAGE.sub("", text)
    text = _SKEAT_TRAILING_LINE_NUMBER.sub("", text)
    text = _SKEAT_SECTION_NUMBER.sub("", text)
    text = _SKEAT_PROSE_LINE_NUMBER.sub("", text)
    text = re.sub(r"\s*/\s*", " ", text)
    text = text.replace("_", "").replace("[", "").replace("]", "")
    text = _SKEAT_ROMAN_ORDINAL.sub("", text)
    text = _SKEAT_ASCII_NUMBER.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _apply_skeat_corrections(text):
    """Apply published preferred readings and audited transcription repairs."""
    # Suggestions marked only "perhaps" (A 467, B 4510, D 2242) and the
    # unresolved alternative at C 291 remain as printed.
    for transcription, correction in (
            _SKEAT_PUBLISHED_ERRATA_CORRECTIONS
            + _SKEAT_AUDITED_TRANSCRIPTION_CORRECTIONS):
        text = text.replace(transcription, correction)
    return text


def _clean_skeat_canterbury(text):
    """Retain only Chaucer's selected text from Skeat's plaintext edition.

    The Gutenberg plaintext encodes the main text, critical apparatus, page
    and line numbers, marginal labels, and explicitly rejected/spurious
    passages by indentation and stable presentation markers. This extraction
    mirrors those structural distinctions without contacting Gutenberg.
    """
    cleaned_blocks = []
    for block in _source_blocks(text):
        if not _keep_skeat_block(block):
            continue
        cleaned_lines = tuple(
            cleaned
            for cleaned in map(_clean_skeat_line, block)
            if cleaned)
        if cleaned_lines:
            cleaned_blocks.append("\n".join(cleaned_lines))
    cleaned = "\n\n".join(cleaned_blocks).strip()
    cleaned = _apply_skeat_corrections(cleaned)

    if (
            not cleaned.startswith(
                "Whan that Aprille with his shoures sote")
            or not cleaned.endswith(
                "the day of dome that shulle be saved: "
                "Qui cum patre, &c.")
            or re.search(r"[\d_=\[\]§*/]", cleaned)):
        raise CorpusValidationError(
            "The Skeat Canterbury Tales layout no longer matches the "
            "audited primary-text extraction rules.")
    return cleaned


@dataclass(frozen=True)
class SourcePreparationProgress:
    phase: str
    current: int
    total: int
    message: str


@dataclass(frozen=True)
class PreparedSource:
    source_key: str
    title: str
    language_key: str
    original_name: str
    original_sha256: str
    original_size: int
    snapshot_path: Path
    build_path: Path
    section_count: int
    unique_word_count: int
    token_occurrence_count: int
    created_at: str

    def to_dict(self, *, root=None):
        result = asdict(self)
        if root is not None:
            root = Path(root).resolve()
            result["snapshot_path"] = (
                Path(self.snapshot_path).resolve().relative_to(root).as_posix())
            result["build_path"] = (
                Path(self.build_path).resolve().relative_to(root).as_posix())
        else:
            result["snapshot_path"] = str(self.snapshot_path)
            result["build_path"] = str(self.build_path)
        result["schema_version"] = CUSTOM_SOURCE_SCHEMA_VERSION
        return result


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha1_text(text):
    return hashlib.sha1(
        text.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _storage_key(title, original_hash):
    slug = _STORAGE_KEY_CHARACTERS.sub(
        "-",
        unicodedata.normalize("NFKD", title)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower(),
    ).strip("-")
    slug = slug[:40] or "source"
    return f"custom-{slug}-{original_hash[:12]}"


def _clean_extracted_text(text):
    """Normalize extraction artifacts while retaining the original separately."""
    text = unicodedata.normalize(
        "NFC",
        str(text).replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(
        character
        for character in text
        if (
            character in "\n\t"
            or ord(character) >= 32))
    # These strings are provenance/navigation artifacts rather than useful
    # lexical context and would otherwise resemble uncleaned web markup.
    text = _URL.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = _MEDIAWIKI_DIRECTIVE.sub("", text)
    text = (
        text
        .replace("{{", "")
        .replace("}}", "")
        .replace("[[", "")
        .replace("]]", ""))
    if _looks_like_skeat_canterbury(text):
        return _clean_skeat_canterbury(text)
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines).strip()
    return re.sub(r"\n{4,}", "\n\n\n", text)


def _extract_text_sections(data):
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CorpusValidationError(
            "Text sources must use UTF-8 encoding.") from error
    pieces = text.split("\f")
    return tuple(
        (f"Section {index}", piece)
        for index, piece in enumerate(pieces, start=1)
        if piece.strip())


def _extract_pdf_sections(path):
    try:
        from pypdf import PdfReader
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "PDF extraction requires pypdf. Install the ordinary "
            "requirements.txt dependencies again.") from error

    try:
        reader = PdfReader(path)
    except Exception as error:
        raise CorpusValidationError(
            f"Could not read the PDF: {error}") from error
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as error:
            raise CorpusValidationError(
                "The PDF is encrypted and could not be opened.") from error
        if not unlocked:
            raise CorpusValidationError(
                "The PDF is password-protected.")

    sections = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:
            raise CorpusValidationError(
                f"Could not extract PDF page {index}: {error}") from error
        if text.strip():
            sections.append((f"Page {index}", text))
    if not sections:
        raise CorpusValidationError(
            "The PDF contains no extractable text. It may be scanned; "
            "run OCR externally, inspect it, and provide the resulting "
            "searchable PDF or UTF-8 text file.")
    return tuple(sections)


def extract_source_sections(path):
    """Return ordered `(title, raw_text)` pairs from a supported local file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_sections(path)
    if suffix in {".txt", ".md"}:
        return _extract_text_sections(path.read_bytes())
    raise ValueError(
        "Source files must be PDF, UTF-8 text, or Markdown.")


def default_document_tokenizer(language_key):
    if language_key == "middle_english":
        return HistoricalEnglishTokenizer.for_middle_english()
    if language_key == "old_english":
        return HistoricalEnglishTokenizer.for_old_english()
    options = {
        "device": "auto",
        "batch_size": recommended_hardware_batch_size(),
    }
    if language_key in {
            "classical_chinese_han",
            "classical_chinese_wang_bi",
            "classical_chinese_warring_states"}:
        return CkipHanTokenizer.for_shanggu(**options)
    if language_key == "classical_chinese_ming":
        return CkipHanTokenizer.for_jindai(**options)
    raise ValueError(
        "Local document preparation supports Classical Chinese "
        "(Warring States, early Han, Wang Bi recension, or Ming), "
        "Middle English, and Old English.")


def _make_snapshot(
        path,
        title,
        language_key,
        source_key,
        extracted):
    retrieved_at = _utc_now()
    pages = []
    raw_sections = []
    for order, (section_title, raw_text) in enumerate(
            extracted,
            start=1):
        raw_text = unicodedata.normalize(
            "NFC",
            str(raw_text).replace("\r\n", "\n").replace("\r", "\n"))
        cleaned = _clean_extracted_text(raw_text)
        if not cleaned:
            continue
        page_key = f"page-{order:04d}"
        pages.append(SourcePage(
            page_key=page_key,
            order=len(pages) + 1,
            title=section_title,
            url=f"local:{Path(path).name}#{order}",
            revision_id=order,
            # The immutable content hash is the meaningful local revision.
            # A fixed placeholder keeps the same file retryable; wall-clock
            # extraction time remains available separately in retrieved_at.
            revision_timestamp=LOCAL_DOCUMENT_REVISION_TIMESTAMP,
            revision_sha1=_sha1_text(raw_text),
            retrieved_at=retrieved_at,
            raw_wikitext=raw_text,
            raw_sha256=_sha256_text(raw_text),
        ))
        raw_sections.append(TextSection(
            section_id=f"{source_key}:section:{len(raw_sections) + 1:04d}",
            order=len(raw_sections) + 1,
            title=section_title,
            source_page_key=page_key,
            text=cleaned,
        ))
    if not raw_sections:
        raise CorpusValidationError(
            "No usable text remained after source cleaning.")
    sections, canonical_text = assemble_sections(raw_sections)
    snapshot = CorpusSnapshot(
        spec_key=source_key,
        edition=f"Local document: {title}",
        source_language_key=language_key,
        pages=tuple(pages),
        sections=sections,
        canonical_text=canonical_text,
        cleaner_version=CUSTOM_SOURCE_CLEANER_VERSION,
    )
    audit_snapshot(snapshot)
    return snapshot


def _registry_directory(corpus_root):
    return Path(corpus_root).resolve() / "_custom_sources"


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent)
    try:
        with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="\n") as output:
            json.dump(
                value,
                output,
                ensure_ascii=False,
                sort_keys=True,
                indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _implementation_fingerprint(tokenizer):
    """Describe executable tokenizer code without inspecting private state.

    The public ``TokenizerIdentity`` remains responsible for model/config
    identity.  This additional digest prevents a changed Python tokenizer
    implementation from reusing spans produced by older code.  If Python
    source is unavailable, a Python code object is accepted; opaque native
    callables deliberately disable checkpoint reuse instead of weakening the
    cache identity.
    """
    tokenizer_type = type(tokenizer)
    tokenize = getattr(tokenizer, "tokenize", None)
    descriptor = {
        "module": tokenizer_type.__module__,
        "qualname": tokenizer_type.__qualname__,
    }
    components = []
    try:
        components.append(
            b"class\0"
            + inspect.getsource(tokenizer_type).encode("utf-8"))
    except (OSError, TypeError):
        pass
    try:
        components.append(
            b"tokenize\0"
            + inspect.getsource(tokenize).encode("utf-8"))
    except (OSError, TypeError):
        code = getattr(
            getattr(tokenize, "__func__", tokenize),
            "__code__",
            None)
        if code is None:
            return None
        else:
            components.append(b"tokenize-code\0" + marshal.dumps(code))
    descriptor["digest_kind"] = "python-implementation-sha256"
    descriptor["digest"] = hashlib.sha256(
        b"\0component\0".join(components)).hexdigest()
    return descriptor


def _tokenizer_identity(tokenizer):
    identity = tokenizer.identity
    if not isinstance(identity, TokenizerIdentity):
        raise CorpusValidationError(
            "The document tokenizer returned an invalid identity.")
    return identity


def _snapshot_checkpoint_fingerprint(snapshot):
    """Hash the exact retained semantic snapshot used for tokenization."""
    value = {
        "spec_key": snapshot.spec_key,
        "edition": snapshot.edition,
        "source_language_key": snapshot.source_language_key,
        "cleaner_version": snapshot.cleaner_version,
        "include_section_titles": snapshot.include_section_titles,
        "canonical_sha256": _sha256_text(snapshot.canonical_text),
        "pages": [
            page.to_dict(include_wikitext=False)
            for page in snapshot.pages
        ],
        "sections": [
            {
                **{
                    key: value
                    for key, value in section.to_dict().items()
                    if key != "text"
                },
                "text_sha256": _sha256_text(section.text),
            }
            for section in snapshot.sections
        ],
    }
    return _sha256_text(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":")))


def _strict_token_spans(value, section_text):
    """Decode and fully validate cached/raw local token spans."""
    if not isinstance(value, (tuple, list)):
        raise CorpusValidationError(
            "A tokenization checkpoint has no valid token list.")
    result = []
    previous_end = 0
    for index, item in enumerate(value, start=1):
        if isinstance(item, TokenSpan):
            token = item
        else:
            if not isinstance(item, dict) or set(item) != {
                    "surface",
                    "start_offset",
                    "end_offset",
                    "confidence"}:
                raise CorpusValidationError(
                    "A tokenization checkpoint contains malformed spans.")
            start = item["start_offset"]
            end = item["end_offset"]
            confidence = item["confidence"]
            if (
                    isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                    or not isinstance(item["surface"], str)
                    or (
                        confidence is not None
                        and (
                            isinstance(confidence, bool)
                            or not isinstance(confidence, (int, float))
                            or not math.isfinite(confidence)
                            or not 0.0 <= confidence <= 1.0))):
                raise CorpusValidationError(
                    "A tokenization checkpoint contains invalid span types.")
            token = TokenSpan(
                surface=item["surface"],
                start_offset=start,
                end_offset=end,
                confidence=(
                    None
                    if confidence is None
                    else float(confidence)))
        if (
                isinstance(token.start_offset, bool)
                or not isinstance(token.start_offset, int)
                or isinstance(token.end_offset, bool)
                or not isinstance(token.end_offset, int)
                or not isinstance(token.surface, str)
                or (
                    token.confidence is not None
                    and (
                        isinstance(token.confidence, bool)
                        or not isinstance(token.confidence, (int, float))
                        or not math.isfinite(token.confidence)
                        or not 0.0 <= token.confidence <= 1.0))
                or token.start_offset < previous_end
                or token.end_offset <= token.start_offset
                or token.end_offset > len(section_text)
                or section_text[
                    token.start_offset:token.end_offset] != token.surface):
            raise CorpusValidationError(
                f"Token span {index} does not reproduce its section text.")
        previous_end = token.end_offset
        result.append(TokenSpan(
            surface=token.surface,
            start_offset=token.start_offset,
            end_offset=token.end_offset,
            confidence=(
                None
                if token.confidence is None
                else float(token.confidence))))
    return tuple(result)


def _checkpoint_directory(corpus_root, snapshot, namespace):
    corpus_base = resolve_corpus_base(
        corpus_root,
        snapshot.spec_key,
        create_root=True)
    relative = (
        Path("checkpoints")
        / "tokenization"
        / snapshot_id(snapshot)
        / namespace)
    current = corpus_base
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise CorpusValidationError(
                "Tokenization checkpoint storage cannot be a symlink.")
    resolved = current.resolve()
    try:
        resolved.relative_to(corpus_base)
    except ValueError as error:
        raise CorpusValidationError(
            "Tokenization checkpoint storage escapes the corpus.") from error
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


class _CheckpointingTokenizer:
    """Sequential section-tokenizer adapter with fail-closed reuse."""

    def __init__(self, delegate, snapshot, corpus_root):
        self._delegate = delegate
        self._snapshot = snapshot
        self._cursor = 0
        self._identity = _tokenizer_identity(delegate)
        self._implementation = _implementation_fingerprint(delegate)
        self._enabled = self._implementation is not None
        self._snapshot_identifier = snapshot_id(snapshot)
        self._snapshot_fingerprint = _snapshot_checkpoint_fingerprint(
            snapshot)
        self._canonical_sha256 = _sha256_text(snapshot.canonical_text)
        identity = {
            "checkpoint_implementation":
                TOKENIZATION_CHECKPOINT_IMPLEMENTATION,
            "corpus_processing_version": CORPUS_PROCESSING_VERSION,
            "snapshot_id": self._snapshot_identifier,
            "snapshot_sha256": self._snapshot_fingerprint,
            "tokenizer": self._identity.to_dict(),
            "tokenizer_implementation": self._implementation,
        }
        namespace = _sha256_text(json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":")))[:24]
        self._directory = (
            _checkpoint_directory(corpus_root, snapshot, namespace)
            if self._enabled
            else None)

    @property
    def identity(self):
        return self._identity

    def _metadata(self, section):
        return {
            "schema_version": TOKENIZATION_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_implementation":
                TOKENIZATION_CHECKPOINT_IMPLEMENTATION,
            "corpus_processing_version": CORPUS_PROCESSING_VERSION,
            "snapshot_id": self._snapshot_identifier,
            "snapshot_sha256": self._snapshot_fingerprint,
            "canonical_sha256":
                self._canonical_sha256,
            "section": {
                "section_id": section.section_id,
                "order": section.order,
                "start_offset": section.start_offset,
                "end_offset": section.end_offset,
                "text_sha256": _sha256_text(section.text),
            },
            "tokenizer": self._identity.to_dict(),
            "tokenizer_implementation": self._implementation,
        }

    @staticmethod
    def _encoded_tokens(tokens):
        return [
            {
                "surface": token.surface,
                "start_offset": token.start_offset,
                "end_offset": token.end_offset,
                "confidence": token.confidence,
            }
            for token in tokens
        ]

    def _read(self, path, metadata, text):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                    not isinstance(value, dict)
                    or set(value) != {"metadata", "tokens"}
                    or value["metadata"] != metadata):
                raise CorpusValidationError(
                    "Tokenization checkpoint identity does not match.")
            return _strict_token_spans(value["tokens"], text)
        except (
                CorpusValidationError,
                json.JSONDecodeError,
                OSError,
                TypeError,
                ValueError):
            # A checkpoint is only an optimization. Any unreadable,
            # incomplete, stale, or inconsistent artifact is recomputed from
            # the retained source text and never contributes to the build.
            return None

    def tokenize(self, text):
        if self._cursor >= len(self._snapshot.sections):
            raise CorpusValidationError(
                "The vocabulary builder requested too many sections.")
        section = self._snapshot.sections[self._cursor]
        self._cursor += 1
        if text != section.text:
            raise CorpusValidationError(
                "The vocabulary builder requested sections out of order.")

        metadata = self._metadata(section)
        path = (
            self._directory / f"section-{section.order:06d}.json"
            if self._enabled
            else None)
        if path is not None and path.is_file():
            cached = self._read(path, metadata, text)
            if cached is not None:
                return cached

        if _tokenizer_identity(self._delegate) != self._identity:
            raise CorpusValidationError(
                "Tokenizer identity changed before section tokenization. "
                "Select an explicit device/configuration and try again.")
        tokens = _strict_token_spans(
            self._delegate.tokenize(text),
            text)
        if _tokenizer_identity(self._delegate) != self._identity:
            raise CorpusValidationError(
                "Tokenizer identity changed during section tokenization. "
                "Select an explicit device/configuration and try again.")
        if path is not None:
            _write_json_atomic(path, {
                "metadata": metadata,
                "tokens": self._encoded_tokens(tokens),
            })
        return tokens


def prepare_source_file(
        path,
        *,
        title,
        language_key,
        corpus_root=None,
        tokenizer=None,
        refiner=_AUTOMATIC_REFINER,
        progress_callback=None):
    """Extract, tokenize, audit, and register one local source file."""
    title = str(title).strip()
    if not title:
        raise ValueError("A source title is required.")
    if language_key not in SUPPORTED_SOURCE_LANGUAGES:
        raise ValueError("Select a supported historical language.")
    path = Path(path).resolve()
    data = path.read_bytes()
    original_hash = _sha256_bytes(data)
    source_key = _storage_key(title, original_hash)
    corpus_root = Path(
        corpus_root
        or runtime_paths.get_corpus_output_directory()).resolve()

    def emit(phase, current, total, message):
        if progress_callback is not None:
            progress_callback(SourcePreparationProgress(
                phase,
                current,
                total,
                message))

    emit("extract", 0, 1, f"Extracting {path.name}.")
    extracted = extract_source_sections(path)
    snapshot = _make_snapshot(
        path,
        title,
        language_key,
        source_key,
        extracted)
    emit(
        "extract",
        1,
        1,
        f"Extracted {len(snapshot.sections)} sections.")

    snapshot_path = write_snapshot(snapshot, corpus_root)
    # Always tokenize the verified retained snapshot. On an explicit rerun
    # this preserves its exact provenance timestamps as well as its text,
    # rather than allowing a new wall-clock extraction time to perturb cache
    # identity or the eventual build.
    snapshot = read_snapshot(snapshot_path)
    source_directory = corpus_root / source_key / "original"
    source_directory.mkdir(parents=True, exist_ok=True)
    copied_source = source_directory / path.name
    if not copied_source.exists():
        shutil.copyfile(path, copied_source)
    elif _sha256_bytes(copied_source.read_bytes()) != original_hash:
        raise CorpusValidationError(
            "The retained original source does not match its hash.")

    selected_tokenizer = (
        tokenizer
        if tokenizer is not None
        else default_document_tokenizer(language_key))
    emit(
        "tokenize",
        0,
        len(snapshot.sections),
        "Loading the selected historical-language tokenizer.")
    # Device-auto CKIP tokenizers do not expose their resolved runtime
    # identity until their public preparation hook loads the model. Resolve
    # it before deriving checkpoint keys so a cache-only rerun retains
    # exactly the same build identity. Custom tokenizers must expose a stable
    # identity; the adapter verifies that invariant around every call.
    if isinstance(selected_tokenizer, CkipHanTokenizer):
        selected_tokenizer.prepare()
    checkpointing_tokenizer = _CheckpointingTokenizer(
        selected_tokenizer,
        snapshot,
        corpus_root)

    def token_progress(current, total, section_title):
        emit(
            "tokenize",
            current,
            total,
            f"Tokenized {section_title} ({current}/{total}).")

    build = build_vocabulary(
        snapshot,
        checkpointing_tokenizer,
        BuildConfig(chunk_size=500),
        progress_callback=token_progress)

    selected_refiner = refiner
    if selected_refiner is _AUTOMATIC_REFINER:
        selected_refiner = None
        if language_key == "classical_chinese_ming":
            dictionary = ensure_jieba_traditional_dictionary(corpus_root)
            selected_refiner = JiebaLongSpanRefiner(dictionary)
    if selected_refiner is not None:
        emit("refine", 0, 1, "Refining unresolved long word spans.")
        build = refine_build_long_spans(build, selected_refiner)
        emit("refine", 1, 1, "Refined unresolved long word spans.")

    audit_build(build)
    emit("write", 0, 1, "Saving the audited processed source.")
    build_path = write_build(build, corpus_root)
    created_at = _utc_now()
    prepared = PreparedSource(
        source_key=source_key,
        title=title,
        language_key=language_key,
        original_name=path.name,
        original_sha256=original_hash,
        original_size=len(data),
        snapshot_path=snapshot_path,
        build_path=build_path,
        section_count=len(snapshot.sections),
        unique_word_count=len(build.unique_words),
        token_occurrence_count=len(build.occurrences),
        created_at=created_at,
    )
    _write_json_atomic(
        _registry_directory(corpus_root) / f"{source_key}.json",
        prepared.to_dict(root=corpus_root))
    emit(
        "write",
        1,
        1,
        f"Saved {len(build.unique_words)} unique candidates.")
    return prepared


def list_prepared_sources(corpus_root=None):
    """List valid registered user-supplied sources newest first."""
    corpus_root = Path(
        corpus_root
        or runtime_paths.get_corpus_output_directory()).resolve()
    registry = _registry_directory(corpus_root)
    if not registry.is_dir():
        return ()
    sources = []
    for path in sorted(registry.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CorpusValidationError(
                f"Could not read prepared-source record {path.name}.") from error
        if data.get("schema_version") != CUSTOM_SOURCE_SCHEMA_VERSION:
            raise CorpusValidationError(
                f"Unsupported prepared-source record: {path.name}")
        source_key = data.get("source_key")
        if path.stem != source_key:
            raise CorpusValidationError(
                "Prepared-source filename and identity disagree.")
        try:
            snapshot_path = (
                corpus_root / data["snapshot_path"]).resolve()
            build_path = (corpus_root / data["build_path"]).resolve()
            snapshot_path.relative_to(corpus_root)
            build_path.relative_to(corpus_root)
        except (KeyError, ValueError) as error:
            raise CorpusValidationError(
                "Prepared-source path escapes corpus storage.") from error
        if not snapshot_path.is_dir() or not build_path.is_dir():
            raise CorpusValidationError(
                f"Prepared source is incomplete: {source_key}")
        sources.append(PreparedSource(
            source_key=source_key,
            title=data["title"],
            language_key=data["language_key"],
            original_name=data["original_name"],
            original_sha256=data["original_sha256"],
            original_size=data["original_size"],
            snapshot_path=snapshot_path,
            build_path=build_path,
            section_count=data["section_count"],
            unique_word_count=data["unique_word_count"],
            token_occurrence_count=data["token_occurrence_count"],
            created_at=data["created_at"],
        ))
    return tuple(sorted(
        sources,
        key=lambda source: source.created_at,
        reverse=True))
