"""Fixed source catalogue for the first AutoAnki corpus imports.

The catalogue deliberately names exact Wikisource pages.  A build can record
the revision fetched from those pages without silently drifting to a
different edition or a modernised text.
"""

from dataclasses import dataclass
from urllib.parse import quote


WIKISOURCE_API_URL = "https://zh.wikisource.org/w/api.php"
WIKISOURCE_PAGE_URL = "https://zh.wikisource.org/wiki/"


@dataclass(frozen=True)
class CorpusSpec:
    """Immutable description of an ordered Wikisource corpus."""

    key: str
    title: str
    edition: str
    author: str
    period: str
    source_language_key: str
    index_title: str | None
    content_titles: tuple[str, ...]
    expected_section_count: int
    long_span_refiner: str | None = None
    default_include_section_titles: bool = False
    api_url: str = WIKISOURCE_API_URL

    def __post_init__(self):
        if not self.key:
            raise ValueError("A corpus key is required.")
        if not self.content_titles:
            raise ValueError("A corpus must contain at least one source page.")
        if self.expected_section_count < 1:
            raise ValueError(
                "A corpus must contain at least one expected section.")
        if self.long_span_refiner not in {
                None,
                "jieba_traditional_max4"}:
            raise ValueError("Unknown corpus long-span refinement policy.")
        if not isinstance(self.default_include_section_titles, bool):
            raise ValueError(
                "Default section-title inclusion must be true or false.")
        if len(set(self.requested_titles)) != len(self.requested_titles):
            raise ValueError("Corpus source page titles must be unique.")

    @property
    def requested_titles(self):
        """Return index (when present), followed by ordered content pages."""
        if self.index_title is None:
            return self.content_titles
        return (self.index_title,) + self.content_titles

    def page_key(self, title):
        """Return the stable provenance key for an exact source title."""
        if title == self.index_title:
            return "index"
        try:
            content_index = self.content_titles.index(title)
        except ValueError as exc:
            raise KeyError(
                f"{title!r} is not part of corpus {self.key!r}.") from exc
        if self.key == "journey_to_the_west":
            return f"chapter-{content_index + 1:03d}"
        return "source"

    def page_url(self, title):
        """Return the canonical human-readable URL for a source title."""
        if title not in self.requested_titles:
            raise KeyError(
                f"{title!r} is not part of corpus {self.key!r}.")
        return WIKISOURCE_PAGE_URL + quote(title.replace(" ", "_"), safe="/()")


def _journey_chapter_titles():
    return tuple(
        f"西遊記/第{chapter_number:03d}回"
        for chapter_number in range(1, 101)
    )


DAODEJING_WANG_BI = CorpusSpec(
    key="daodejing_wang_bi",
    title="道德經",
    edition="王弼本",
    author="老子",
    period="Received text preserved with Wang Bi's third-century commentary",
    source_language_key="classical_chinese_wang_bi",
    index_title=None,
    content_titles=("道德經 (王弼本)",),
    expected_section_count=81,
)

DAODEJING_MAWANGDUI = CorpusSpec(
    key="daodejing_mawangdui",
    title="道德經",
    edition="馬王堆帛書校勘版",
    author="老子",
    period="Early Western Han silk-manuscript recension",
    source_language_key="classical_chinese_han",
    index_title=None,
    content_titles=("老子 (帛書校勘版)",),
    expected_section_count=81,
)

# Keep the original public constant as the conventional default edition.
DAODEJING = DAODEJING_WANG_BI


JOURNEY_TO_THE_WEST = CorpusSpec(
    key="journey_to_the_west",
    title="西遊記",
    edition="Chinese Wikisource 西遊記",
    author="吳承恩",
    period="Ming",
    source_language_key="classical_chinese_ming",
    index_title="西遊記",
    content_titles=_journey_chapter_titles(),
    expected_section_count=100,
    long_span_refiner="jieba_traditional_max4",
    default_include_section_titles=True,
)


_CORPORA = (
    DAODEJING_WANG_BI,
    DAODEJING_MAWANGDUI,
    JOURNEY_TO_THE_WEST,
)
_CORPORA_BY_KEY = {
    corpus.key: corpus
    for corpus in _CORPORA
}


def list_corpus_specs():
    """Return the supported corpora in their UI display order."""
    return _CORPORA


def get_corpus_spec(key):
    """Return one exact corpus description by stable key."""
    try:
        return _CORPORA_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown corpus: {key!r}.") from exc
