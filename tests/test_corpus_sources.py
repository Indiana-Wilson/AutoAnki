import hashlib
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from corpus_pipeline import catalogue, cleaners, mediawiki
from corpus_pipeline.models import CorpusValidationError, SourcePage


def source_page(title, raw_wikitext, *, page_key="fixture", order=1):
    return SourcePage(
        page_key=page_key,
        order=order,
        title=title,
        url="https://zh.wikisource.org/wiki/fixture",
        revision_id=123,
        revision_timestamp="2026-01-02T03:04:05Z",
        revision_sha1="base36-sha1",
        retrieved_at="2026-01-03T04:05:06Z",
        raw_wikitext=raw_wikitext,
        raw_sha256=hashlib.sha256(
            raw_wikitext.encode("utf-8")).hexdigest(),
    )


class FakeResponse:
    def __init__(self, payload, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def json(self):
        return self.payload


class FakeMediaWikiClient:
    def __init__(self):
        self.calls = []

    def get(self, url, *, params, headers, timeout):
        self.calls.append({
            "url": url,
            "params": params,
            "headers": headers,
            "timeout": timeout,
        })
        titles = params["titles"].split("|")
        pages = []
        for number, title in enumerate(reversed(titles), start=1):
            pages.append({
                "title": title,
                "fullurl": (
                    "https://zh.wikisource.org/wiki/"
                    + title.replace(" ", "_")
                ),
                "revisions": [{
                    "revid": 1000 + number,
                    "timestamp": "2026-01-02T03:04:05Z",
                    "sha1": f"sha1-{number}",
                    "slots": {
                        "main": {
                            "content": f"raw {title}",
                        },
                    },
                }],
            })
        return FakeResponse({
            "query": {
                "pages": pages,
            },
        })


class CatalogueTests(unittest.TestCase):
    def test_exact_corpus_sources_are_pinned_in_catalogue(self):
        daodejing = catalogue.get_corpus_spec("daodejing_huijiao")
        journey = catalogue.get_corpus_spec("journey_to_the_west")

        self.assertEqual(
            daodejing.content_titles,
            ("老子 (匯校版)",))
        self.assertIsNone(daodejing.index_title)
        self.assertEqual(daodejing.expected_section_count, 81)
        self.assertIsNone(daodejing.long_span_refiner)
        self.assertFalse(daodejing.default_include_section_titles)
        self.assertEqual(journey.index_title, "西遊記")
        self.assertEqual(
            journey.long_span_refiner,
            "jieba_traditional_max4")
        self.assertTrue(journey.default_include_section_titles)
        self.assertEqual(len(journey.content_titles), 100)
        self.assertEqual(
            journey.content_titles[0],
            "西遊記/第001回")
        self.assertEqual(
            journey.content_titles[-1],
            "西遊記/第100回")
        self.assertEqual(len(journey.requested_titles), 101)

    def test_page_keys_and_urls_are_stable(self):
        journey = catalogue.JOURNEY_TO_THE_WEST

        self.assertEqual(
            journey.page_key("西遊記"),
            "index")
        self.assertEqual(
            journey.page_key("西遊記/第007回"),
            "chapter-007")
        self.assertIn(
            "%E8%A5%BF%E9%81%8A%E8%A8%98",
            journey.page_url("西遊記"))


class MediaWikiTests(unittest.TestCase):
    def test_fetches_batched_revisions_in_requested_order(self):
        client = FakeMediaWikiClient()
        titles = ("甲", "乙", "丙")
        progress = []

        pages = mediawiki.fetch_pages(
            titles,
            page_keys=("a", "b", "c"),
            client=client,
            user_agent="AutoAnki source tests/1",
            maxlag=7,
            batch_size=2,
            retrieved_at="2026-07-24T00:00:00Z",
            progress_callback=lambda current, total: progress.append(
                (current, total)),
        )

        self.assertEqual(
            tuple(page.title for page in pages),
            titles)
        self.assertEqual(
            tuple(page.page_key for page in pages),
            ("a", "b", "c"))
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(
            client.calls[0]["params"]["rvprop"],
            "ids|timestamp|sha1|content")
        self.assertEqual(client.calls[0]["params"]["maxlag"], "7")
        self.assertEqual(
            client.calls[0]["headers"]["User-Agent"],
            "AutoAnki source tests/1")
        self.assertEqual(
            pages[0].raw_sha256,
            hashlib.sha256("raw 甲".encode("utf-8")).hexdigest())
        self.assertEqual(
            pages[0].retrieved_at,
            "2026-07-24T00:00:00Z")
        self.assertEqual(progress, [(2, 3), (3, 3)])

    def test_missing_or_redirected_page_fails_closed(self):
        class MissingClient:
            def get(self, *args, **kwargs):
                return FakeResponse({
                    "query": {
                        "pages": [{
                            "title": "Different page",
                            "missing": True,
                        }],
                    },
                })

        with self.assertRaises(mediawiki.MediaWikiFetchError):
            mediawiki.fetch_pages(
                ("Required page",),
                client=MissingClient(),
            )

    def test_fetch_option_validation_happens_without_network(self):
        with self.assertRaises(ValueError):
            mediawiki.fetch_pages(("甲",), user_agent="")
        with self.assertRaises(ValueError):
            mediawiki.fetch_pages(("甲",), batch_size=51)
        with self.assertRaises(ValueError):
            mediawiki.fetch_pages(("甲",), maxlag=-1)
        with self.assertRaises(ValueError):
            mediawiki.fetch_pages(("甲",), maxlag=True)
        with self.assertRaises(ValueError):
            mediawiki.fetch_pages(("甲",), batch_size=True)


class CleanerTests(unittest.TestCase):
    def test_cleaning_retains_preferred_readings_poetry_and_punctuation(self):
        raw = """{{header|title=書|section=第一回}}
:詩曰：
<onlyinclude><poem>:道，{{參|可{{另|道|說}}|variant}}也。
:-{谷}-神；-{zh-hans:徵;zh-hant:徵}-。</poem></onlyinclude>
[[w:臨江仙|臨江仙]]
{{footer}}
[[Category:書]]
[[en:Tao Te Ching]]
"""

        cleaned = cleaners.clean_wikitext(raw)

        self.assertEqual(
            cleaned,
            "詩曰：\n道，可道也。\n谷神；徵。\n臨江仙")

    def test_unknown_content_markup_fails_closed(self):
        with self.assertRaises(cleaners.UnsupportedTemplateError):
            cleaners.clean_wikitext("正文{{unknown|額外文字}}")
        with self.assertRaises(cleaners.WikitextCleaningError):
            cleaners.clean_wikitext("正文<ruby>字</ruby>")
        with self.assertRaises(cleaners.WikitextCleaningError):
            cleaners.clean_wikitext("[[File:scan.jpg|正文]]")

    def test_daodejing_splits_exactly_81_headings(self):
        chapters = []
        numerals = (
            "一 二 三 四 五 六 七 八 九 十 十一 十二 十三 十四 十五 "
            "十六 十七 十八 十九 二十 二十一 二十二 二十三 二十四 "
            "二十五 二十六 二十七 二十八 二十九 三十 三十一 三十二 "
            "三十三 三十四 三十五 三十六 三十七 三十八 三十九 四十 "
            "四十一 四十二 四十三 四十四 四十五 四十六 四十七 四十八 "
            "四十九 五十 五十一 五十二 五十三 五十四 五十五 五十六 "
            "五十七 五十八 五十九 六十 六十一 六十二 六十三 六十四 "
            "六十五 六十六 六十七 六十八 六十九 七十 七十一 七十二 "
            "七十三 七十四 七十五 七十六 七十七 七十八 七十九 八十 "
            "八十一"
        ).split()
        for number, numeral in enumerate(numerals, start=1):
            body = (
                "{{參|道，可道。|別本}}"
                if number == 1
                else f"本章正文{number}。"
            )
            chapters.append(f"=== {numeral}章 ===\n{body}\n")
        raw = (
            "{{header|title=老子}}\n== 道經 ==\n"
            + "".join(chapters)
            + "{{footer}}\n[[Category:道經]]"
        )

        sections = cleaners.clean_daodejing(source_page(
            "老子 (匯校版)",
            raw,
            page_key="source",
        ))

        self.assertEqual(len(sections), 81)
        self.assertEqual(sections[0].text, "道，可道。")
        self.assertEqual(sections[0].title, "第一章")
        self.assertEqual(
            sections[-1].section_id,
            "daodejing_huijiao:chapter:081")
        self.assertNotIn("===", sections[1].text)

    def test_daodejing_rejects_a_missing_chapter(self):
        raw = "".join(
            f"=== {number}章 ===\n正文。\n"
            for number in ("一", "二", "四")
        )
        with self.assertRaises(CorpusValidationError):
            cleaners.clean_daodejing(source_page(
                "老子 (匯校版)",
                raw,
            ))

    def test_source_specific_simplified_slips_are_corrected_only_in_cleaners(
            self):
        numerals = (
            "一 二 三 四 五 六 七 八 九 十 十一 十二 十三 十四 十五 "
            "十六 十七 十八 十九 二十 二十一 二十二 二十三 二十四 "
            "二十五 二十六 二十七 二十八 二十九 三十 三十一 三十二 "
            "三十三 三十四 三十五 三十六 三十七 三十八 三十九 四十 "
            "四十一 四十二 四十三 四十四 四十五 四十六 四十七 四十八 "
            "四十九 五十 五十一 五十二 五十三 五十四 五十五 五十六 "
            "五十七 五十八 五十九 六十 六十一 六十二 六十三 六十四 "
            "六十五 六十六 六十七 六十八 六十九 七十 七十一 七十二 "
            "七十三 七十四 七十五 七十六 七十七 七十八 七十九 八十 "
            "八十一"
        ).split()
        raw = "".join(
            f"=== {numeral}章 ===\n"
            + (
                "如春登台，其事好还，九層之台；膻中登台。\n"
                if index == 1
                else "正文。\n"
            )
            for index, numeral in enumerate(numerals, start=1)
        )
        dao = cleaners.clean_daodejing(source_page(
            "老子 (匯校版)",
            raw,
        ))
        self.assertEqual(
            dao[0].text,
            "如春登臺，其事好還，九層之臺；膻中登台。")

        journey_raw = (
            "{{header|title=西遊記|section=第一回<br>章題}}\n"
            "攛將上来。獅驼王。日晒花心。"
            "跨了刬馬，又跨著刬馬。聞腥膻，撐上腭子。"
            "膻中登台。"
        )
        journey_pages = list(self._journey_pages())
        journey_pages[0] = source_page(
            "西遊記/第001回",
            journey_raw,
            page_key="chapter-001",
            order=2,
        )
        journey = cleaners.clean_journey_chapters(journey_pages)
        self.assertEqual(
            journey[0].text,
            "攛將上來。獅駝王。日曬花心。"
            "跨了剗馬，又跨著剗馬。聞腥羶，撐上齶子。"
            "膻中登台。")

    def _journey_index(self, *, omit=None):
        lines = [
            f"*[[/第{number:03d}回|第{number}回]]"
            for number in range(1, 101)
            if number != omit
        ]
        return source_page(
            "西遊記",
            "{{header|title=西遊記}}\n" + "\n".join(lines),
            page_key="index",
        )

    def _journey_pages(self):
        pages = []
        for number, title in enumerate(
                catalogue.JOURNEY_TO_THE_WEST.content_titles,
                start=1):
            numeral = self._integer_to_chinese(number)
            raw = (
                "{{header\n"
                "| title = 西遊記\n"
                f"| section = 第{numeral}回<br>章題{number}\n"
                "}}\n"
                f"正文第{number}頁。"
            )
            if number == 90:
                raw += "\n<onlyinclude><poem>:詩句。</poem></onlyinclude>"
            if number == 100:
                raw += "\n《西遊記》至此終。"
            pages.append(source_page(
                title,
                raw,
                page_key=f"chapter-{number:03d}",
                order=number + 1,
            ))
        return tuple(pages)

    @staticmethod
    def _integer_to_chinese(value):
        digits = "零一二三四五六七八九"
        if value == 100:
            return "一百"
        tens, ones = divmod(value, 10)
        if tens == 0:
            return digits[ones]
        prefix = "" if tens == 1 else digits[tens]
        return prefix + "十" + (digits[ones] if ones else "")

    def test_journey_index_and_all_chapters_are_cross_checked(self):
        index_page = self._journey_index()
        pages = self._journey_pages()

        linked = cleaners.validate_journey_index(index_page)
        sections = cleaners.clean_corpus_pages(
            catalogue.JOURNEY_TO_THE_WEST,
            (index_page,) + pages,
        )

        self.assertEqual(linked, tuple(page.title for page in pages))
        self.assertEqual(len(sections), 100)
        self.assertEqual(
            sections[0].title,
            "第一回 章題1")
        self.assertEqual(sections[0].text, "正文第1頁。")
        self.assertEqual(
            sections[89].text,
            "正文第90頁。\n詩句。")
        self.assertNotIn("第九十回", sections[89].text)
        self.assertEqual(sections[-1].text, "正文第100頁。")
        self.assertNotIn("西遊記", sections[-1].text)

    def test_journey_rejects_incomplete_index(self):
        with self.assertRaises(CorpusValidationError):
            cleaners.validate_journey_index(
                self._journey_index(omit=57))

    def test_hash_mismatch_is_rejected_before_cleaning(self):
        page = source_page("老子 (匯校版)", "=== 一章 ===\n道。")
        tampered = SourcePage(
            **{
                **page.__dict__,
                "raw_sha256": "0" * 64,
            },
        )
        with self.assertRaises(CorpusValidationError):
            cleaners.clean_daodejing(tampered)


if __name__ == "__main__":
    unittest.main()
