"""Package and import completed source-generation jobs."""

import hashlib
import html
import json
from pathlib import Path

import genanki

import anki_integration
import pipeline_store
import process_text
import templates


SOURCE_DIRECTION_LABELS = {
    "context": "Sentence to Meaning",
    "word_to_meaning": "Word to Meaning",
    "meaning_to_word": "Meaning to Word",
}
SOURCE_DIRECTION_PRIORITY = {
    "context": 0,
    "word_to_meaning": 1,
    "meaning_to_word": 2,
}


def source_deck_name(source_title):
    title = " ".join(str(source_title).split())
    if not title:
        raise ValueError("A source title is required.")
    return f"Vocabulary from {title}"


def source_deck_id(source_key):
    return _stable_deck_id(f"autoanki-source-deck:{source_key}")


def source_direction_deck_id(source_key, direction_key):
    if direction_key not in SOURCE_DIRECTION_LABELS:
        raise ValueError(f"Unknown source card direction: {direction_key}")
    return _stable_deck_id(
        f"autoanki-source-deck:{source_key}:{direction_key}")


def _stable_deck_id(identity):
    text = str(identity).strip()
    if not text:
        raise ValueError("A source identity is required.")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") & 0x7FFFFFFF) or 1


def _response_mapping(combined_response):
    if isinstance(combined_response, bytes):
        combined_response = combined_response.decode("utf-8")
    if isinstance(combined_response, str):
        value = json.loads(combined_response)
    elif isinstance(combined_response, dict):
        value = dict(combined_response)
    else:
        raise TypeError("A combined source response must be JSON or a map.")
    cards = value.get("cards")
    if not isinstance(cards, list):
        raise ValueError("A combined source response requires a cards array.")
    return value


def _source_optional_fields(pipeline):
    lexical_fields = {
        pipeline_store.response_field_name(field_setting)
        for field_setting
        in pipeline_store.get_source_lexical_field_settings(pipeline)
    }
    return tuple(sorted({
        "Sentences",
        process_text.SENTENCE_TRANSLATIONS_FIELD_NAME,
        *(
            pipeline_store.response_field_name(field_setting)
            for field_setting
            in pipeline_store.get_requested_field_settings(pipeline)
            if (
                pipeline_store.response_field_name(field_setting)
                not in lexical_fields)
        ),
    }))


def _legacy_source_contexts(cards, term_field):
    """Recover enough sidecar data to package pre-refactor saved jobs."""
    contexts = []
    seen = set()
    for rank, card in enumerate(cards, start=1):
        sentences = card.get("Sentences")
        if not isinstance(sentences, str) or not sentences:
            continue
        if sentences.startswith(templates.SOURCE_CONTEXT_BLOCK_PREFIX):
            encoded = sentences[len(templates.SOURCE_CONTEXT_BLOCK_PREFIX):]
            marker = templates.SOURCE_CONTEXT_ESCAPE_MARKER
            sentence = (
                encoded
                .replace(marker + "1", "|")
                .replace(marker + "0", marker))
        elif "|" not in sentences:
            sentence = sentences
        else:
            continue
        sentence = html.unescape(
            process_text._STRONG_TAG_PATTERN.sub("", sentence))
        identity = hashlib.sha256(
            sentence.encode("utf-8")).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        contexts.append({
            "context_id": f"legacy-{identity[:20]}",
            "sentence_id": f"legacy-{identity[:20]}",
            "original_sentence": sentence,
            "english_translation": card.get(
                process_text.SENTENCE_TRANSLATIONS_FIELD_NAME,
                ""),
            "nuance": "",
            "word_ranks": [rank],
            "terms": [card.get(term_field, "")],
        })
    return contexts


def _normalise_source_contexts(value, cards, term_field):
    raw_contexts = value.get("source_contexts")
    if not isinstance(raw_contexts, list):
        return _legacy_source_contexts(cards, term_field)
    contexts = []
    seen = set()
    for raw in raw_contexts:
        if not isinstance(raw, dict):
            raise ValueError("A source sentence record must be an object.")
        sentence_id = raw.get("sentence_id", raw.get("context_id"))
        original = raw.get("original_sentence")
        translation = raw.get("english_translation")
        nuance = raw.get("nuance", "")
        if (
                not isinstance(sentence_id, str)
                or not sentence_id
                or not isinstance(original, str)
                or not original.strip()
                or not isinstance(translation, str)
                or not translation.strip()
                or not isinstance(nuance, str)):
            raise ValueError(
                "Each source sentence requires an identity, original text, "
                "English translation, and text nuance.")
        if sentence_id in seen:
            raise ValueError(
                f"Source sentence {sentence_id!r} appears more than once.")
        seen.add(sentence_id)
        ranks = [
            rank
            for rank in raw.get("word_ranks", ())
            if (
                not isinstance(rank, bool)
                and isinstance(rank, int)
                and rank > 0)
        ]
        terms = [
            term
            for term in raw.get("terms", ())
            if isinstance(term, str) and term
        ]
        ranked_terms = [
            {
                "rank": item["rank"],
                "term": item["term"],
            }
            for item in raw.get("ranked_terms", ())
            if (
                isinstance(item, dict)
                and not isinstance(item.get("rank"), bool)
                and isinstance(item.get("rank"), int)
                and item.get("rank") > 0
                and isinstance(item.get("term"), str)
                and item.get("term"))
        ]
        if not ranks:
            raise ValueError(
                "Every source sentence must be associated with at least one "
                "selected source-word rank.")
        if ranked_terms:
            if {item["rank"] for item in ranked_terms} != set(ranks):
                raise ValueError(
                    "A source sentence's ranked terms do not match its "
                    "associated word ranks.")
        elif len(terms) != len(ranks):
            raise ValueError(
                "A source sentence's terms do not match its associated word "
                "ranks.")
        contexts.append({
            **raw,
            "sentence_id": sentence_id,
            "word_ranks": sorted(set(ranks)),
            "terms": terms,
            "ranked_terms": ranked_terms,
        })
    return contexts


def _validate_expected_source_contexts(value, contexts):
    expected = value.get("expected_source_sentence_ids")
    if expected is None:
        return
    if (
            not isinstance(expected, list)
            or any(
                not isinstance(sentence_id, str)
                or not sentence_id
                for sentence_id in expected)
            or len(expected) != len(set(expected))):
        raise ValueError(
            "Expected source sentence identities are malformed.")
    actual = {
        context["sentence_id"]
        for context in contexts
    }
    if actual != set(expected):
        raise ValueError(
            "Validated source sentence data is incomplete or contains an "
            "unexpected sentence.")


def _lexical_note(
        note_data,
        pipeline,
        card,
        *,
        guid_seed,
        due):
    language = pipeline_store.get_language(pipeline.language_key)
    settings = pipeline_store.get_language_settings(
        pipeline,
        pipeline.language_key)
    effective_fields = pipeline_store.get_effective_fields(settings, card)
    response_fields_by_key = {
        field_setting.field_key: (
            pipeline_store.response_field_name(field_setting))
        for field_setting in effective_fields
    }
    fields = [
        note_data[language.term_field],
        "",
        *(
            note_data.get(response_fields_by_key.get(field_key, ""), "")
            for field_key, _field_name in templates.CONTENT_FIELDS
        ),
        "",
    ]
    card_type = templates.get_direction_card_type(
        language.model_language_key,
        card.direction_key)
    return genanki.Note(
        model=card_type.model,
        fields=fields,
        due=due,
        guid=genanki.guid_for(
            "autoanki",
            guid_seed,
            card_type.key,
            note_data[language.term_field],
            *(
                note_data.get(
                    pipeline_store.response_field_name(field_setting),
                    "")
                for field_setting in effective_fields
            )))


def _generated_note(
        note_data,
        pipeline,
        card,
        *,
        guid_seed,
        due):
    language = pipeline_store.get_language(pipeline.language_key)
    settings = pipeline_store.get_language_settings(
        pipeline,
        pipeline.language_key)
    effective_fields = pipeline_store.get_effective_fields(settings, card)
    response_fields_by_key = {
        field_setting.field_key: (
            pipeline_store.response_field_name(field_setting))
        for field_setting in effective_fields
    }
    sentences = note_data.get("Sentences", "")
    translations = note_data.get(
        process_text.SENTENCE_TRANSLATIONS_FIELD_NAME,
        "")
    if sentences:
        sentences = process_text.emphasize_term_in_sentences(
            sentences,
            note_data[language.term_field],
            pipeline.language_key)
    if translations:
        translations = process_text.sanitize_sentence_translations(
            translations)
    card_type = templates.get_direction_card_type(
        language.model_language_key,
        card.direction_key)
    return genanki.Note(
        model=card_type.model,
        fields=[
            note_data[language.term_field],
            sentences,
            *(
                note_data.get(
                    response_fields_by_key.get(field_key, ""),
                    "")
                for field_key, _field_name in templates.CONTENT_FIELDS
            ),
            translations,
        ],
        due=due,
        guid=genanki.guid_for(
            "autoanki",
            guid_seed,
            card_type.key,
            note_data[language.term_field],
            *(
                note_data.get(
                    pipeline_store.response_field_name(field_setting),
                    "")
                for field_setting in effective_fields
            )))


def _sentence_note(context, *, guid_seed, due):
    def safe_text(value):
        return html.escape(html.unescape(value), quote=False)

    return genanki.Note(
        model=templates.SOURCE_SENTENCE_MODEL,
        fields=[
            safe_text(context["original_sentence"]),
            safe_text(context["english_translation"]),
            safe_text(context.get("nuance", "")),
        ],
        due=due,
        guid=genanki.guid_for(
            "autoanki",
            guid_seed,
            "source_sentence",
            context["sentence_id"]))


def create_source_package(
        combined_response,
        *,
        source_title,
        source_key,
        pipeline,
        output_path,
        guid_seed=None,
        allow_accepted_content_problems=False,
        use_source_for_example_sentences=False,
        require_sentence_translations=True,
        separate_source_decks=False,
        shared_source_sentence_cards=None):
    """Validate output and package selected source card directions."""
    value = _response_mapping(combined_response)
    if shared_source_sentence_cards is None:
        shared_source_sentence_cards = bool(
            use_source_for_example_sentences)
    if not isinstance(shared_source_sentence_cards, bool):
        raise TypeError(
            "Shared source-sentence card mode must be true or false.")
    if (
            use_source_for_example_sentences
            and not shared_source_sentence_cards):
        if separate_source_decks:
            raise ValueError(
                "Legacy source packaging cannot split card directions.")
        output_path = Path(output_path)
        deck = templates.create_deck(
            source_deck_id(source_key),
            source_deck_name(source_title))
        notes_created = process_text.process_json_text(
            json.dumps(
                {"cards": value["cards"]},
                ensure_ascii=False,
                separators=(",", ":")),
            deck=deck,
            output_path=output_path,
            pipeline=pipeline,
            due_start=1,
            allow_accepted_content_problems=(
                allow_accepted_content_problems),
            enforce_sentence_count=False,
            require_sentence_translations=(
                require_sentence_translations),
            guid_seed=(
                guid_seed
                or f"source:{source_key}:{pipeline.pipeline_id}"))
        return output_path, notes_created
    response_text = json.dumps(
        {"cards": value["cards"]},
        ensure_ascii=False,
        separators=(",", ":"))
    optional_fields = (
        _source_optional_fields(pipeline)
        if use_source_for_example_sentences
        else ())
    cards = process_text.validate_generated_cards(
        response_text,
        pipeline,
        allow_accepted_content_problems=(
            allow_accepted_content_problems),
        enforce_sentence_count=(
            not use_source_for_example_sentences),
        require_sentence_translations=require_sentence_translations,
        optional_fields=optional_fields)

    enabled = tuple(sorted(
        pipeline_store.get_enabled_cards(pipeline),
        key=lambda card: SOURCE_DIRECTION_PRIORITY[card.direction_key]))
    base_name = source_deck_name(source_title)
    if separate_source_decks:
        decks = {
            card.direction_key: templates.create_deck(
                source_direction_deck_id(
                    source_key,
                    card.direction_key),
                f"{base_name}::{SOURCE_DIRECTION_LABELS[card.direction_key]}")
            for card in enabled
        }
    else:
        combined_deck = templates.create_deck(
            source_deck_id(source_key),
            base_name)
        decks = {
            card.direction_key: combined_deck
            for card in enabled
        }

    guid_seed = (
        guid_seed
        or f"source:{source_key}:{pipeline.pipeline_id}")
    events = []
    if use_source_for_example_sentences:
        contexts = _normalise_source_contexts(
            value,
            cards,
            pipeline_store.get_language(
                pipeline.language_key).term_field)
        _validate_expected_source_contexts(value, contexts)
        context_enabled = any(
            card.direction_key == "context"
            for card in enabled)
        if context_enabled and not contexts:
            raise ValueError(
                "Sentence → Meaning packaging requires at least one "
                "validated source sentence.")
        rank_by_term = {}
        for context in contexts:
            ranked_terms = context.get("ranked_terms", ())
            if not ranked_terms:
                ranked_terms = (
                    {
                        "rank": rank,
                        "term": term,
                    }
                    for rank, term in zip(
                        context.get("word_ranks", ()),
                        context.get("terms", ())))
            for item in ranked_terms:
                rank = item["rank"]
                term = item["term"]
                rank_by_term.setdefault(term, rank)
        lexical_directions = [
            card
            for card in enabled
            if card.direction_key != "context"
        ]
        context_direction = next((
            card
            for card in enabled
            if card.direction_key == "context"
        ), None)
        seen_terms = set()
        for sense_index, note_data in enumerate(cards):
            term = note_data[
                pipeline_store.get_language(
                    pipeline.language_key).term_field]
            rank = rank_by_term.get(term)
            if rank is None:
                raise ValueError(
                    f"Source lexical card term {term!r} is not associated "
                    "with any validated source sentence.")
            is_contextual_sense = term not in seen_terms
            seen_terms.add(term)
            has_generated_examples = bool(
                note_data.get("Sentences"))
            note_kind = (
                "generated"
                if (
                    not is_contextual_sense
                    and has_generated_examples)
                else "lexical")
            for direction_index, card in enumerate(lexical_directions):
                events.append((
                    (
                        rank,
                        0,
                        sense_index,
                        direction_index,
                    ),
                    card.direction_key,
                    note_kind,
                    note_data,
                    card,
                ))
            if (
                    not is_contextual_sense
                    and has_generated_examples
                    and context_direction is not None):
                events.append((
                    (
                        rank,
                        0,
                        sense_index,
                        len(lexical_directions),
                    ),
                    "context",
                    "generated",
                    note_data,
                    context_direction,
                ))
        if any(
                card.direction_key == "context"
                for card in enabled):
            for source_index, context in enumerate(contexts):
                ranks = context.get("word_ranks", ())
                after_rank = max(ranks, default=source_index + 1)
                events.append((
                    (after_rank, 1, source_index, 0),
                    "context",
                    "sentence",
                    context,
                    None,
                ))
    else:
        for note_index, note_data in enumerate(cards):
            for direction_index, card in enumerate(enabled):
                events.append((
                    (note_index, direction_index),
                    card.direction_key,
                    "generated",
                    note_data,
                    card,
                ))

    due_by_direction = {
        direction: 0
        for direction in decks
    }
    notes_created = 0
    for _order, direction, kind, data, card in sorted(
            events,
            key=lambda event: event[0]):
        if separate_source_decks:
            due_by_direction[direction] += 1
            due = due_by_direction[direction]
        else:
            notes_created += 1
            due = notes_created
        if kind == "sentence":
            note = _sentence_note(
                data,
                guid_seed=guid_seed,
                due=due)
        elif kind == "lexical":
            note = _lexical_note(
                data,
                pipeline,
                card,
                guid_seed=guid_seed,
                due=due)
        else:
            note = _generated_note(
                data,
                pipeline,
                card,
                guid_seed=guid_seed,
                due=due)
        decks[direction].add_note(note)
        if separate_source_decks:
            notes_created += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_decks = list(dict.fromkeys(decks.values()))
    genanki.Package(
        unique_decks if len(unique_decks) > 1 else unique_decks[0]
    ).write_to_file(output_path)
    return output_path, notes_created


def import_source_package(
        package_path,
        *,
        client=None,
        importer=anki_integration.import_standalone_deck):
    """Import without moving cards or deleting retained source decks."""
    return importer(
        package_path,
        client=client)
