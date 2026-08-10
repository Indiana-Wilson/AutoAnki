"""Package and import completed source-generation jobs."""

import hashlib
import html
import json
from pathlib import Path
import tempfile

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
        due,
        enhanced_payload=None):
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
    if card.enhanced:
        fields.extend(_enhanced_note_fields(enhanced_payload))
    card_type = templates.get_direction_card_type(
        language.model_language_key,
        card.direction_key,
        enhanced=card.enhanced)
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
        due,
        enhanced_payload=None):
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
        card.direction_key,
        enhanced=card.enhanced)
    fields = [
        note_data[language.term_field],
        sentences,
        *(
            note_data.get(
                response_fields_by_key.get(field_key, ""),
                "")
            for field_key, _field_name in templates.CONTENT_FIELDS
        ),
        translations,
    ]
    if card.enhanced:
        fields.extend(_enhanced_note_fields(enhanced_payload))
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


def _sentence_note(
        context,
        *,
        guid_seed,
        due,
        enhanced_payload=None):
    def safe_text(value):
        return html.escape(html.unescape(value), quote=False)

    return genanki.Note(
        model=(
            templates.ENHANCED_SOURCE_SENTENCE_MODEL
            if enhanced_payload is not None
            else templates.SOURCE_SENTENCE_MODEL),
        fields=[
            safe_text(context["original_sentence"]),
            safe_text(context["english_translation"]),
            safe_text(context.get("nuance", "")),
            *(
                _enhanced_source_sentence_fields(enhanced_payload)
                if enhanced_payload is not None
                else ()
            ),
        ],
        due=due,
        guid=genanki.guid_for(
            "autoanki",
            guid_seed,
            (
                "enhanced_source_sentence"
                if enhanced_payload is not None
                else "source_sentence"),
            context["sentence_id"]))


def _sound_field(artifact):
    return (
        f"[sound:{artifact.media_filename}]"
        if artifact is not None
        else "")


def _enhanced_note_fields(payload):
    if not isinstance(payload, dict):
        raise ValueError(
            "Enhanced card packaging requires synthesized local audio.")
    audio_first = bool(payload.get("audio_first"))
    return [
        _sound_field(payload.get("word_audio")),
        payload.get("sentence", ""),
        payload.get("sentence_translation", ""),
        _sound_field(payload.get("sentence_audio")),
        "1" if audio_first else "",
        "" if audio_first else "1",
        payload["presentation_key"],
    ]


def _enhanced_source_sentence_fields(payload):
    if not isinstance(payload, dict):
        raise ValueError(
            "Enhanced source-sentence packaging requires local audio.")
    audio_first = bool(payload.get("audio_first"))
    return [
        _sound_field(payload.get("sentence_audio")),
        "1" if audio_first else "",
        "" if audio_first else "1",
        payload["presentation_key"],
    ]


def _prepare_enhanced_event_audio(
        ordered_events,
        *,
        pipeline,
        guid_seed,
        audio_service=None,
        audio_progress_callback=None):
    import enhanced_audio

    context_card = next((
        card
        for card in pipeline_store.get_enabled_cards(pipeline)
        if card.direction_key == "context"
    ), None)
    language = pipeline_store.get_language(pipeline.language_key)
    payloads = {}
    requests = []
    request_slots = []
    for event_index, (
            _order,
            direction,
            kind,
            data,
            card) in enumerate(ordered_events):
        effective_card = card if card is not None else context_card
        if effective_card is None or not effective_card.enhanced:
            continue
        presentation_key = enhanced_audio.deterministic_presentation_key(
            guid_seed,
            pipeline.language_key,
            direction,
            kind,
            event_index,
            (
                data.get("sentence_id")
                if isinstance(data, dict)
                else None),
            (
                data.get(language.term_field)
                if isinstance(data, dict)
                else None))
        audio_first = (
            direction in {"context", "word_to_meaning"}
            and enhanced_audio.deterministic_audio_first(
                presentation_key))
        payload = {
            "presentation_key": presentation_key,
            "audio_first": audio_first,
            "word_audio": None,
            "sentence_audio": None,
            "sentence": "",
            "sentence_translation": "",
        }
        payloads[event_index] = payload
        if kind == "sentence":
            sentence = data["original_sentence"]
            payload["sentence"] = html.escape(
                html.unescape(sentence),
                quote=False)
            payload["sentence_translation"] = html.escape(
                html.unescape(data["english_translation"]),
                quote=False)
            requests.append(enhanced_audio.AudioRequest(
                sentence,
                pipeline.language_key,
                "sentence"))
            request_slots.append((event_index, "sentence_audio"))
        elif direction == "context":
            term = data[language.term_field]
            sentences = data.get("Sentences", "").split("|")
            translations_text = data.get(
                process_text.SENTENCE_TRANSLATIONS_FIELD_NAME,
                "")
            translations = (
                translations_text.split("|")
                if translations_text
                else [""] * len(sentences))
            if (
                    not sentences
                    or not sentences[0].strip()
                    or len(sentences) != len(translations)):
                raise ValueError(
                    "Enhanced Sentence → Meaning cards require aligned "
                    "example sentences and any enabled English "
                    "translations.")
            choice = enhanced_audio.deterministic_choice_index(
                len(sentences),
                presentation_key)
            payload["sentence"] = (
                process_text.emphasize_term_in_sentences(
                    sentences[choice],
                    term,
                    pipeline.language_key))
            payload["sentence_translation"] = (
                process_text.sanitize_sentence_translations(
                    translations[choice]))
            requests.append(enhanced_audio.AudioRequest(
                payload["sentence"],
                pipeline.language_key,
                "sentence"))
            request_slots.append((event_index, "sentence_audio"))
        else:
            term = data[language.term_field]
            requests.append(enhanced_audio.AudioRequest(
                term,
                pipeline.language_key,
                "word"))
            request_slots.append((event_index, "word_audio"))

    if not requests:
        return payloads, ()
    service = audio_service or enhanced_audio.LocalTTSService()
    synthesis_arguments = {}
    if audio_progress_callback is not None:
        synthesis_arguments["progress_callback"] = audio_progress_callback
    artifacts = service.synthesize_many(
        requests,
        **synthesis_arguments)
    for (event_index, field_name), artifact in zip(
            request_slots,
            artifacts,
            strict=True):
        payloads[event_index][field_name] = artifact
    return payloads, artifacts


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
        require_sentence_translations=None,
        separate_source_decks=False,
        shared_source_sentence_cards=None,
        audio_service=None,
        audio_media_directory=None,
        audio_progress_callback=None):
    """Validate output and package selected source card directions."""
    if require_sentence_translations is None:
        require_sentence_translations = (
            pipeline_store.requires_sentence_translations(pipeline))
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
                or f"source:{source_key}:{pipeline.pipeline_id}"),
            audio_service=audio_service,
            audio_media_directory=audio_media_directory,
            audio_progress_callback=audio_progress_callback)
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

    ordered_events = sorted(
        events,
        key=lambda event: event[0])
    enhanced_payloads, audio_artifacts = (
        _prepare_enhanced_event_audio(
            ordered_events,
            pipeline=pipeline,
            guid_seed=guid_seed,
            audio_service=audio_service,
            audio_progress_callback=audio_progress_callback))

    due_by_direction = {
        direction: 0
        for direction in decks
    }
    notes_created = 0
    for event_index, (
            _order,
            direction,
            kind,
            data,
            card) in enumerate(ordered_events):
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
                due=due,
                enhanced_payload=enhanced_payloads.get(event_index))
        elif kind == "lexical":
            note = _lexical_note(
                data,
                pipeline,
                card,
                guid_seed=guid_seed,
                due=due,
                enhanced_payload=enhanced_payloads.get(event_index))
        else:
            note = _generated_note(
                data,
                pipeline,
                card,
                guid_seed=guid_seed,
                due=due,
                enhanced_payload=enhanced_payloads.get(event_index))
        decks[direction].add_note(note)
        if separate_source_decks:
            notes_created += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_decks = list(dict.fromkeys(decks.values()))
    package_target = (
        unique_decks
        if len(unique_decks) > 1
        else unique_decks[0])
    if audio_artifacts:
        import enhanced_audio

        def write_audio_package(media_directory):
            enhanced_audio.materialize_media(
                audio_artifacts,
                media_directory)
            media_files = [
                str(media_directory / artifact.media_filename)
                for artifact in dict.fromkeys(audio_artifacts)
            ]
            genanki.Package(
                package_target,
                media_files=media_files).write_to_file(output_path)

        if audio_media_directory is None:
            with tempfile.TemporaryDirectory(
                    prefix=".autoanki-source-audio-",
                    dir=output_path.parent) as directory:
                write_audio_package(Path(directory))
        else:
            write_audio_package(Path(audio_media_directory))
    else:
        genanki.Package(package_target).write_to_file(output_path)
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
