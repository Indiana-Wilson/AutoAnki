"""Compose the minimum prompt required by a configurable pipeline."""

import pipeline_store


class PromptComponentError(ValueError):
    pass


def _read_component(components, key):
    component = components.get(key)
    if component is None:
        raise PromptComponentError(
            f'Required prompt component "{key}" was not found.')
    try:
        return component.path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise PromptComponentError(
            f'Could not read prompt component "{key}": {error}'
        ) from error


def _format_component(text, key, **values):
    try:
        return text.format(**values)
    except (KeyError, ValueError) as error:
        raise PromptComponentError(
            f'Prompt component "{key}" contains an invalid placeholder: '
            f"{error}") from error


def build_prompt(
        pipeline,
        project_root=None,
        *,
        include_sentence_translations=None,
        include_example_sentence_instructions=None,
        requested_field_settings=None,
        core_component_key=None,
        include_language_instructions=True,
        sentence_collections_as_arrays=False,
        grouped_source_examples=False,
        compact_source_examples=False,
        compact_source_protocol=9,
        include_ending=True):
    if (
            compact_source_examples
            and compact_source_protocol not in {9, 10}):
        raise ValueError(
            "Compact source examples require protocol 9 or 10.")
    language = pipeline_store.get_language(
        pipeline.language_key)
    if include_sentence_translations is None:
        include_sentence_translations = (
            pipeline_store.requires_sentence_translations(pipeline))
    components = pipeline_store.prompt_component_map(project_root)
    shared_values = {
        "source_language": language.name,
    }
    sections = []

    core_key = (
        core_component_key
        if core_component_key is not None
        else (
            f"core_source_v{compact_source_protocol}"
            if compact_source_examples
            else "core"))
    component_keys = [core_key]
    if include_language_instructions:
        component_keys.append(f"languages/{language.key}")
    for key in component_keys:
        sections.append(_format_component(
            _read_component(components, key),
            key,
            **shared_values))

    if include_example_sentence_instructions is None:
        include_example_sentence_instructions = (
            pipeline_store.requires_sentences(pipeline))
    if include_example_sentence_instructions:
        if compact_source_examples:
            key = (
                f"directions/context_arrays_v"
                f"{compact_source_protocol}")
        elif sentence_collections_as_arrays:
            key = (
                "directions/context_arrays_v8"
                if grouped_source_examples
                else "directions/context_arrays")
        else:
            key = "directions/context"
        sections.append(_format_component(
            _read_component(components, key),
            key,
            **shared_values))
        if (
                language.model_language_key == "classical_chinese"
                and not compact_source_examples):
            key = (
                "directions/context_classical_chinese_arrays"
                if sentence_collections_as_arrays
                else "directions/context_classical_chinese")
            sections.append(_format_component(
                _read_component(components, key),
                key,
                **shared_values))
        if include_sentence_translations:
            if sentence_collections_as_arrays:
                key = (
                    f"directions/sentence_translation_arrays_v"
                    f"{compact_source_protocol}"
                    if compact_source_examples
                    else "directions/sentence_translation_arrays")
            else:
                key = "directions/sentence_translations"
            sections.append(_format_component(
                _read_component(components, key),
                key,
                **shared_values))
            if language.key == "english":
                key = "directions/sentence_translations_english"
                sections.append(_format_component(
                    _read_component(components, key),
                    key,
                    **shared_values))

    if requested_field_settings is None:
        requested_field_settings = (
            pipeline_store.get_requested_field_settings(pipeline))
    for field_setting in requested_field_settings:
        field = pipeline_store.get_field_option(
            field_setting.field_key)
        target_language = pipeline_store.get_language(
            field_setting.target_language_key)
        key = (
            "fields/dictionary_meaning_v10"
            if (
                compact_source_examples
                and compact_source_protocol == 10
                and field.key == "dictionary_meaning")
            else f"fields/{field.key}")
        sections.append(_format_component(
            _read_component(components, key),
            key,
            **shared_values,
            target_language=target_language.name,
            response_field=(
                pipeline_store.response_field_name(field_setting))))

    if include_ending:
        key = "ending"
        sections.append(_format_component(
            _read_component(components, key),
            key,
            **shared_values))
    return "\n\n".join(sections) + "\n"
