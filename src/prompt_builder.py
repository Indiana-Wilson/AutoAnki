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


def build_prompt(pipeline, project_root=None):
    language = pipeline_store.get_language(
        pipeline.language_key)
    components = pipeline_store.prompt_component_map(project_root)
    shared_values = {
        "source_language": language.name,
    }
    sections = []

    for key in ("core", f"languages/{language.key}"):
        sections.append(_format_component(
            _read_component(components, key),
            key,
            **shared_values))

    if pipeline_store.requires_sentences(pipeline):
        key = "directions/context"
        sections.append(_format_component(
            _read_component(components, key),
            key,
            **shared_values))

    for field_setting in (
            pipeline_store.get_requested_field_settings(pipeline)):
        field = pipeline_store.get_field_option(
            field_setting.field_key)
        target_language = pipeline_store.get_language(
            field_setting.target_language_key)
        key = f"fields/{field.key}"
        sections.append(_format_component(
            _read_component(components, key),
            key,
            **shared_values,
            target_language=target_language.name,
            response_field=(
                pipeline_store.response_field_name(field_setting))))

    key = "ending"
    sections.append(_format_component(
        _read_component(components, key),
        key,
        **shared_values))
    return "\n\n".join(sections) + "\n"
