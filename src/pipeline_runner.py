from dataclasses import dataclass
from pathlib import Path

import anki_integration
import pipeline_store
import process_text
import templates


@dataclass(frozen=True)
class PipelineProgress:
    pipeline: pipeline_store.PipelineConfig
    index: int
    total: int
    stage: str


@dataclass(frozen=True)
class PipelineSuccess:
    pipeline: pipeline_store.PipelineConfig
    output_path: Path
    import_result: anki_integration.AnkiImportResult


@dataclass(frozen=True)
class PipelineFailure:
    pipeline: pipeline_store.PipelineConfig
    stage: str
    error: Exception
    output_path: Path | None = None


@dataclass(frozen=True)
class PipelineRunSummary:
    successes: tuple[PipelineSuccess, ...]
    failures: tuple[PipelineFailure, ...]

    @property
    def cards_moved(self):
        return sum(
            success.import_result.cards_moved
            for success in self.successes)


def get_pipeline_output_paths(pipeline, output_root):
    output_root = Path(output_root)
    if pipeline.pipeline_id == pipeline_store.DEFAULT_PIPELINE_ID:
        return {
            "response": output_root / "response.json",
            "response_log": output_root / "response_log",
            "package": output_root / "output.apkg",
        }

    pipeline_directory = (
        output_root / "pipelines" / pipeline.pipeline_id)
    return {
        "response": pipeline_directory / "response.json",
        "response_log": pipeline_directory / "response_log",
        "package": pipeline_directory / "output.apkg",
    }


def run_pipelines(
        words,
        pipelines,
        *,
        project_root=process_text.PROJECT_ROOT,
        output_root=None,
        progress_callback=None,
        openai_client=None,
        anki_client=None,
        generator=process_text.generate_deck,
        importer=anki_integration.import_generated_deck):
    pipelines = pipeline_store.validate_pipelines(pipelines)
    if not pipelines:
        raise ValueError("Add at least one pipeline before generating.")

    project_root = Path(project_root)
    output_root = Path(
        output_root or project_root / "output")
    prompts = pipeline_store.prompt_map(project_root)
    successes = []
    failures = []
    total = len(pipelines)

    for index, pipeline in enumerate(pipelines, start=1):
        prompt_key = pipeline_store.get_prompt_key(pipeline)
        prompt = prompts.get(prompt_key)
        if prompt is None:
            failures.append(PipelineFailure(
                pipeline=pipeline,
                stage="configuration",
                error=ValueError(
                    f'Prompt "{prompt_key}" was not found.')))
            continue

        output_paths = get_pipeline_output_paths(
            pipeline,
            output_root)
        if progress_callback:
            progress_callback(PipelineProgress(
                pipeline=pipeline,
                index=index,
                total=total,
                stage="generating"))

        try:
            package_path = generator(
                words,
                client=openai_client,
                prompt_path=prompt.path,
                response_path=output_paths["response"],
                response_log_path=output_paths["response_log"],
                output_path=output_paths["package"],
                deck_id=pipeline.generated_deck_id,
                deck_name=pipeline.generated_deck_name,
                card_type_keys=pipeline.card_type_keys,
                guid_seed=pipeline.pipeline_id,
                legacy_guid_card_type_key=(
                    templates.DEFAULT_CARD_TYPE_KEY
                    if pipeline.pipeline_id
                    == pipeline_store.DEFAULT_PIPELINE_ID
                    else None))
        except Exception as error:
            failures.append(PipelineFailure(
                pipeline=pipeline,
                stage="generation",
                error=error))
            continue

        package_path = Path(package_path)
        if progress_callback:
            progress_callback(PipelineProgress(
                pipeline=pipeline,
                index=index,
                total=total,
                stage="importing"))

        try:
            import_result = importer(
                package_path,
                target_deck=pipeline.target_deck,
                source_deck=pipeline.generated_deck_name,
                client=anki_client)
        except Exception as error:
            failures.append(PipelineFailure(
                pipeline=pipeline,
                stage="import",
                error=error,
                output_path=package_path))
            continue

        successes.append(PipelineSuccess(
            pipeline=pipeline,
            output_path=package_path,
            import_result=import_result))

    return PipelineRunSummary(
        successes=tuple(successes),
        failures=tuple(failures))
