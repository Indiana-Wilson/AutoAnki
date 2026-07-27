from dataclasses import dataclass
from pathlib import Path

import anki_integration
import pipeline_store
import prompt_builder
import process_text


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
        project_root=None,
        output_root=None,
        progress_callback=None,
        paid_dispatch_control=None,
        openai_client=None,
        anki_client=None,
        generator=process_text.generate_deck,
        importer=anki_integration.import_generated_deck):
    pipelines = pipeline_store.validate_pipelines(pipelines)
    if not pipelines:
        raise ValueError("Add at least one pipeline before generating.")

    output_root = Path(
        output_root
        or (
            Path(project_root) / "output"
            if project_root is not None
            else process_text.OUTPUT_DIRECTORY))
    successes = []
    failures = []
    total = len(pipelines)

    for index, pipeline in enumerate(pipelines, start=1):
        try:
            prompt_text = prompt_builder.build_prompt(
                pipeline,
                project_root)
        except Exception as error:
            failures.append(PipelineFailure(
                pipeline=pipeline,
                stage="configuration",
                error=error))
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
            generation_arguments = {
                "client": openai_client,
                "prompt_text": prompt_text,
                "response_path": output_paths["response"],
                "response_log_path": output_paths["response_log"],
                "output_path": output_paths["package"],
                "deck_id": pipeline.generated_deck_id,
                "deck_name": pipeline.generated_deck_name,
                "pipeline": pipeline,
                "guid_seed": pipeline.pipeline_id,
            }
            if paid_dispatch_control is not None:
                generation_arguments["paid_dispatch_control"] = (
                    paid_dispatch_control)
            package_path = generator(
                words,
                **generation_arguments)
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
            import_arguments = {
                "target_deck": pipeline.target_deck,
                "source_deck": pipeline.generated_deck_name,
                "client": anki_client,
            }
            if pipeline.separate_target_decks:
                import_arguments["target_decks"] = (
                    pipeline_store.get_model_target_decks(pipeline))
            import_result = importer(
                package_path,
                **import_arguments)
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
