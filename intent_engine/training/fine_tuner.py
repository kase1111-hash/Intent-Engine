"""FineTuner -- configurable fine-tuning for local LLMs.

Wraps Hugging Face ``transformers`` (or ``llama-cpp-python``)
fine-tuning with IML-aware data formatting so models learn to
understand prosody markup natively.

Depends on:
    - ``prosody_protocol.DatasetLoader`` for loading training data
    - ``prosody_protocol.Dataset`` / ``DatasetEntry`` for typed entries
    - Optional: ``transformers``, ``peft``, ``trl`` for actual training
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from prosody_protocol import DatasetEntry, DatasetLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameter configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters and training configuration.

    Attributes
    ----------
    base_model:
        Hugging Face model identifier or local path (e.g.,
        ``"meta-llama/Llama-3.1-70B"``).
    task:
        Training task identifier.  Supported tasks:

        * ``"prosody_to_intent"`` -- given IML, predict intent + emotion
        * ``"prosody_to_response"`` -- given IML, generate an
          emotionally appropriate response

    epochs:
        Number of training epochs.
    learning_rate:
        Learning rate for the optimizer.
    batch_size:
        Per-device training batch size.
    max_seq_length:
        Maximum token sequence length.
    warmup_ratio:
        Fraction of total steps used for learning-rate warmup.
    weight_decay:
        L2 regularization weight decay.
    lora_r:
        LoRA rank (``0`` disables LoRA and does full fine-tuning).
    lora_alpha:
        LoRA scaling factor.
    lora_dropout:
        LoRA dropout probability.
    output_dir:
        Directory for checkpoints and final model.
    seed:
        Random seed for reproducibility.
    gradient_accumulation_steps:
        Gradient accumulation steps before optimizer update.
    fp16:
        Whether to use FP16 mixed precision.
    bf16:
        Whether to use BF16 mixed precision.
    extra:
        Additional trainer-specific kwargs passed through.
    """

    base_model: str = "meta-llama/Llama-3.1-70B"
    task: str = "prosody_to_intent"
    epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    output_dir: str = "./fine_tuned_model"
    seed: int = 42
    gradient_accumulation_steps: int = 4
    fp16: bool = False
    bf16: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dict."""
        d = {
            "base_model": self.base_model,
            "task": self.task,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": self.fp16,
            "bf16": self.bf16,
        }
        d.update(self.extra)
        return d

    def save(self, path: str | Path) -> None:
        """Save config as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> TrainingConfig:
        """Load config from a JSON file."""
        data = json.loads(Path(path).read_text())
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        known = {k: v for k, v in data.items() if k in known_fields and k != "extra"}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        return cls(**known, extra=extra)


# ---------------------------------------------------------------------------
# IML-aware data formatting
# ---------------------------------------------------------------------------


_PROSODY_TO_INTENT_TEMPLATE = (
    "Below is an utterance transcribed with IML (Intent Markup Language) "
    "prosody annotations. Determine the speaker's intent and emotional state.\n\n"
    "### IML Input:\n{iml}\n\n"
    "### Transcript:\n{transcript}\n\n"
    "### Analysis:\n"
    "Intent: {intent}\n"
    "Emotion: {emotion}\n"
)

_PROSODY_TO_RESPONSE_TEMPLATE = (
    "Below is an utterance transcribed with IML (Intent Markup Language) "
    "prosody annotations. Generate an emotionally appropriate response.\n\n"
    "### IML Input:\n{iml}\n\n"
    "### Transcript:\n{transcript}\n\n"
    "### Response:\n"
    "{response}\n"
)

_TEMPLATES: dict[str, str] = {
    "prosody_to_intent": _PROSODY_TO_INTENT_TEMPLATE,
    "prosody_to_response": _PROSODY_TO_RESPONSE_TEMPLATE,
}

SUPPORTED_TASKS = list(_TEMPLATES.keys())


def format_entry(entry: DatasetEntry, task: str) -> dict[str, str]:
    """Format a ``DatasetEntry`` into a training example.

    Parameters
    ----------
    entry:
        A single dataset entry with IML, transcript, and emotion label.
    task:
        One of ``"prosody_to_intent"`` or ``"prosody_to_response"``.

    Returns
    -------
    dict[str, str]
        A dict with ``"text"`` key containing the formatted prompt.

    Raises
    ------
    ValueError
        If the task is not supported.
    """
    if task not in _TEMPLATES:
        raise ValueError(
            f"Unsupported task '{task}'. Choose from: {SUPPORTED_TASKS}"
        )

    template = _TEMPLATES[task]

    if task == "prosody_to_intent":
        # Use emotion_label as both intent proxy and emotion
        text = template.format(
            iml=entry.iml,
            transcript=entry.transcript,
            intent=entry.emotion_label,
            emotion=entry.emotion_label,
        )
    else:
        # prosody_to_response -- use transcript as a placeholder response
        text = template.format(
            iml=entry.iml,
            transcript=entry.transcript,
            response=f"[Responding with {entry.emotion_label} tone]",
        )

    return {"text": text}


def format_dataset(
    entries: list[DatasetEntry], task: str
) -> list[dict[str, str]]:
    """Format a list of entries into training examples."""
    return [format_entry(e, task) for e in entries]


# ---------------------------------------------------------------------------
# FineTuner
# ---------------------------------------------------------------------------


class FineTuner:
    """Fine-tunes local LLMs on prosody-protocol datasets.

    Wraps Hugging Face ``transformers`` / ``peft`` / ``trl`` training
    with IML-aware data formatting.  Loading the actual training
    libraries is deferred until :meth:`train` is called, so this
    class can be imported without GPU dependencies installed.

    Parameters
    ----------
    base_model:
        Model identifier or path.  Overrides ``config.base_model``
        when provided.
    dataset:
        Path to a prosody-protocol dataset directory, or a
        pre-loaded ``Dataset`` object.
    config:
        Training hyperparameters.  Uses defaults if not provided.
    """

    def __init__(
        self,
        base_model: str | None = None,
        dataset: str | Dataset | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        self._config = config or TrainingConfig()
        if base_model is not None:
            self._config = TrainingConfig(
                **{**self._config.to_dict(), "base_model": base_model}
            )

        self._dataset: Dataset | None = None
        self._dataset_path: str | None = None

        if isinstance(dataset, Dataset):
            self._dataset = dataset
        elif isinstance(dataset, str):
            self._dataset_path = dataset

        self._loader = DatasetLoader(validate_iml=True)
        self._trained_model_path: str | None = None

        logger.info(
            "FineTuner initialized (model=%s, task=%s)",
            self._config.base_model,
            self._config.task,
        )

    @property
    def config(self) -> TrainingConfig:
        """Return the current training config."""
        return self._config

    @property
    def trained_model_path(self) -> str | None:
        """Path to the last trained model output, or ``None``."""
        return self._trained_model_path

    def load_dataset(self, path: str | None = None) -> Dataset:
        """Load a dataset from disk using ``DatasetLoader``.

        Parameters
        ----------
        path:
            Path to dataset directory.  Falls back to the path
            provided at construction time.

        Returns
        -------
        Dataset
            The loaded dataset.

        Raises
        ------
        ValueError
            If no path is available.
        """
        target = path or self._dataset_path
        if target is None:
            raise ValueError("No dataset path provided")

        self._dataset = self._loader.load(target)
        logger.info(
            "Loaded dataset '%s' with %d entries",
            self._dataset.name,
            self._dataset.size,
        )
        return self._dataset

    def prepare_training_data(
        self,
        dataset: Dataset | None = None,
        task: str | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        """Prepare training, validation, and test splits.

        Parameters
        ----------
        dataset:
            Dataset to prepare.  Uses the internally loaded dataset
            if not provided.
        task:
            Training task.  Defaults to ``config.task``.
        train_ratio, val_ratio, test_ratio:
            Split ratios (must sum to 1.0).
        seed:
            Random seed for splitting.

        Returns
        -------
        dict[str, list[dict[str, str]]]
            Keys ``"train"``, ``"val"``, ``"test"`` mapping to
            lists of formatted examples.
        """
        ds = dataset or self._dataset
        if ds is None:
            raise ValueError("No dataset available -- call load_dataset() first")

        effective_task = task or self._config.task
        effective_seed = seed if seed is not None else self._config.seed

        train_entries, val_entries, test_entries = self._loader.split(
            ds,
            train=train_ratio,
            val=val_ratio,
            test=test_ratio,
            seed=effective_seed,
        )

        return {
            "train": format_dataset(train_entries, effective_task),
            "val": format_dataset(val_entries, effective_task),
            "test": format_dataset(test_entries, effective_task),
        }

    def train(
        self,
        epochs: int | None = None,
        learning_rate: float | None = None,
        task: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Run fine-tuning and return the output model path.

        Requires ``transformers``, ``peft``, and ``trl`` to be
        installed.  Raises ``ImportError`` if missing.

        Parameters
        ----------
        epochs:
            Override ``config.epochs``.
        learning_rate:
            Override ``config.learning_rate``.
        task:
            Override ``config.task``.
        **kwargs:
            Additional overrides passed to the trainer.

        Returns
        -------
        str
            Path to the saved fine-tuned model.

        Raises
        ------
        ImportError
            If training dependencies are not installed.
        ValueError
            If no dataset has been loaded.
        """
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "Fine-tuning requires 'transformers', 'peft', and 'trl'. "
                "Install with: pip install transformers peft trl"
            ) from None

        if self._dataset is None:
            if self._dataset_path:
                self.load_dataset()
            else:
                raise ValueError(
                    "No dataset available -- provide a dataset path or "
                    "call load_dataset() before training"
                )

        effective_task = task or self._config.task
        effective_epochs = epochs if epochs is not None else self._config.epochs
        effective_lr = (
            learning_rate if learning_rate is not None else self._config.learning_rate
        )

        logger.info(
            "Starting fine-tuning: model=%s, task=%s, epochs=%d, lr=%s",
            self._config.base_model,
            effective_task,
            effective_epochs,
            effective_lr,
        )

        splits = self.prepare_training_data(task=effective_task)

        # Build trainer configuration
        output_dir = kwargs.pop("output_dir", self._config.output_dir)

        training_args = {
            "output_dir": output_dir,
            "num_train_epochs": effective_epochs,
            "learning_rate": effective_lr,
            "per_device_train_batch_size": self._config.batch_size,
            "warmup_ratio": self._config.warmup_ratio,
            "weight_decay": self._config.weight_decay,
            "seed": self._config.seed,
            "gradient_accumulation_steps": self._config.gradient_accumulation_steps,
            "fp16": self._config.fp16,
            "bf16": self._config.bf16,
            **kwargs,
        }

        self._run_training(splits, training_args)

        self._trained_model_path = output_dir
        logger.info("Fine-tuning complete. Model saved to: %s", output_dir)

        return output_dir

    def _run_training(
        self,
        splits: dict[str, list[dict[str, str]]],
        training_args: dict[str, Any],
    ) -> None:
        """Execute the actual training loop.

        This is separated to allow subclasses or mocks to override
        the training backend.

        Parameters
        ----------
        splits:
            Formatted train/val/test data.
        training_args:
            Arguments for the training framework.
        """
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )

        tokenizer = AutoTokenizer.from_pretrained(self._config.base_model)
        model = AutoModelForCausalLM.from_pretrained(self._config.base_model)

        # Apply LoRA if configured
        if self._config.lora_r > 0:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self._config.lora_r,
                lora_alpha=self._config.lora_alpha,
                lora_dropout=self._config.lora_dropout,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        # Tokenize
        def _tokenize(examples: list[dict[str, str]]) -> list[dict[str, Any]]:
            return [
                tokenizer(
                    ex["text"],
                    truncation=True,
                    max_length=self._config.max_seq_length,
                    padding="max_length",
                )
                for ex in examples
            ]

        train_encodings = _tokenize(splits["train"])
        val_encodings = _tokenize(splits["val"]) if splits["val"] else None

        args = TrainingArguments(**training_args)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_encodings,
            eval_dataset=val_encodings,
        )

        trainer.train()
        trainer.save_model(training_args["output_dir"])
        tokenizer.save_pretrained(training_args["output_dir"])

    def save_config(self, path: str | Path) -> None:
        """Save the current training configuration."""
        self._config.save(path)

    def __repr__(self) -> str:
        return (
            f"FineTuner(model={self._config.base_model!r}, "
            f"task={self._config.task!r}, "
            f"dataset={'loaded' if self._dataset else 'not loaded'})"
        )
