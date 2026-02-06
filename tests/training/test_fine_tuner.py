"""Tests for the FineTuner class and training data formatting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from prosody_protocol import Dataset, DatasetEntry

from intent_engine.training.fine_tuner import (
    SUPPORTED_TASKS,
    FineTuner,
    TrainingConfig,
    format_dataset,
    format_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    entry_id: str = "e001",
    text: str = "Hello world",
    emotion: str = "joyful",
) -> DatasetEntry:
    iml = f'<iml><utterance emotion="{emotion}" confidence="0.9">{text}</utterance></iml>'
    return DatasetEntry(
        id=entry_id,
        timestamp="2024-01-01T00:00:00Z",
        source="synthetic",
        language="en",
        audio_file="audio/e001.wav",
        transcript=text,
        iml=iml,
        emotion_label=emotion,
        annotator="human",
        consent=True,
    )


def _make_dataset(n: int = 20) -> Dataset:
    emotions = ["joyful", "sad", "neutral", "angry", "frustrated"]
    entries = [
        _make_entry(f"e{i:03d}", f"Text {i}", emotions[i % len(emotions)])
        for i in range(n)
    ]
    return Dataset(name="test-training", entries=entries)


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    """TrainingConfig data structure and serialization."""

    def test_default_values(self) -> None:
        config = TrainingConfig()
        assert config.base_model == "meta-llama/Llama-3.1-70B"
        assert config.task == "prosody_to_intent"
        assert config.epochs == 3
        assert config.learning_rate == 1e-5
        assert config.lora_r == 16
        assert config.bf16 is True

    def test_custom_values(self) -> None:
        config = TrainingConfig(
            base_model="mistralai/Mistral-7B",
            epochs=5,
            learning_rate=2e-5,
            lora_r=8,
        )
        assert config.base_model == "mistralai/Mistral-7B"
        assert config.epochs == 5
        assert config.lora_r == 8

    def test_to_dict(self) -> None:
        config = TrainingConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["base_model"] == "meta-llama/Llama-3.1-70B"
        assert d["epochs"] == 3
        assert "lora_r" in d
        assert "lora_alpha" in d

    def test_extra_kwargs_in_dict(self) -> None:
        config = TrainingConfig(extra={"custom_flag": True, "custom_val": 42})
        d = config.to_dict()
        assert d["custom_flag"] is True
        assert d["custom_val"] == 42

    def test_save_and_load(self) -> None:
        config = TrainingConfig(
            base_model="test-model",
            epochs=10,
            learning_rate=3e-4,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            config.save(path)
            loaded = TrainingConfig.load(path)
            assert loaded.base_model == "test-model"
            assert loaded.epochs == 10
            assert loaded.learning_rate == 3e-4
        finally:
            Path(path).unlink()

    def test_load_with_extra_keys(self) -> None:
        data = {"base_model": "m", "epochs": 1, "unknown_key": "value"}
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            loaded = TrainingConfig.load(path)
            assert loaded.base_model == "m"
            assert loaded.extra == {"unknown_key": "value"}
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# format_entry / format_dataset
# ---------------------------------------------------------------------------


class TestFormatEntry:
    """IML-aware data formatting."""

    def test_prosody_to_intent(self) -> None:
        entry = _make_entry()
        result = format_entry(entry, "prosody_to_intent")
        assert "text" in result
        assert "IML Input:" in result["text"]
        assert "Hello world" in result["text"]
        assert "Intent:" in result["text"]
        assert "joyful" in result["text"]

    def test_prosody_to_response(self) -> None:
        entry = _make_entry(emotion="sad")
        result = format_entry(entry, "prosody_to_response")
        assert "text" in result
        assert "Response:" in result["text"]
        assert "sad" in result["text"]

    def test_unsupported_task_raises(self) -> None:
        entry = _make_entry()
        with pytest.raises(ValueError, match="Unsupported task"):
            format_entry(entry, "invalid_task")

    def test_format_dataset(self) -> None:
        entries = [_make_entry(f"e{i}") for i in range(5)]
        formatted = format_dataset(entries, "prosody_to_intent")
        assert len(formatted) == 5
        assert all("text" in item for item in formatted)

    def test_supported_tasks(self) -> None:
        assert "prosody_to_intent" in SUPPORTED_TASKS
        assert "prosody_to_response" in SUPPORTED_TASKS
        assert len(SUPPORTED_TASKS) == 2


# ---------------------------------------------------------------------------
# FineTuner construction
# ---------------------------------------------------------------------------


class TestFineTunerConstruction:
    """FineTuner initialization and configuration."""

    def test_default_construction(self) -> None:
        tuner = FineTuner()
        assert tuner.config.base_model == "meta-llama/Llama-3.1-70B"
        assert tuner.config.task == "prosody_to_intent"
        assert tuner.trained_model_path is None

    def test_custom_model(self) -> None:
        tuner = FineTuner(base_model="custom/model")
        assert tuner.config.base_model == "custom/model"

    def test_dataset_object(self) -> None:
        ds = _make_dataset()
        tuner = FineTuner(dataset=ds)
        assert tuner._dataset is ds

    def test_dataset_path(self) -> None:
        tuner = FineTuner(dataset="/path/to/dataset")
        assert tuner._dataset_path == "/path/to/dataset"
        assert tuner._dataset is None

    def test_custom_config(self) -> None:
        config = TrainingConfig(epochs=10, learning_rate=1e-4)
        tuner = FineTuner(config=config)
        assert tuner.config.epochs == 10
        assert tuner.config.learning_rate == 1e-4

    def test_repr(self) -> None:
        tuner = FineTuner(base_model="test/model")
        r = repr(tuner)
        assert "test/model" in r
        assert "prosody_to_intent" in r


# ---------------------------------------------------------------------------
# FineTuner data preparation
# ---------------------------------------------------------------------------


class TestFineTunerPrepareData:
    """FineTuner.prepare_training_data() produces correct splits."""

    def test_prepare_with_loaded_dataset(self) -> None:
        ds = _make_dataset(n=100)
        tuner = FineTuner(dataset=ds)
        splits = tuner.prepare_training_data()
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_prepare_custom_ratios(self) -> None:
        ds = _make_dataset(n=100)
        tuner = FineTuner(dataset=ds)
        splits = tuner.prepare_training_data(
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        assert len(splits["train"]) == 60
        assert len(splits["val"]) == 20
        assert len(splits["test"]) == 20

    def test_prepare_custom_task(self) -> None:
        ds = _make_dataset(n=10)
        tuner = FineTuner(dataset=ds)
        splits = tuner.prepare_training_data(task="prosody_to_response")
        assert "Response:" in splits["train"][0]["text"]

    def test_prepare_without_dataset_raises(self) -> None:
        tuner = FineTuner()
        with pytest.raises(ValueError, match="No dataset available"):
            tuner.prepare_training_data()

    def test_prepare_with_explicit_dataset(self) -> None:
        ds = _make_dataset(n=50)
        tuner = FineTuner()
        splits = tuner.prepare_training_data(dataset=ds)
        assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 50


# ---------------------------------------------------------------------------
# FineTuner.load_dataset
# ---------------------------------------------------------------------------


class TestFineTunerLoadDataset:
    """FineTuner.load_dataset() loads via DatasetLoader."""

    def test_load_from_path(self) -> None:
        tuner = FineTuner()
        ds = _make_dataset()

        with patch.object(tuner._loader, "load", return_value=ds):
            result = tuner.load_dataset("/fake/path")
            assert result is ds
            assert tuner._dataset is ds

    def test_load_uses_constructor_path(self) -> None:
        tuner = FineTuner(dataset="/my/dataset")
        ds = _make_dataset()

        with patch.object(tuner._loader, "load", return_value=ds):
            result = tuner.load_dataset()
            tuner._loader.load.assert_called_once_with("/my/dataset")
            assert result is ds

    def test_load_no_path_raises(self) -> None:
        tuner = FineTuner()
        with pytest.raises(ValueError, match="No dataset path"):
            tuner.load_dataset()


# ---------------------------------------------------------------------------
# FineTuner.train
# ---------------------------------------------------------------------------


class TestFineTunerTrain:
    """FineTuner.train() requires transformers and dataset."""

    def test_train_without_transformers_raises(self) -> None:
        ds = _make_dataset()
        tuner = FineTuner(dataset=ds)

        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                tuner.train()

    def test_train_without_dataset_raises(self) -> None:
        tuner = FineTuner()
        # Mock transformers to be available
        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            with pytest.raises(ValueError, match="No dataset available"):
                tuner.train()

    def test_train_calls_run_training(self) -> None:
        ds = _make_dataset()
        tuner = FineTuner(dataset=ds)

        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            with patch.object(tuner, "_run_training") as mock_run:
                result = tuner.train()
                mock_run.assert_called_once()
                assert result == tuner.config.output_dir

    def test_train_with_overrides(self) -> None:
        ds = _make_dataset()
        tuner = FineTuner(dataset=ds)

        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            with patch.object(tuner, "_run_training") as mock_run:
                tuner.train(epochs=5, learning_rate=1e-3, task="prosody_to_response")
                call_args = mock_run.call_args
                training_args = call_args[0][1]
                assert training_args["num_train_epochs"] == 5
                assert training_args["learning_rate"] == 1e-3

    def test_train_loads_dataset_from_path(self) -> None:
        ds = _make_dataset()
        tuner = FineTuner(dataset="/fake/path")

        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            with patch.object(tuner, "_run_training"):
                with patch.object(tuner, "load_dataset", return_value=ds):
                    tuner._dataset = ds  # simulate load
                    tuner.train()


# ---------------------------------------------------------------------------
# FineTuner.save_config
# ---------------------------------------------------------------------------


class TestFineTunerSaveConfig:
    """FineTuner.save_config() persists configuration."""

    def test_save_config(self) -> None:
        tuner = FineTuner(base_model="test/model")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            tuner.save_config(path)
            data = json.loads(Path(path).read_text())
            assert data["base_model"] == "test/model"
        finally:
            Path(path).unlink()
