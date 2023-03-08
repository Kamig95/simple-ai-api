from unittest.mock import call

from torch import tensor

from ai.text_processors.pytorch_text_processor import PytorchTextProcessor

TEST_TEXT = "test_text"
TEST_TEXT_2 = "test_text_2"
MODEL_FILENAME = "test_filename.pt"
LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}
RESULT_LABEL = "Sci/Tec"
MODEL_INPUT = tensor([76, 30980, 29212, 35, 7, 117, 1])
MODEL_OUTPUT = tensor([[0.1152, 0.1483, 0.1505, 0.2129]])


def test_pytorch_process(mocker):
    preprocessor_mock = mocker.MagicMock()
    preprocessor_mock.return_value = MODEL_INPUT
    model_mock = mocker.MagicMock()
    model_mock.return_value = MODEL_OUTPUT
    text_processor = PytorchTextProcessor(
        model_mock, MODEL_FILENAME, LABELS, preprocessor_mock
    )

    result = text_processor.process(TEST_TEXT)

    assert RESULT_LABEL == result
    preprocessor_mock.assert_called_once_with(TEST_TEXT)


def test_pytorch_process_batch(mocker):
    preprocessor_mock = mocker.MagicMock()
    preprocessor_mock.return_value = MODEL_INPUT
    model_mock = mocker.MagicMock()
    model_mock.return_value = MODEL_OUTPUT
    text_processor = PytorchTextProcessor(
        model_mock, MODEL_FILENAME, LABELS, preprocessor_mock
    )

    result = text_processor.process_batch([TEST_TEXT, TEST_TEXT_2])

    assert [RESULT_LABEL, RESULT_LABEL] == result
    preprocessor_mock.assert_has_calls(calls=[call(TEST_TEXT), call(TEST_TEXT_2)])
