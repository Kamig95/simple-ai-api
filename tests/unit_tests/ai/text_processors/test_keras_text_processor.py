from ai.text_processors.keras_text_processor import KerasTextProcessor


def test_keras_process(mocker):
    model_mock = mocker.MagicMock()
    model_mock.predict.return_value.argmax.return_value = [0]
    labels = {0: "label_0", 1: "label_1"}
    text_processor = KerasTextProcessor(model_mock, labels)

    result = text_processor.process("text")

    assert "label_0" == result
    model_mock.predict.assert_called_once_with(["text"])
    model_mock.predict.return_value.argmax.assert_called_once_with(axis=-1)


def test_keras_process_batch(mocker):
    model_mock = mocker.MagicMock()
    model_mock.predict.return_value.argmax.return_value = [0, 1]
    labels = {0: "label_0", 1: "label_1"}
    text_processor = KerasTextProcessor(model_mock, labels)

    result = text_processor.process_batch(["text_0", "text_1"])

    assert ["label_0", "label_1"] == result
    model_mock.predict.assert_called_once_with(["text_0", "text_1"])
    model_mock.predict.return_value.argmax.assert_called_once_with(axis=-1)
