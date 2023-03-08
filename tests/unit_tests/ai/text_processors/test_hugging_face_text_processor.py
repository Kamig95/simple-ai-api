from ai.text_processors.hugging_face_text_processor import HuggingFaceTextProcessor


POSITIVE_LABEL = "POSITIVE"
TEST_TEXT = "test_text"
RESULT_KEY = "label"


def test_hugging_face_process(mocker):
    model_mock = mocker.MagicMock()
    model_mock.return_value = [{RESULT_KEY: POSITIVE_LABEL}]
    text_processor = HuggingFaceTextProcessor(model_mock, RESULT_KEY)

    result = text_processor.process(TEST_TEXT)

    assert POSITIVE_LABEL == result
    model_mock.assert_called_once_with(TEST_TEXT)


def test_hugging_face_process_batch(mocker):
    model_mock = mocker.MagicMock()
    model_mock.return_value = [
        {RESULT_KEY: POSITIVE_LABEL},
        {RESULT_KEY: POSITIVE_LABEL},
    ]
    text_processor = HuggingFaceTextProcessor(model_mock, RESULT_KEY)

    input_texts = [TEST_TEXT, TEST_TEXT]
    result = text_processor.process_batch(input_texts)

    assert [POSITIVE_LABEL, POSITIVE_LABEL] == result
    model_mock.assert_called_once_with(input_texts)
