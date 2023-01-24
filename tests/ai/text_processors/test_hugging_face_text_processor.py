from ai.text_processors.hugging_face_text_processor import HuggingFaceTextProcessor


TEST_LABEL = "POSITIVE"
TEST_TEXT = "test_text"


def test_hugging_face_process(mocker):
    model_mock = mocker.MagicMock()
    model_mock.return_value = [{"label": TEST_LABEL}]
    text_processor = HuggingFaceTextProcessor(model_mock)

    result = text_processor.process(TEST_TEXT)

    assert result == TEST_LABEL
    model_mock.assert_called_once_with(TEST_TEXT)
