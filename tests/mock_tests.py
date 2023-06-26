import base64


class MockGetResponse:
    def __init__(self):
        self.content = base64.b64encode(b"test\n :\n 1")

    def json(self):
        return {"sha": "sha", "content": self.content, "encoding": "base64"}


class MockPostResponse:
    def json(self):
        return {"objects": [{"actions": {"download": {"href": "test"}}}]}


def mock_get(*args, **kwargs):
    return MockGetResponse()


def mock_post(*args, **kwargs):
    return MockPostResponse()


def mock_train(*args, **kwargs):
    pass
