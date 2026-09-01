import unittest

from llm_proxy.stt_protocol import (
    ENCODED_INITIAL_PROMPT_HEADER,
    INITIAL_PROMPT_HEADER,
    InitialPromptEncodingError,
    decode_initial_prompt,
    forward_initial_prompt_headers,
)


class InitialPromptProtocolTests(unittest.TestCase):
    def test_decodes_unicode_v1_header(self):
        headers = {
            ENCODED_INITIAL_PROMPT_HEADER:
                "v1:0JDQstCw0LvQvtC9IOKAlCDQs9C+0LvQvtGB0L7QstC+0Lkg0LDRgdGB0LjRgdGC0LXQvdGC"
        }

        self.assertEqual(
            decode_initial_prompt(headers), "Авалон — голосовой ассистент"
        )

    def test_keeps_plain_ascii_header(self):
        self.assertEqual(
            decode_initial_prompt({INITIAL_PROMPT_HEADER: "Avalon names"}),
            "Avalon names",
        )

    def test_rejects_malformed_or_ambiguous_headers(self):
        malformed = (
            {ENCODED_INITIAL_PROMPT_HEADER: "v2:YWJj"},
            {ENCODED_INITIAL_PROMPT_HEADER: "v1:not base64!"},
            {ENCODED_INITIAL_PROMPT_HEADER: "v1://4="},
            {
                INITIAL_PROMPT_HEADER: "plain",
                ENCODED_INITIAL_PROMPT_HEADER: "v1:cGxhaW4=",
            },
        )
        for headers in malformed:
            with self.subTest(headers=headers), self.assertRaises(
                InitialPromptEncodingError
            ):
                decode_initial_prompt(headers)

    def test_proxy_forwards_encoded_header_unchanged(self):
        source = {ENCODED_INITIAL_PROMPT_HEADER: "v1:0JDQstCw0LvQvtC9"}
        target = {"Content-Type": "application/octet-stream"}

        forward_initial_prompt_headers(source, target)

        self.assertEqual(target[ENCODED_INITIAL_PROMPT_HEADER], source[ENCODED_INITIAL_PROMPT_HEADER])


if __name__ == "__main__":
    unittest.main()
