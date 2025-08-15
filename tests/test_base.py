# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from argilla_haystack import ArgillaCallbackHandler
from haystack.agents.base import Agent
from haystack.schema import Answer


class TestArgillaCallback(unittest.TestCase):
    def setUp(self):
        self.prompt_node = MagicMock()
        self.agent = Agent(prompt_node=self.prompt_node)
        self.dataset_name = "test_dataset"
        self.api_url = "http://localhost:6900/"
        self.api_key = "argilla.apikey"
        self.callback = ArgillaCallbackHandler(
            agent=self.agent,
            dataset_name=self.dataset_name,
            api_url=self.api_url,
            api_key=self.api_key,
        )

    def test_on_agent_start(self):
        self.callback.on_agent_start(name="test_agent", query="test_query", params={})
        self.assertIsInstance(self.callback.start_time, datetime)
        self.assertEqual(self.callback.tool_start_times, {})
        self.assertEqual(self.callback.tool_durations, {})
        self.assertEqual(self.callback.metadata, {})

    def test_on_agent_step(self):
        self.callback.on_agent_step(agent_step=MagicMock())

    def test_on_agent_finish(self):
        self.callback.on_agent_finish(agent_step=MagicMock())
        self.assertIsInstance(self.callback.finish_time, datetime)
        self.assertIsInstance(self.callback.agent_duration, timedelta)

    def test_on_agent_final_answer(self):
        final_answer = {
            "query": "test_query",
            "answers": [Answer(answer="test_answer")],
            "transcript": "test_transcript",
        }
        self.callback.dataset.add_records = MagicMock()
        self.callback.on_agent_final_answer(final_answer)
        call_args = self.callback.dataset.add_records.call_args
        records_arg = call_args[1]["records"][0]
        assert records_arg.fields["prompt"] == "test_query"
        assert records_arg.fields["response"] == "test_answer"
        assert isinstance(records_arg.fields.get("time-details"), str)
        assert records_arg.metadata == {"type": "extractive"}

    def test_on_tool_start(self):
        mock_tool = MagicMock()
        self.callback.on_tool_start(tool_input="test_tool_input", tool=mock_tool)
        self.assertIsInstance(self.callback.tool_start_time, datetime)
        expected_key = mock_tool.name
        self.assertDictEqual(
            self.callback.tool_start_times,
            {expected_key: self.callback.tool_start_time},
        )

    def test_on_tool_finish(self):
        self.callback.on_tool_finish(
            tool_result="test_tool_result",
            tool_name="test_tool_name",
            tool_input="test_tool_input",
        )
        self.assertEqual(self.callback.tool_durations, {})

    def test_on_tool_error(self):
        self.callback.on_tool_error(exception=Exception(), tool=MagicMock())


if __name__ == "__main__":
    unittest.main()


@pytest.fixture
def argilla_callback():
    prompt_node = MagicMock()
    agent = Agent(prompt_node=prompt_node)
    dataset_name = "test_dataset"
    api_url = "http://localhost:6900/"
    api_key = "argilla.apikey"
    questions = ["test_question"]
    guidelines = "test_guideline"
    return ArgillaCallbackHandler(
        agent=agent,
        dataset_name=dataset_name,
        api_url=api_url,
        api_key=api_key,
        questions=questions,
        guidelines=guidelines,
    )
