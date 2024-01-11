import unittest
from unittest.mock import MagicMock
from haystack.agents.base import Agent, ToolsManager
from haystack.schema import Answer
from argilla_haystack import ArgillaCallback
import pytest

class TestArgillaCallback(unittest.TestCase):
    def setUp(self):
        self.prompt_node = MagicMock()  # Create a mock prompt_node object
        self.agent = Agent(prompt_node=self.prompt_node)
        self.dataset_name = "test_dataset"
        self.api_url = "http://localhost:6900/"
        self.api_key = "argilla.apikey"
        self.callback = ArgillaCallback(
            agent=self.agent,
            dataset_name=self.dataset_name,
            api_url=self.api_url,
            api_key=self.api_key
        )

    def test_on_agent_start(self):
        # Test that on_agent_start does nothing
        self.callback.on_agent_start("test_name", "test_query", {"param": "value"})

    def test_on_agent_step(self):
        # Test that on_agent_step updates the metadata
        agent_step = MagicMock()
        agent_step.prompt_node_response = "test_tool_output"
        self.callback.on_agent_step(agent_step)
        self.assertEqual(self.callback.metadata["tool_output"], "test_tool_output")

    def test_on_agent_finish(self):
        # Test that on_agent_finish does nothing
        self.callback.on_agent_finish(None)

    def test_on_agent_final_answer(self):
        # Test that on_agent_final_answer adds the final answer and query to the dataset
        final_answer = {
            "query": "test_query",
            "answers": [Answer(answer="test_answer")]
        }
        self.callback.dataset.add_records = MagicMock()
        self.callback.on_agent_final_answer(final_answer)
        self.callback.dataset.add_records.assert_called_once_with(records=[{
            "fields": {"prompt": "test_query", "response": "test_answer"},
            "metadata": self.callback.metadata
        }])

    def test_on_tool_start(self):
        # Test that on_tool_start does nothing
        self.callback.on_tool_start("test_tool_input", MagicMock())

    def test_on_tool_finish(self):
        # Test that on_tool_finish updates the metadata with the tool name
        self.callback.on_tool_finish("test_tool_result", "test_tool_name")
        self.assertEqual(self.callback.metadata["tool_name"], "test_tool_name")

    def test_on_tool_error(self):
        # Test that on_tool_error does nothing
        self.callback.on_tool_error(Exception(), MagicMock())

if __name__ == "__main__":
    unittest.main()

@pytest.fixture
def argilla_callback():
    prompt_node = MagicMock()
    agent = Agent(prompt_node=prompt_node)
    dataset_name = "test_dataset"
    api_url = "http://localhost:6900/"
    api_key = "argilla.apikey"
    return ArgillaCallback(agent=agent, dataset_name=dataset_name, api_url=api_url, api_key=api_key)
