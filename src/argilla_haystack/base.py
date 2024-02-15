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

import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import argilla as rg
from argilla._constants import DEFAULT_API_KEY, DEFAULT_API_URL
from haystack.agents import Agent, Tool
from haystack.agents.agent_step import AgentStep
from packaging.version import parse

from argilla_haystack.helpers import (
    create_svg_with_durations,
    create_tree_with_durations,
)

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)


class ArgillaCallbackHandler:
    """Callback manager that logs into Argilla

    Args:
        agent: The Haystack Agent that will be used to run the pipeline
        dataset_name: The name of the dataset in Argilla. If the dataset does not exist,
            a new one will be created.
        workspace_name: The name of the workspace in Argilla. Defaults to 'None', in which
            case the default workspace will be used.
        api_url: The URL of the Argilla API. Defaults to 'None', in which case the default
            API URL will be used.
        api_key: The API key of the Argilla API. Defaults to 'None', in which case the default
            API key will be used.

    Raises:
        ImportError: If the `argilla` Python package is not installed or the one installed is not compatible
        ConnectionError: If the connection to Argilla fails
        FileNotFoundError: If the retrieval and creation of the `FeedbackDataset` fails

    Example:
        >>> from haystack.nodes import PromptNode
        >>> from haystack.agents.memory import ConversationSummaryMemory
        >>> from haystack.agents.conversational import ConversationalAgent
        >>> prompt_node = PromptNode(
                model_name_or_path="gpt-3.5-turbo-instruct", api_key=openai_api_key, max_length=256, stop_words=["Human"]
            )
        >>> summary_memory = ConversationSummaryMemory(prompt_node)
        >>> conversational_agent = ConversationalAgent(prompt_node=prompt_node, memory=summary_memory)
        >>> ArgillaCallback(agent=conversational_agent, dataset_name="conversational_ai", api_url="http://localhost:6900/", api_key=argilla_api_key)
        >>> conversational_agent.run(query="What is another name of Artemis?")
        "Diana"
    """

    REPO_URL: str = "https://github.com/argilla-io/argilla"
    ISSUES_URL: str = f"{REPO_URL}/issues"

    def __init__(
        self,
        agent: Agent,
        dataset_name: str,
        workspace_name: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        questions: Optional[List[Any]] = None,
        guidelines: Optional[str] = None,
    ) -> None:
        """Initialize the ArgillaCallback

        Args:
            agent: The Haystack Agent that will be used to run the pipeline
            dataset_name: The name of the dataset in Argilla. If the dataset does not exist,
                a new one will be created.
            workspace_name: The name of the workspace in Argilla. Defaults to 'None', in which
                case the default workspace will be used.
            api_url: The URL of the Argilla API. Defaults to 'None', in which case the default
                API URL will be used.
            api_key: The API key of the Argilla API. Defaults to 'None', in which case the default
                API key will be used.

        Raises:
            ImportError: If the `argilla` Python package is not installed or the one installed is not compatible
            ConnectionError: If the connection to Argilla fails
            FileNotFoundError: If the retrieval and creation of the `FeedbackDataset` fails
        """
        self._setup_callbacks(agent)
        self.tool_names = agent.tm.get_tool_names().split(", ")

        self.ARGILLA_VERSION = rg.__version__
        self._validate_argilla_version()
        self.api_key = api_key or os.getenv("ARGILLA_API_KEY", DEFAULT_API_KEY)
        self.api_url = api_url or os.getenv("ARGILLA_API_URL", DEFAULT_API_URL)
        self._init_argilla()
        self.dataset_name = dataset_name
        self.workspace_name = workspace_name or rg.get_workspace()
        self.questions = questions
        self.guidelines = guidelines
        self._prepare_dataset()
        self.metadata = {}

        self.start_time = None
        self.finish_time = None
        self.agent_duration = timedelta(0)
        self.tool_start_times = {}
        self.tool_durations = {}

    def _setup_callbacks(self, agent: Agent):
        """Configure the agent to use the callback manager's methods"""
        agent.callback_manager.on_agent_start += self.on_agent_start
        agent.callback_manager.on_agent_step += self.on_agent_step
        agent.callback_manager.on_agent_final_answer += self.on_agent_final_answer
        agent.callback_manager.on_agent_finish += self.on_agent_finish

        agent.tm.callback_manager.on_tool_start += self.on_tool_start
        agent.tm.callback_manager.on_tool_finish += self.on_tool_finish
        agent.tm.callback_manager.on_tool_error += self.on_tool_error

    def _validate_argilla_version(self):
        if parse(self.ARGILLA_VERSION) < parse("1.18.0"):
            raise ImportError(
                f"The installed `argilla` version is {self.ARGILLA_VERSION} but "
                "`ArgillaCallbackHandler` requires at least version 1.18.0. Please "
                "upgrade `argilla` with `pip install --upgrade argilla`."
            )

    def _init_argilla(self):
        if self.api_key == DEFAULT_API_KEY:
            warnings.warn(
                "Using default api_key='argilla.apikey'. Set `api_key` or `ARGILLA_API_KEY` to override.",
                stacklevel=2,
            )

        if self.api_url == DEFAULT_API_URL:
            warnings.warn(
                "Using default api_url='http://localhost:6900'. Set `api_url` or `ARGILLA_API_URL` to override.",
                stacklevel=2,
            )

        try:
            rg.init(api_key=self.api_key, api_url=self.api_url)
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Argilla: '{e}'. Check `api_key` and `api_url` and ensure Argilla server is running. "
                f"Report persistent issues to {self.ISSUES_URL} as an `integration` issue."
            ) from e

    def _prepare_dataset(self):
        try:
            if self.dataset_name in [ds.name for ds in rg.FeedbackDataset.list()]:
                self.dataset = rg.FeedbackDataset.from_argilla(
                    name=self.dataset_name,
                    workspace=self.workspace_name,
                )
                if self.tool_names != [""]:
                    supported_fields = [
                        "prompt",
                        "response",
                        "transcript",
                        "time-details",
                    ]
                else:
                    supported_fields = ["prompt", "response", "time-details"]
                if supported_fields != [field.name for field in self.dataset.fields]:
                    raise ValueError(
                        f"`FeedbackDataset` with name={self.dataset_name} in the workspace="
                        f"{self.workspace_name} had fields that are not supported for the"
                        f"`haystack` integration. Supported fields are: {supported_fields}."
                        f" But the current `FeedbackDataset` fields are {[field.name for field in self.dataset.fields]}."
                    )
                _LOGGER.info(
                    f"`FeedbackDataset` with name={self.dataset_name} in the workspace="
                    f"{self.workspace_name} was correctly retrieved from Argilla. The current fields are"
                    f"{[field.name for field in self.dataset.fields]}. The current questions are {[question.name for question in self.dataset.questions]}.",
                )
                if self.questions or self.guidelines:
                    warnings.warn(
                        f"Questions and guidelines are not updated for the existing `FeedbackDataset` with name={self.dataset_name} in the workspace="
                        f"{self.workspace_name}. Create a new `FeedbackDataset` if you want to use the provided questions and guidelines.",
                        UserWarning,
                        stacklevel=2,
                    )

            else:
                fields = [
                    rg.TextField(name="prompt"),
                    rg.TextField(name="response"),
                    rg.TextField(
                        name="time-details", title="Time Details", use_markdown=True
                    ),
                ]
                if self.tool_names != [""]:
                    fields.insert(2, rg.TextField(name="transcript"))
                if self.questions is None:
                    self.questions = [
                        rg.RatingQuestion(
                            name="response-rating",
                            title="How would you rate the quality of the response?",
                            description="Rate the quality of the response on a scale of 1-7.",
                            values=[1, 2, 3, 4, 5, 6, 7],
                            required=True,
                        ),
                        rg.TextQuestion(
                            name="response-feedback",
                            title="Provide your feedback for the response.",
                            description="Provide feedback for the response.",
                            required=False,
                        ),
                    ]
                if self.guidelines is None:
                    self.guidelines = "You're asked to rate the quality of the response and provide feedback."

                dataset = rg.FeedbackDataset(
                    fields=fields,
                    questions=self.questions,
                    guidelines=self.guidelines,
                    allow_extra_metadata=True,
                )
                if self.tool_names != [""]:
                    dataset.add_metadata_property(
                        rg.TermsMetadataProperty(
                            name="tool_name", title="Tool Name", values=self.tool_names
                        )
                    )
                dataset.add_metadata_property(
                    rg.TermsMetadataProperty(name="type", title="Type")
                )
                self.dataset = dataset.push_to_argilla(self.dataset_name)
        except Exception as e:
            raise FileNotFoundError(
                f"`FeedbackDataset` retrieval and creation both failed with exception `{e}`."
                f" If the problem persists please report it to {self.ISSUES_URL} "
                f"as an `integration` issue."
            ) from e

    def on_agent_start(
        self, name: str, query: str, params: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Save the starting time of the agent and reset the dictionaries for each query"""
        self.start_time = datetime.now()
        self.tool_start_times = {}
        self.tool_durations = {}
        self.metadata = {}

    def on_agent_step(self, agent_step: AgentStep, **kwargs: Any) -> None:
        """Do nothing when the agent steps"""
        pass

    def on_agent_finish(self, agent_step: AgentStep, **kwargs: Any) -> None:
        """Save the finish time of the agent and calculate the agent duration"""
        self.finish_time = datetime.now()
        if self.start_time:
            self.agent_duration = self.finish_time - self.start_time

    def on_agent_final_answer(self, final_answer, **kwargs: Any) -> None:
        """Add the query, final answer, transcript and metadata to the record and submit it to Argilla"""
        query = final_answer["query"]
        answer = final_answer["answers"][0].answer
        transcript = final_answer["transcript"]
        time_data = create_tree_with_durations(
            str(round(self.agent_duration.total_seconds(), 3)), self.tool_durations
        )
        time_svg = create_svg_with_durations(time_data)
        self.metadata["type"] = final_answer["answers"][0].type

        add_fields = {
            "prompt": query,
            "response": answer,
            "time-details": time_svg,
        }
        if len(self.dataset.fields) == 4:
            add_fields["transcript"] = transcript
            print("transcript added")

        self.dataset.add_records(
            records=[
                rg.FeedbackRecord(
                    fields=add_fields,
                    metadata=self.metadata,
                )
            ]
        )
        _LOGGER.info("Records have been updated to Argilla")

    def on_tool_start(self, tool_input: str, tool: Tool):
        """Save the starting time of the tool"""
        self.tool_start_time = datetime.now()
        self.tool_start_times[tool.name] = datetime.now()

    def on_tool_finish(
        self,
        tool_result: str,
        tool_name: Optional[str] = None,
        tool_input: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Update the metadata with the tool name"""
        if self.metadata.get("tool_name"):
            self.metadata["tool_name"].append(tool_name)
        else:
            self.metadata["tool_name"] = [tool_name]

        if tool_name and tool_name in self.tool_start_times:
            start_time = self.tool_start_times[tool_name]
            finish_time = datetime.now()
            duration = finish_time - start_time
            self.tool_durations[tool_name] = str(round(duration.total_seconds(), 3))

    def on_tool_error(self, exception: Exception, tool: Tool, **kwargs: Any) -> None:
        """Do nothing when the tool errors out"""
        pass
