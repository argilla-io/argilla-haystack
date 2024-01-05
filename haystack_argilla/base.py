#TODO: add __init__.py file
#TODO: Review requirements.txt
import warnings
from typing import Any, Dict, List, Optional
import os
from argilla._constants import DEFAULT_API_KEY, DEFAULT_API_URL
import logging

from haystack.agents import Agent, Tool
from haystack.agents.agent_step import AgentStep
from packaging.version import parse

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)

class ArgillaCallback():
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
    
    Examples:                                                 #TODO: Correct this example if needed
        >>> from argilla_haystack.base import ArgillaCallback
        >>> from haystack.agents.base import Agent, ToolsManager
        >>> ... # Other imports
        >>> ... # Define your Haystack pipelines and nodes for agent
        >>> conversational_agent = Agent(
                agent_prompt_node,
                prompt_template=agent_prompt,
                prompt_parameters_resolver=resolver_function,
                memory=memory,
                tools_manager=ToolsManager([search_tool]),
            )
        >>> ArgillaCallback(agent=conversational_agent, dataset_name=dataset_id, api_url=api_url, api_key=api_key)
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
        
        # Add event handlers to Agent and ToolsManager
        agent.callback_manager.on_agent_start += self.on_agent_start
        agent.callback_manager.on_agent_step += self.on_agent_step
        agent.callback_manager.on_agent_final_answer += self.on_agent_final_answer
        agent.callback_manager.on_agent_finish += self.on_agent_finish
        
        agent.tm.callback_manager.on_tool_start += self.on_tool_start
        agent.tm.callback_manager.on_tool_finish += self.on_tool_finish
        agent.tm.callback_manager.on_tool_error += self.on_tool_error

        # Import Argilla 
        try:
            import argilla as rg
            self.ARGILLA_VERSION = rg.__version__

        except ImportError:
            raise ImportError(
                "To use the Argilla callback manager you need to have the `argilla` "
                "Python package installed. Please install it with `pip install argilla`"
            )

        ## Check whether the Argilla version is compatible
        if parse(self.ARGILLA_VERSION) < parse("1.20.0"):
            raise ImportError(
                f"The installed `argilla` version is {self.ARGILLA_VERSION} but "
                "`ArgillaCallbackHandler` requires at least version 1.20.0. Please "
                "upgrade `argilla` with `pip install --upgrade argilla`."
            )
        
        # API_URL and API_KEY
        # Show a warning message if Argilla will assume the default values will be used
        if api_url is None and os.getenv("ARGILLA_API_URL") is None:
            warnings.warn(
                (
                    "Since `api_url` is None, and the env var `ARGILLA_API_URL` is not"
                    f" set, it will default to `{DEFAULT_API_URL}`, which is the"
                    " default API URL in Argilla Quickstart."
                ),
            )
            api_url = DEFAULT_API_URL

        if api_key is None and os.getenv("ARGILLA_API_KEY") is None:
            warnings.warn(
                (
                    "Since `api_key` is None, and the env var `ARGILLA_API_KEY` is not"
                    f" set, it will default to `{DEFAULT_API_KEY}`, which is the"
                    " default API key in Argilla Quickstart."
                ),
            )
            api_key = DEFAULT_API_KEY
    
        # Connect to Argilla with the provided credentials, if applicable
        try:
            rg.init(
                api_key=api_key, 
                api_url=api_url
            )
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Argilla with exception: '{e}'.\n"
                "Please check your `api_key` and `api_url`, and make sure that "
                "the Argilla server is up and running. If the problem persists "
                f"please report it to {self.ISSUES_URL} as an `integration` issue."
            ) from e
        
        # Set the Argilla variables
        self.dataset_name = dataset_name
        self.workspace_name = workspace_name or rg.get_workspace()
        
        # Retrieve the `FeedbackDataset` from Argilla
        try:
            if self.dataset_name in [ds.name for ds in rg.FeedbackDataset.list()]:
                self.dataset = rg.FeedbackDataset.from_argilla(
                    name=self.dataset_name,
                    workspace=self.workspace_name,
                )
            # If the dataset does not exist, create a new one with the given name
            else:
                dataset = rg.FeedbackDataset(
                    fields=[
                        rg.TextField(name="prompt"),
                        rg.TextField(name="response"),
                    ],
                    questions=[rg.RatingQuestion(name="rating", values=[1, 2, 3, 4, 5])
                    ],
                )
                self.dataset = dataset.push_to_argilla(self.dataset_name)
                warnings.warn(
                (
                    f"No dataset with the name {self.dataset_name} was found in workspace "
                    f"{self.workspace_name}. A new dataset with the name {self.dataset_name} "
                    "has been created with the question fields `prompt` and `response`"
                    "and the rating question `rating` with values 1-5."
                ),
            )
            ## TODO: Should it be with or without records (as in Langchain integration)?
        except Exception as e:
            raise FileNotFoundError(
                f"`FeedbackDataset` retrieval and creation both failed with exception `{e}`."
                f" If the problem persists please report it to {self.ISSUES_URL} "
                f"as an `integration` issue."
            ) from e
        
        self.field_names = [field.name for field in self.dataset.fields]
        self.metadata = {}

    def on_agent_start(self, name: str, query: str, params: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing when the agent starts"""
        pass

    def on_agent_step(self, agent_step: AgentStep, **kwargs: Any) -> None:
        """Update the metadata for the record with the tool output at agent step"""
        self.metadata["tool_output"] = agent_step.prompt_node_response
    
    def on_agent_finish(self, agent_step: AgentStep, **kwargs: Any) -> None:
        """Do nothing when the agent finishes"""
    
    def on_agent_final_answer(self, final_answer, **kwargs: Any) -> None:
        """Add the final answer and the query to the record and submit it to Argilla"""
        query = final_answer["query"]
        answer = final_answer["answers"][0].answer
        self.dataset.add_records(
            records=[
                {
                    "fields": {
                        self.field_names[0]: query, 
                        self.field_names[1]: answer
                        },
                    "metadata": self.metadata
                    },
                ]
            )
        _LOGGER.info("Records have been updated to Argilla")

    def on_tool_start(self, tool_input: str, tool: Tool):
        """Do nothing when the tool starts"""
        pass

    def on_tool_finish(self, tool_result: str, tool_name: Optional[str] = None, tool_input: Optional[str] = None, **kwargs: Any) -> None:
        """Update the metadata with the tool name"""
        self.metadata["tool_name"] = tool_name

    def on_tool_error(self, exception: Exception, tool: Tool, **kwargs: Any) -> None:
        """Do nothing when the tool errors out"""
        pass
