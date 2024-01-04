#TODO: add __init__.py file

import warnings
from typing import Any, Dict, List, Optional
import os

from haystack.agents import Agent, Tool
from haystack.agents.agent_step import AgentStep
from packaging.version import parse

print("Importing base")

class ArgillaCallback():
    """base"""

    DEFAULT_API_URL: str = "http://localhost:6900"
    DEFAULT_API_KEY: str = "argilla.apikey"

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
                    f" set, it will default to `{self.DEFAULT_API_URL}`, which is the"
                    " default API URL in Argilla Quickstart."
                ),
            )
            api_url = self.DEFAULT_API_URL

        if api_key is None and os.getenv("ARGILLA_API_KEY") is None:
            warnings.warn(
                (
                    "Since `api_key` is None, and the env var `ARGILLA_API_KEY` is not"
                    f" set, it will default to `{self.DEFAULT_API_KEY}`, which is the"
                    " default API key in Argilla Quickstart."
                ),
            )
            api_key = self.DEFAULT_API_KEY

        ## TODO: Add checks for Server versions
    
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
            self.dataset = rg.FeedbackDataset.from_argilla(
                name=self.dataset_name,
                workspace=self.workspace_name,
            )
            ## TODO: Should it be with or without records (as in Langchain integration)?
        except Exception as e:
            raise FileNotFoundError(
                f"`FeedbackDataset` retrieval from Argilla failed with exception `{e}`."
                f"\nPlease check that the dataset with name={self.dataset_name} in the"
                f" workspace={self.workspace_name} exists in advance. If the problem persists"
                f" please report it to {self.ISSUES_URL} as an `integration` issue."
            ) from e
        
        self.field_names = [field.name for field in self.dataset.fields]
        self.metadata = {}

        ## TODO: Add checks to confirm dataset exists
        ## TODO: Add checks to confirm the question types are appropriate for the task
        ## TODO: Add checks to confirm the field types are appropriate for the task
        ## TODO: Add checks if dataset is empty
        ## TODO: Add checks if records have responses and/or are submitted
        ## TODO: Add checks if there are suggestions
        ## TODO: Add checks for workspace validity

    def on_agent_start(self, name: str, query: str, params: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing when the agent starts"""
        pass

    def on_agent_step(self, agent_step: AgentStep, **kwargs: Any) -> None:
        """Update the metadata for the record with the tool output at agent step"""
        self.metadata["tool_output"] = agent_step.prompt_node_response
    
    def on_agent_finish(self, agent_step: AgentStep, **kwargs: Any) -> None:
        """Print message when the agent finishes"""
        print("Records have been updated to Argilla")
    
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

    def on_tool_start(self, tool_input: str, tool: Tool):
        """Do nothing when the tool starts"""
        pass

    def on_tool_finish(self, tool_result: str, tool_name: Optional[str] = None, tool_input: Optional[str] = None, **kwargs: Any) -> None:
        """Update the metadata with the tool name"""
        self.metadata["tool_name"] = tool_name

    def on_tool_error(self, exception: Exception, tool: Tool, **kwargs: Any) -> None:
        """Do nothing when the tool errors out"""
        pass
        