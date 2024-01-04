#TODO: add __init__.py file

import warnings
from typing import Any, Dict, List, Optional

from haystack.agents import Agent
from haystack.agents.agent_step import AgentStep

print("Importing base")

class ArgillaCallback():
    """base"""


    def __init__(
            self,
            agent: Agent,
            dataset_name: str,
            workspace_name: Optional[str] = None,
            api_url: Optional[str] = None,
            api_key: Optional[str] = None,
            ) -> None:
        
        agent.callback_manager.on_agent_start += self.on_agent_start
        agent.callback_manager.on_agent_final_answer += self.on_agent_final_answer
        agent.callback_manager.on_agent_finish += self.on_agent_finish

        # Import Argilla modules
        try:
            import argilla as rg
        except:
            raise ImportError("Please install argilla with: pip install argilla")
            ## TODO: Add other warnings/errors
        
        ## TODO: Add checks for SDK and Server versions
        ## TODO: Add checks to confirm ApiKey and ApiUrl are valid
        
        # Connect to Argilla with the given credentials
        try:
            rg.init(
                api_key=api_key,
                api_url=api_url,
            )
        except:
            raise ValueError("Please provide valid credentials for Argilla")
        
        # Get dataset

        ## TODO: Add checks to confirm dataset exists
        ## TODO: Add checks to confirm the question types are appropriate for the task
        ## TODO: Add checks to confirm the field types are appropriate for the task
        ## TODO: Add checks if dataset is empty
        ## TODO: Add checks if records have responses and/or are submitted
        ## TODO: Add checks if there are suggestions
        ## TODO: Add checks for workspace validity

        self.dataset = rg.FeedbackDataset.from_argilla(
            name=dataset_name,
            workspace=workspace_name,
        )

        #TODO: get corresponding field and question names from dataset

    def on_agent_start(self, name: str, query: str, params: Dict[str, Any]):
        pass

    def on_agent_step(self, agent_step: AgentStep):
        pass

    def on_agent_finish(self, agent_step: AgentStep):
        print("Records have been updated to Argilla")
        
    
    def on_agent_final_answer(self, final_answer):
        query = final_answer["query"]
        answer = final_answer["answers"][0].answer
        self.dataset.add_records(
        records=[
            {
                "fields": {
                    "query": query, 
                    "retrieved_document_1": answer
                    },
                }
            ]
        )
        return query, answer

    def on_tool_start(self):
        pass

    def on_tool_finish(self):
        pass

    def on_tool_error(self):
        pass
        