# Haystack-Argilla

With the integration between Argilla and Haystack, monitoring your haystack agent is quite easy. You can use `ArgillaCallback` to send your haystack workflows to Argilla. This file will guide you through the process of setting up your haystack agent and using `ArgillaCallback`.

## Import Necessary Libraries

```python
from haystack_argilla.base import ArgillaCallback
import argilla as rg
from datasets import load_dataset

from haystack.agents import Tool
from haystack.agents.memory import ConversationSummaryMemory
from haystack.agents import Agent
from haystack.agents.base import Agent, ToolsManager
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, BM25Retriever
from haystack.pipelines import Pipeline
```

## Insert your OPENAI API key

As we will use OpenAI to generate answers, we will need a valid API key.
```python
openai_api_key = ...
```

## Create a Haystack Agent

Let us create a basic haystack agent. This agent below is taken from a [haystack tutorial](https://haystack.deepset.ai/tutorials/25_customizing_agent). We will use this agent to demonstrate how to use `ArgillaCallback`.

```python
# based on tutorial https://haystack.deepset.ai/tutorials/25_customizing_agent

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")

## CREATE GENERATIVE PIPELINE
document_store = InMemoryDocumentStore(use_bm25=True)
document_store.write_documents(dataset)
retriever = BM25Retriever(document_store=document_store)
prompt0 = "Please use a maximum of 50 tokens. Question: {query}\nDocuments: {join(documents)}\nAnswer:"
prompt_template = PromptTemplate(
    prompt=prompt0,
    output_parser=AnswerParser(),
)
prompt_node = PromptNode(
    model_name_or_path="gpt-3.5-turbo-instruct", api_key=openai_api_key, default_prompt_template=prompt_template
)
generative_pipeline = Pipeline()
generative_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
generative_pipeline.add_node(component=prompt_node, name="Prompt", inputs=["Retriever"])

## CREATE AGENT
search_tool = Tool(
    name="Search_the_documents_tool",
    pipeline_or_node=generative_pipeline,
    description="useful for when you need to answer questions about the seven wonders of the world",
    output_variable="answers",
)

agent_prompt_node = PromptNode(
    "gpt-3.5-turbo",
    api_key=openai_api_key,
    max_length=256,
    stop_words=["Observation:"],
    model_kwargs={"temperature": 0.5},
)
memory_prompt_node = PromptNode(
    model_name_or_path="philschmid/bart-large-cnn-samsum", max_length=256, model_kwargs={"task_name":"text2text-generation"}  # MODEL
)
memory = ConversationSummaryMemory(
    prompt_node=memory_prompt_node, prompt_template="{chat_transcript}"
)
agent_prompt = """
In the following conversation, a human user interacts with an AI Agent. The human user poses questions, and the AI Agent goes through several steps to provide well-informed answers.
The AI Agent must use the available tools to find the up-to-date information. The final answer to the question should be truthfully based solely on the output of the tools. The AI Agent should ignore its knowledge when answering the questions.
The AI Agent has access to these tools:
{tool_names_with_descriptions}

The following is the previous conversation between a human and The AI Agent:
{memory}

AI Agent responses must start with one of the following:

Thought: [the AI Agent's reasoning process]
Tool: [tool names] (on a new line) Tool Input: [input as a question for the selected tool WITHOUT quotation marks and on a new line] (These must always be provided together and on separate lines.)
Observation: [tool's result]
Final Answer: [final answer to the human user's question]

When selecting a tool, the AI Agent must provide both the "Tool:" and "Tool Input:" pair in the same response, but on separate lines.

The AI Agent should not ask the human user for additional information, clarification, or context.
If the AI Agent cannot find a specific answer, it should accept the first word of the answer it found as the answer to the query.

Question: {query}
Thought:
{transcript}
"""

def resolver_function(query, agent, agent_step):
    return {
        "query": query,
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
        "transcript": agent_step.transcript,
        "memory": agent.memory.load(),
    }
```

Now with everything set up, we can create our agent.

```python
conversational_agent = Agent(
    agent_prompt_node,
    prompt_template=agent_prompt,
    prompt_parameters_resolver=resolver_function,
    memory=memory,
    tools_manager=ToolsManager([search_tool]),
)
```

## Run ArgillaCallback

Now that we have a complete agent set up, we can run `ArgillaCallback`. The code below will add the event handlers we have to the agent. You can change the arguments below according to what you are working on.
    
```python
api_key = "argilla.apikey"
api_url = "http://localhost:6900/"
dataset_id = "haystack_argilla"

ArgillaCallback(agent=conversational_agent, dataset_name=dataset_id, api_url=api_url, api_key=api_key)
```

Let us run the agent with a sample query.

```python
conversational_agent.run(query="What is another name for Artemis?")
```

This will run the agent. As we defined above, our agent will search for the files we have and respond with the final answer. Then, our callback handler will add the data to the dataset on the Argilla server with prompt and response. Also, the tool name used and the tool output will be added to the dataset as metadata.