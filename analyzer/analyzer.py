from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from tools.tools import Tools
from States.states import SharedState


sys_msg = """
You are a supervisor agent coordinating the following agents: 
Query Analyzer, Web Search, Fact Checker, Report Generator, and Quality Controller. 
For each user query, invoke these agents in sequence, passing the state and collecting their outputs. 
Only return the final report after all agents have completed their tasks.
"""


class Analyzer:
    def __call__(self, input: str) -> dict:

        state: SharedState = {
            "question": input,
            "messages": [],
            "remaining_steps": 5
        }

        response = self.supervisor.invoke(state)
        print("Final State: ", response)

        return {"output": response.get("report", "No report generated.")}

    def __init__(self):
        self.model = init_chat_model(
            "mistral-tiny",
            temperature=0.0,
            max_tokens=1000
        )

        self.tools_instance = Tools(self.model)

        self.query_analyzer_assistant = create_react_agent(
            model=self.model,
            tools=[self.tools_instance.query_analyzer_tool],
            prompt="You are a Query Analyzer Agent. Classify the user's question into one of the following types: factual, comparative, technical, analytical, or exploratory. Provide a concise classification.",
            name="Query Analyzer Agent",
            state_schema=SharedState
        )

        self.web_search_assistant = create_react_agent(
            model=self.model,
            tools=[self.tools_instance.web_search_tool],
            prompt="You are a Web Search Agent. Based on the query type provided, search the web for relevant and up-to-date information. Provide a summary of your findings.",
            name="Web Search Agent",
            state_schema=SharedState
        )

        self.fact_checker_assistant = create_react_agent(
            model=self.model,
            tools=[self.tools_instance.fact_checker_tool],
            prompt="You are a Fact Checker Agent. Cross-verify claims from multiple sources to confirm their reliability.",
            name="Fact Checker Agent",
            state_schema=SharedState
        )

        self.report_generator_assistant = create_react_agent(
            model=self.model,
            tools=[self.tools_instance.report_generator_tool],
            prompt="You are a Report Generator Agent. Synthesize validated findings into a coherent, clear answer.",
            name="Report Generator Agent",
            state_schema=SharedState
        )

        self.quality_control_assistant = create_react_agent(
            model=self.model, 
            tools=[self.tools_instance.quality_control_tool],
            prompt="You are a Quality Controller Agent. Review the draft report for completeness and clarity.",
            name="Quality Controller Agent",
            state_schema=SharedState
        )

        self.supervisor = create_supervisor(
            agents=[self.query_analyzer_assistant, 
                    self.web_search_assistant, 
                    self.fact_checker_assistant, 
                    self.report_generator_assistant, 
                    self.quality_control_assistant],
            parallel_tool_calls=False,
            model=self.model,
            prompt=sys_msg,
            state_schema=SharedState
        ).compile()
