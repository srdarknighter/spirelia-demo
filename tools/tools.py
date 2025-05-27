import requests
import logging
from langgraph.types import Command
from States.states import SharedState

class Tools:
    def __init__(self, model):
        self.model = model


    def query_analyzer_tool(self, state: SharedState) -> SharedState:
        """Classifies the user's question into types such as factual, comparative, technical, analytical, or exploratory."""

        query = state.get("question", "")
        classification_prompt = f"""Classify the user's question into one of the following types: factual, comparative, technical, analytical, or exploratory. 
        Provide a concise classification.

        User Question: {query}

        Classification:
     """

        message = [ 
            {"role": "system", "content": "You are a query analyzer expert"},
            {"role": "user", "content": classification_prompt}
        ]

        response = self.model.invoke(message)
        classification = response.content
        logging.info(f"Classification result: {classification}")
        if "remaining_steps" in state:
            state["remaining_steps"] = max(0, state["remaining_steps"] - 1)
        
        state["classification"] = classification
        return state

    def web_search_tool(self, state: SharedState) -> SharedState:

        """Searches the web for relevant and up-to-date information based on the user's query."""

        url = "https://api.tavily.com/search"

        payload = {
            "query": state.get("question", ""),
            "topic": "general",
            "search_depth": "basic",
            "chunks_per_source": 3,
            "max_results": 3,
            "time_range": None,
            "days": 7,
            "include_answer": "advanced",
            "include_raw_content": False,
            "include_images": False,
            "include_image_descriptions": False,
            "include_domains": [],
            "exclude_domains": []
        }
        
        headers = {
            "Authorization": "Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        response_json = response.json()

        answer = response_json.get("answer", "")

        contents = [item.get("content", "") for item in response_json.get("results", [])]
        contents_str = "\n".join(contents)

        memory = {
            "answer": answer,
            "contents": contents_str[:1200]
        }
        
        logging.info(f"Search results: {memory}")
        if "remaining_steps" in state:
            state["remaining_steps"] = max(0, state["remaining_steps"] - 1)
        
        state["search_results"] = memory
        return state
    

    def fact_checker_tool(self, state: SharedState) -> SharedState:
        """Cross-verifies claims from multiple sources to confirm their reliability."""

        query = state.get("question", "")
        search_results = state.get("search_results", {})
        answer = search_results.get("answer", "")
        contents = search_results.get("contents", "")

        claims_text = f"Answer:\n{answer}\n\nSupporting Contents:\n{contents}"

        fact_check_prompt = f"""Cross-verify claims from multiple sources to confirm their reliability for the following query.
        You just need to analyze the supporting claims and check if it correlates with the answer provided. If the claims are verified, return them as is. If not, provide a concise summary of the claims.
        You can add studies, statistics or other relevant information on top of the answer using the claims to make it more reliable.
        The following information is provided:

        User Query: {query}

        Claims: {claims_text}

        Verified Claims:
        """
        message = [
            {"role": "system", "content": "You are a fact-checking expert"},
            {"role": "user", "content": fact_check_prompt}
        ]

        response = self.model.invoke(message)
        verified_claims = response.content
        logging.info(f"Verified claims: {verified_claims}")

        if "remaining_steps" in state:
            state["remaining_steps"] = max(0, state["remaining_steps"] - 1)
        state["verified_claims"] = verified_claims
        return state

    def report_generator_tool(self, state: SharedState) -> SharedState:

        """Synthesizes validated findings into a coherent, clear answer for the user's query."""

        query = state.get("question", "")
        verified_claims = state.get("verified_claims", "")

        report_prompt = f"""Synthesize validated findings into a coherent, clear answer for the following query:

        User Query: {query}
        Verified Claims: {verified_claims}
        Report:
        """
        message = [
            {"role": "system", "content": "You are a report generation expert"},
            {"role": "user", "content": report_prompt}
        ]
        response = self.model.invoke(message)
        report = response.content
        logging.info(f"Generated report: {report}")
        if "remaining_steps" in state:
            state["remaining_steps"] = max(0, state["remaining_steps"] - 1)
        
        state["report"] = report

        return state

    def quality_control_tool(self, state: SharedState) -> Command:
        """Reviews the draft report for completeness and clarity, providing feedback on whether to accept or reject it."""

        quality_control_prompt = f"""Review the draft report for completeness and clarity. Answer whether to accept or reject in one word. The report is based on the following:

        Classification: {state.get("classification", "")}
        User Query: {state.get("question", "")}
        Draft Report: {state.get("report", "")}

        Quality Control Feedback:
        """
        message = [
            {"role": "system", "content": "You are a quality control expert"},
            {"role": "user", "content": quality_control_prompt}
        ]
        response = self.model.invoke(message)
        quality_feedback = response.content
        logging.info(f"Quality control feedback: {quality_feedback}")
        state["quality_feedback"] = quality_feedback

        if quality_feedback.lower() == "reject":
            if "remaining_steps" in state:
                state["remaining_steps"] = 4
            return Command(
                goto="Web Search Agent",
                update={"quality_feedback": quality_feedback},
                graph=Command.PARENT
            )

        else:
            if "remaining_steps" in state:
                state["remaining_steps"] = max(0, state["remaining_steps"] - 1)
            return Command(
                goto="end",
                update={"quality_feedback": quality_feedback},
                graph=Command.PARENT
            )