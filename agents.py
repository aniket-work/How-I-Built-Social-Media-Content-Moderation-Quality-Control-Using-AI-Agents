from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
from models import Agent


class Swarm:
    def __init__(self, model: str, temperature: float = 0):
        self.llm = ChatOllama(model=model, temperature=temperature)

    def run(self, agent: Agent, messages: List[Any], context_variables: Dict = None, max_turns: float = float("inf")):
        context_variables = context_variables or {}
        current_agent = agent
        response_messages = messages[:]
        turn_count = 0

        while turn_count < max_turns:
            messages_for_llm = self._prepare_messages(current_agent, response_messages, context_variables)
            response = self.llm(messages_for_llm)
            response_message = HumanMessage(content=response.content)
            response_messages.append(response_message)

            next_agent = self._handle_handoff(current_agent, response, context_variables)
            if next_agent is None:
                break
            current_agent = next_agent
            turn_count += 1

        return {
            "messages": response_messages,
            "agent": current_agent,
            "context_variables": context_variables,
        }

    def _prepare_messages(self, agent: Agent, messages: List[Any], context_variables: Dict) -> List[Any]:
        instructions = agent.instructions if isinstance(agent.instructions, str) else agent.instructions(context_variables)
        return [SystemMessage(content=instructions)] + messages

    def _handle_handoff(self, agent: Agent, response: Any, context_variables: Dict) -> Agent:
        if "handoff" in response.content:
            return response.content.get("handoff")
        return None
