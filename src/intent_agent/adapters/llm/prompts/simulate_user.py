"""Evaluation prompt — the LLM role-plays the user described by a scenario."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a network engineer providing requirements for a 5G/6G deployment. "
     "Your deployment scenario: {scenario}\n\n"
     "Answer clarification questions concisely and directly. "
     "If given options, choose the most appropriate one. "
     "Do not ask questions back. Do not explain — just answer."),
    MessagesPlaceholder("history"),
    ("human", "Question: {question}"),
])
