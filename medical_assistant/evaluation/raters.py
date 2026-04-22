from medical_assistant.config.settings import Settings
from medical_assistant.core.llm import get_llm

_GROUNDEDNESS_PROMPT = """You are evaluating whether a medical answer is grounded in the provided context.
Score on a scale of 1–5:
  1 = Answer contains claims not found in the context
  3 = Answer mostly follows the context with minor additions
  5 = Answer strictly and completely follows the context

Context:
{context}

Answer:
{answer}

Provide your score (1–5) and a one-sentence justification."""

_RELEVANCE_PROMPT = """You are evaluating whether a medical answer addresses the question asked.
Score on a scale of 1–5:
  1 = Answer does not address the question
  3 = Answer partially addresses the question
  5 = Answer completely addresses all aspects of the question

Question:
{question}

Answer:
{answer}

Provide your score (1–5) and a one-sentence justification."""


def rate_groundedness(context: str, answer: str, settings: Settings) -> dict:
    llm = get_llm(settings)
    prompt = _GROUNDEDNESS_PROMPT.format(context=context, answer=answer)
    result = llm(prompt, max_tokens=150, temperature=0.0)
    return {
        "rating_type": "groundedness",
        "result": result["choices"][0]["text"].strip(),
    }


def rate_relevance(question: str, answer: str, settings: Settings) -> dict:
    llm = get_llm(settings)
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
    result = llm(prompt, max_tokens=150, temperature=0.0)
    return {
        "rating_type": "relevance",
        "result": result["choices"][0]["text"].strip(),
    }
