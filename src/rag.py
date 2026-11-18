"""Retrieval-augmented answering: condense follow-ups, retrieve, stream the answer."""

import logging
from typing import AsyncIterator, Iterable

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.config import Settings
from src.helper import download_hugging_face_embeddings
from src.prompt import CONDENSE_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

ANSWER_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

CONDENSE_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "Follow-up message: {question}"),
    ]
)

SNIPPET_CHARS = 220


def to_lc_messages(history: Iterable[dict], limit: int) -> list[BaseMessage]:
    """Convert [{role, content}] dicts to LangChain messages, keeping the last `limit`."""
    messages: list[BaseMessage] = []
    for item in list(history)[-limit:] if limit > 0 else []:
        if item["role"] == "user":
            messages.append(HumanMessage(item["content"]))
        else:
            messages.append(AIMessage(item["content"]))
    return messages


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)


def summarize_sources(docs: list[Document]) -> list[dict]:
    """One entry per source page, in retrieval order."""
    seen: set[str] = set()
    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        page = meta.get("page_label") or (
            int(meta["page"]) + 1 if isinstance(meta.get("page"), (int, float)) else None
        )
        topic = meta.get("topic")
        key = f"{meta.get('source')}:{topic}:{page}"
        if key in seen:
            continue
        seen.add(key)
        # Drop the "Topic: …\nSection: …" header that chunks carry for embedding.
        body = doc.page_content.split("\n\n", 1)[-1] if topic else doc.page_content
        text = " ".join(body.split())
        sources.append(
            {
                "page": str(page) if page is not None else None,
                "topic": topic,
                "section": meta.get("section"),
                "snippet": text[:SNIPPET_CHARS] + ("…" if len(text) > SNIPPET_CHARS else ""),
            }
        )
    return sources


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Loading embedding model %s", settings.embedding_model)
        embeddings = download_hugging_face_embeddings(settings.embedding_model)

        logger.info("Connecting to Pinecone index '%s'", settings.pinecone_index)
        store = PineconeVectorStore.from_existing_index(
            index_name=settings.pinecone_index, embedding=embeddings
        )
        self.retriever = store.as_retriever(
            search_type="similarity", search_kwargs={"k": settings.top_k}
        )

        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
            max_retries=2,
            timeout=60,
        )
        condense_llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
            max_retries=2,
            timeout=30,
        )
        self.answer_chain = ANSWER_TEMPLATE | llm | StrOutputParser()
        self.condense_chain = CONDENSE_TEMPLATE | condense_llm | StrOutputParser()

    async def _standalone_question(self, question: str, history: list[BaseMessage]) -> str:
        if not history:
            return question
        rewritten = (await self.condense_chain.ainvoke(
            {"chat_history": history, "question": question}
        )).strip()
        logger.info("Condensed follow-up to: %s", rewritten)
        return rewritten or question

    async def stream(self, question: str, history: list[dict]) -> AsyncIterator[dict]:
        """Yield events: one 'sources' event, then 'token' events."""
        chat_history = to_lc_messages(history, self.settings.max_history_messages)
        search_query = await self._standalone_question(question, chat_history)

        docs = await self.retriever.ainvoke(search_query)
        logger.info("Retrieved %d chunks", len(docs))
        yield {"type": "sources", "sources": summarize_sources(docs)}

        async for token in self.answer_chain.astream(
            {
                "context": format_docs(docs),
                "chat_history": chat_history,
                "question": question,
            }
        ):
            if token:
                yield {"type": "token", "content": token}
