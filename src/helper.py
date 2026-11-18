"""PDF loading, encyclopedia-aware chunking, and the embedding model."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 truncates input at 256 tokens (~1000 characters), so chunks
# (including the "Topic/Section" header) must stay under that to be fully embedded.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

# Section headings used inside Gale Encyclopedia of Medicine entries.
SECTION_HEADINGS = {
    "Definition", "Description", "Purpose", "Precautions", "Demographics",
    "Causes and symptoms", "Causes", "Symptoms", "Diagnosis", "Treatment",
    "Alternative treatment", "Alternative treatments", "Prognosis", "Prevention",
    "Preparation", "Aftercare", "Risks", "Results", "Normal results",
    "Abnormal results", "Side effects", "Interactions", "Recommended dosage",
    "Morbidity and mortality rates", "Alternatives", "Health care team roles",
    "Origins", "Benefits", "Applications", "Training and certification",
    "Key terms", "KEY TERMS", "Resources",
}
# Sections that are bibliography/contact lists: noise for retrieval.
SKIPPED_SECTIONS = {"Resources"}

RUNNING_HEADER = re.compile(r"^GALE ENCYCLOPEDIA OF MEDICINE")


def load_pdf_file(data):
    """Load every PDF in a directory into LangChain documents (one per page)."""
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


@dataclass
class _Section:
    name: str
    lines: list[tuple[str, int]] = field(default_factory=list)  # (line, page)


@dataclass
class _Entry:
    topic: str
    page: int
    sections: list[_Section] = field(default_factory=list)


def _page_lines(text: str) -> list[str]:
    """Page text as stripped lines, minus the running footer and its entry-name line."""
    lines = [line.strip() for line in text.replace("�", "'").split("\n")]
    for i, line in enumerate(lines):
        if RUNNING_HEADER.match(line):
            # The footer is followed by the entry name; anything after is figure captions.
            return lines[:i] + lines[i + 2:]
    return lines


def _title_before(buffer: list[tuple[str, int]]) -> tuple[str, int]:
    """Pop the entry title (1-3 wrapped lines) off the end of `buffer`."""
    parts: list[str] = []
    page = buffer[-1][1] if buffer else 0
    while buffer:
        line, page = buffer.pop()
        parts.insert(0, line)
        # A wrapped title continues on a line starting lowercase or with "(".
        if not (line[:1].islower() or line.startswith("(")) or len(parts) >= 3:
            break
    title = ""
    for part in parts:
        if title.endswith("-"):
            title = title + part
        else:
            title = f"{title} {part}".strip()
    return title, page


def _join_lines(lines: list[str]) -> str:
    """Join wrapped PDF lines into prose, undoing end-of-line hyphenation."""
    text = ""
    for line in lines:
        if not line:
            continue
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        elif text:
            text += " " + line
        else:
            text = line
    return re.sub(r"\s+", " ", text).strip()


def parse_entries(pages: list[Document]) -> list[_Entry]:
    """Group page lines into encyclopedia entries, each split into named sections."""
    entries: list[_Entry] = []
    pending: list[tuple[str, int]] = []  # lines seen since the last section heading

    for doc in pages:
        page = int(doc.metadata.get("page", 0))
        for line in _page_lines(doc.page_content):
            if line == "Definition":
                # New entry: its title sits on the line(s) just before "Definition".
                title, title_page = _title_before(pending)
                if entries:
                    entries[-1].sections[-1].lines.extend(pending)
                entries.append(_Entry(topic=title, page=title_page,
                                      sections=[_Section("Definition")]))
                pending = []
            elif entries and line in SECTION_HEADINGS:
                entries[-1].sections[-1].lines.extend(pending)
                name = "Key terms" if line == "KEY TERMS" else line
                entries[-1].sections.append(_Section(name))
                pending = []
            else:
                pending.append((line, page))
    if entries:
        entries[-1].sections[-1].lines.extend(pending)
    # Text before the first "Definition" is front matter (contents, contributors).
    return entries


def text_split(extracted_data: list[Document]) -> list[Document]:
    """Chunk the encyclopedia by entry and section.

    Each chunk carries the entry's topic and section in its text (so the embedding
    "knows" what it is about) and in metadata (for display and filtering).
    """
    source = extracted_data[0].metadata.get("source", "") if extracted_data else ""
    chunks: list[Document] = []

    for entry in parse_entries(extracted_data):
        for section in entry.sections:
            if section.name in SKIPPED_SECTIONS or not section.lines:
                continue
            # Track which page each character offset came from.
            text, offsets = "", []
            for line, page in section.lines:
                before = len(text)
                text = _join_lines([text, line]) if text else _join_lines([line])
                if len(text) > before:
                    offsets.append((before, page))
            if len(text) < 40:
                continue

            header = f"Topic: {entry.topic}\nSection: {section.name}\n\n"
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max(CHUNK_SIZE - len(header), 300),
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
            )
            for piece in splitter.create_documents([text]):
                start = piece.metadata["start_index"]
                page = next((p for off, p in reversed(offsets) if off <= start),
                            offsets[0][1] if offsets else entry.page)
                chunks.append(Document(
                    page_content=header + piece.page_content,
                    metadata={
                        "source": Path(source).name if source else "",
                        "topic": entry.topic,
                        "section": section.name,
                        "page": page,  # 0-based, like PyPDFLoader
                    },
                ))
    return chunks


def download_hugging_face_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load the sentence-transformers embedding model (384 dimensions)."""
    return HuggingFaceEmbeddings(model_name=model_name)
