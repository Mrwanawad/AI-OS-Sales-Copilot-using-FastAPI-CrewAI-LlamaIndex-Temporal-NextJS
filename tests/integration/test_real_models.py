"""Exercises adapters with real, small weights. Run with: pytest -m models"""

import subprocess
import sys
from pathlib import Path

import pytest
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import AsyncQdrantClient
from sentence_transformers import CrossEncoder

from ai_os.modules.crm_copilot.adapters.outbound.speech import WhisperTranscriber
from ai_os.modules.knowledge.adapters.outbound.docling_parser import DoclingParser
from ai_os.modules.knowledge.adapters.outbound.llama_index_store import LlamaIndexVectorStore
from ai_os.modules.knowledge.adapters.outbound.reranker import CrossEncoderReranker
from ai_os.modules.knowledge.adapters.outbound.stores import qdrant_store
from ai_os.modules.knowledge.application.retrieve import RetrieveKnowledge
from ai_os.modules.knowledge.domain.chunking import chunk_sections
from ai_os.modules.knowledge.domain.models import DocumentKind, ParsedSection, SourceDocument

pytestmark = pytest.mark.models


async def test_real_embedder_and_cross_encoder_rank_the_relevant_chunk_first() -> None:
    embedder = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = AsyncQdrantClient(":memory:")
    store = LlamaIndexVectorStore(qdrant_store(client), embedder)
    reranker = CrossEncoderReranker(CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2"))
    document = SourceDocument("faq", "Palm Hills", DocumentKind.FAQ, 1)
    await store.upsert(
        chunk_sections(
            document,
            [
                ParsedSection("The down payment is 10 percent with 8 years of installments."),
                ParsedSection("The clubhouse has a gym and two swimming pools."),
                ParsedSection("Delivery of phase one is expected in 2027."),
            ],
        )
    )

    [hit] = await RetrieveKnowledge(store, reranker)("How much is the down payment?", top_k=1)

    assert "down payment" in hit.chunk.text
    await client.close()


def _minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = b"%PDF-1.4\n"
    offsets = []
    for number, content in enumerate(objects, start=1):
        offsets.append(len(body))
        body += b"%d 0 obj\n%s\nendobj\n" % (number, content)
    xref = len(body)
    body += b"xref\n0 6\n0000000000 65535 f \n"
    body += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    body += b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref
    return body


def _spoken_wav(path: Path, sentence: str) -> None:
    if sys.platform != "win32":
        pytest.skip("Generates audio with the Windows speech synthesizer")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{path}'); $s.Speak('{sentence}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, timeout=60)


async def test_real_whisper_transcribes_a_recording(tmp_path: Path) -> None:
    path = tmp_path / "call.wav"
    _spoken_wav(path, "The client wants a three bedroom apartment with a garden.")

    segments = await WhisperTranscriber("tiny").transcribe(path)

    text = " ".join(segment.text for segment in segments).casefold()
    assert "bedroom" in text
    assert "garden" in text
    assert segments[0].end > segments[0].start


async def test_docling_reads_pdf_and_reports_page(tmp_path: Path) -> None:
    path = tmp_path / "brochure.pdf"
    path.write_bytes(_minimal_pdf("Delivery in 2027"))

    sections = await DoclingParser().parse(path, SourceDocument("b", "P", DocumentKind.BROCHURE, 1))

    assert sections[0].page == 1
    assert "Delivery in 2027" in sections[0].text
