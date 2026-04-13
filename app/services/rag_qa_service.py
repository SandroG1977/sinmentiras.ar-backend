"""RAG Q&A service - Answer questions and evaluate truth index using legal documents."""

import json
import uuid
from datetime import datetime, timezone
from hashlib import sha256 as _sha256

from app.core.config import settings
from app.services.rag_service import rag_service


def answer_question(question: str, top_k: int = 3) -> dict[str, object]:
    """Answer a user question using RAG + LLM.

    Args:
        question: The user's question
        top_k: Number of relevant documents to retrieve

    Returns:
        Dict with: answer, sources, context, question
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError as exc:
        raise RuntimeError("langchain_openai required for answer_question") from exc

    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")

    # Retrieve relevant documents
    results = rag_service.retrieve(question, top_k)
    if not results:
        return {
            "answer": "No se encontró información relevante en la base de conocimientos.",
            "sources": [],
            "context": "",
            "question": question,
        }

    # Build context from retrieved documents
    context_parts: list[str] = []
    sources: list[dict[str, object]] = []

    for result in results:
        context_parts.append(str(result.get("text", "")))
        sources.append(
            {
                "text": result.get("text", "")[:200],
                "source": result.get("source", ""),
                "score": result.get("score", 0),
                "metadata": result.get("metadata", {}),
            }
        )

    context = "\n\n".join(context_parts)

    # System prompt for legal advisor
    system_prompt = """Vas a actuar como un Asesor Legislativo Experto y sin ideologías políticas en interpretar leyes argentinas. Tu tarea es ayudar a entender el contenido de estas leyes, explicando su articulado, contexto y posibles implicancias de manera clara y precisa. No debes inventar información ni hacer suposiciones sin base en el texto legal. Si no sabes la respuesta o el texto no lo especifica, debes decir que no se puede determinar a partir de la información disponible."""

    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Contexto legal:\n{contexto}\n\nPregunta: {pregunta}",
            ),
        ]
    )

    # Initialize LLM and chain
    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    chain = prompt | model | StrOutputParser()

    # Generate answer
    answer = chain.invoke({"contexto": context, "pregunta": question})

    return {
        "answer": answer,
        "sources": sources,
        "context": context,
        "question": question,
    }


def calculate_truth_index(statement: str, top_k: int = 3) -> dict[str, object]:
    """Calculate a truth index (0-1) for a statement against legal context.

    Args:
        statement: The statement to evaluate
        top_k: Number of relevant documents to retrieve

    Returns:
        Dict with: indice_verdad, justificacion, statement, sources
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError as exc:
        raise RuntimeError(
            "langchain_openai required for calculate_truth_index"
        ) from exc

    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")

    # Retrieve relevant legal documents
    results = rag_service.retrieve(statement, top_k)
    if not results:
        return {
            "indice_verdad": None,
            "justificacion": "No se encontró información legal relevante para evaluar.",
            "statement": statement,
            "sources": [],
        }

    # Build context from retrieved documents
    context_parts: list[str] = []
    sources: list[dict[str, object]] = []

    for result in results:
        context_parts.append(str(result.get("text", "")))
        sources.append(
            {
                "text": result.get("text", "")[:200],
                "source": result.get("source", ""),
                "score": result.get("score", 0),
                "metadata": result.get("metadata", {}),
            }
        )

    context = "\n\n".join(context_parts)

    # Prompt for truth evaluation
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Eres un verificador de consistencia legal.
Debes puntuar de 0 a 1 cuanto coincide una afirmación con el contexto legal proporcionado.
Responde SOLO JSON válido con este formato:
{"indice_verdad": 0.0, "justificacion": "texto breve"}
- 0: Completamente falso o contradice la ley
- 0.5: Parcialmente cierto o ambiguo
- 1.0: Completamente consistente con la ley""",
            ),
            (
                "human",
                "Afirmación: {afirmacion}\n\nContexto legal:\n{contexto_leyes}",
            ),
        ]
    )

    # Initialize LLM and chain
    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    chain = prompt | model | StrOutputParser()

    # Generate evaluation
    raw_response = chain.invoke({"afirmacion": statement, "contexto_leyes": context})

    # Parse JSON response
    try:
        resultado = json.loads(raw_response)
    except json.JSONDecodeError:
        resultado = {"indice_verdad": None, "justificacion": raw_response}

    resultado["statement"] = statement
    resultado["sources"] = sources
    resultado["context_used"] = context

    return resultado


def resolve_query(question: str, top_k: int = 5) -> dict[str, object]:
    """Resolve a question and return a structured audit resolution.

    Returns a dict matching ResolutionResponse:
        id, query, verdict, summary_ia, hash, source_law,
        source_url, original_text, highlights, news_context
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError as exc:
        raise RuntimeError("langchain_openai required for resolve_query") from exc

    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")

    results = rag_service.retrieve(question, top_k)

    if not results:
        year = datetime.now(timezone.utc).year
        short_id = uuid.uuid4().hex[:6].upper()
        return {
            "id": f"AUD-{year}-{short_id}",
            "query": question,
            "verdict": "SIN INFORMACIÓN",
            "summary_ia": "No se encontró información relevante en la base de conocimientos legales.",
            "hash": f"sha256:{_sha256(question.encode()).hexdigest()}",
            "source_law": "",
            "source_url": "",
            "original_text": "",
            "highlights": [],
            "news_context": [],
        }

    # Build context and pick the best source (highest score)
    context_parts: list[str] = []
    best_source = results[0]
    for r in results:
        context_parts.append(str(r.get("text", "")))
        if r.get("score", 0) > best_source.get("score", 0):
            best_source = r

    context = "\n\n".join(context_parts)

    # Extract source metadata for the best chunk
    meta: dict = best_source.get("metadata", {})
    source_law = (
        meta.get("source_norma")
        or meta.get("ley_numero")
        or best_source.get("source", "")
    )
    source_url = meta.get("publicado_en") or meta.get("source_uri", "")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Eres un auditor legal argentino. Dada una pregunta y el contexto legal,
debes generar una resolución estructurada.
Responde ÚNICAMENTE con un JSON válido con este esquema exacto:
{
  "verdict": "<uno de: VERDADERO | FALSO | INCONSISTENCIA TÉCNICA | AMBIGUO | SIN INFORMACIÓN>",
  "summary_ia": "<explicación clara y técnica de 2-4 oraciones>",
  "original_text": "<fragmento textual más relevante del contexto legal>",
  "highlights": ["<frase clave 1>", "<frase clave 2>"]
}
- verdict: clasifica la afirmación implícita en la pregunta
- summary_ia: explica el veredicto con base en el texto legal
- original_text: copia literal del fragmento más determinante
- highlights: 1-3 frases clave del original_text""",
            ),
            (
                "human",
                "Pregunta: {pregunta}\n\nContexto legal:\n{contexto}",
            ),
        ]
    )

    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    chain = prompt | model | StrOutputParser()

    raw = chain.invoke({"pregunta": question, "contexto": context})

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "verdict": "AMBIGUO",
            "summary_ia": raw,
            "original_text": best_source.get("text", ""),
            "highlights": [],
        }

    year = datetime.now(timezone.utc).year
    short_id = uuid.uuid4().hex[:6].upper()
    summary_hash = _sha256(parsed.get("summary_ia", "").encode()).hexdigest()

    return {
        "id": f"AUD-{year}-{short_id}",
        "query": question,
        "verdict": parsed.get("verdict", "AMBIGUO"),
        "summary_ia": parsed.get("summary_ia", ""),
        "hash": f"sha256:{summary_hash}",
        "source_law": str(source_law),
        "source_url": str(source_url),
        "original_text": parsed.get("original_text", best_source.get("text", "")),
        "highlights": parsed.get("highlights", []),
        "news_context": [],
    }
