import os
import json
from typing import Dict, Any, List, Set, Optional

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ---- Configuration ----
DATA_DIR = os.path.join("data")
CHROMA_DIR = os.path.join("chroma_db", "jee_math_pyq")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Unique identifier field in metadata to track which questions are already ingested
DOC_KEY_FIELD = "doc_key"


# --------------------- JSON Loading & Parsing ---------------------


def load_paper_json(path: str) -> Dict[str, Any]:
    """Load one paper JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_documents_for_paper(
    paper_json: Dict[str, Any],
    source_file: str,
) -> List[Document]:
    """
    Convert one paper JSON into a list of Documents.

    Each question + solution + metadata becomes ONE chunk.
    One question = one Document with full question text, solution, and metadata.
    """
    paper_info = paper_json.get("paper_info", {}) or {}
    questions = paper_json.get("questions", []) or []

    # Paper-level metadata
    exam_year = paper_info.get("exam_year")
    exam_session = paper_info.get("exam_session")
    exam_date = paper_info.get("exam_date")
    exam_shift = paper_info.get("exam_shift")
    paper_subject = paper_info.get("subject")

    paper_id = (
        f"{exam_year}_s{exam_session}_p1"
        if exam_year and exam_session
        else os.path.basename(source_file).replace(".json", "")
    )

    docs: List[Document] = []

    for q in questions:
        qid = q.get("id", "")
        question_text = q.get("question_text", "").strip()
        options = q.get("options", []) or []
        solution = q.get("solution", {}) or {}
        tags = q.get("tags", {}) or {}

        intuition = solution.get("intuition", "")
        steps = solution.get("steps", []) or []
        trick = solution.get("trick", "")

        text_parts: List[str] = []

        # Paper context
        if exam_year:
            text_parts.append(f"Exam Year: {exam_year}")
        if exam_session:
            text_parts.append(f"Session: {exam_session}")
        if exam_date:
            text_parts.append(f"Date: {exam_date}")
        if exam_shift:
            text_parts.append(f"Shift: {exam_shift}")

        # Question
        text_parts.append(f"Question ID: {qid}")
        text_parts.append(f"Question: {question_text}")

        # Options
        if options:
            text_parts.append("\nOptions:")
            for opt in options:
                opt_id = opt.get("id", "")
                opt_text = opt.get("text", "")
                is_correct = opt.get("is_correct", False)
                marker = " ✓" if is_correct else ""
                text_parts.append(f"  {opt_id}. {opt_text}{marker}")

        # Solution
        text_parts.append("\nSolution:")
        if intuition:
            text_parts.append(f"Intuition: {intuition}")
        if steps:
            text_parts.append("Steps:")
            for i, step in enumerate(steps, 1):
                text_parts.append(f"  {i}. {step}")
        if trick:
            text_parts.append(f"Trick: {trick}")

        # Tags as readable summary
        if tags:
            tag_items = [f"{k}: {v}" for k, v in tags.items() if v]
            if tag_items:
                text_parts.append(f"\nTags: {', '.join(tag_items)}")

        page_content = "\n".join(text_parts)

        # Unique key per question for idempotent ingestion
        doc_key = f"{paper_id}::{qid}"

        metadata: Dict[str, Any] = {
            DOC_KEY_FIELD: doc_key,
            "paper_id": paper_id,
            "question_id": qid,
            "source_file": os.path.basename(source_file),
            "question_text": question_text,
            # Paper-level
            "exam_year": exam_year,
            "exam_session": exam_session,
            "exam_date": exam_date,
            "exam_shift": exam_shift,
            # Question-level from tags
            "subject": tags.get("subject") or paper_subject,
            "unit": tags.get("unit"),
            "chapter": tags.get("chapter"),
            "topic": tags.get("topic"),
            "difficulty": tags.get("difficulty"),
            "question_type": tags.get("question_type"),
            # Flatten tags dict into JSON string so Chroma accepts it
            "tags_json": json.dumps(tags, ensure_ascii=False) if tags else None,
        }

        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs


# --------------------- Chroma & Embeddings ---------------------

encode_kwargs = {
        'normalize_embeddings': True, 
        'batch_size': 4 
    }
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create HuggingFace embeddings using LangChain wrapper."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                 encode_kwargs=encode_kwargs)


def load_or_create_vectorstore() -> Chroma:
    """Load existing Chroma DB or create a new one."""
    embeddings = get_embeddings()

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def get_existing_doc_keys(vectorstore: Chroma) -> Set[str]:
    """
    Read all existing doc_keys from Chroma so we can skip already ingested questions.
    """
    existing_keys: Set[str] = set()

    try:
        collection = vectorstore._collection  # type: ignore[attr-defined]
        offset = 0
        page_size = 500

        while True:
            batch = collection.get(
                include=["metadatas"],
                limit=page_size,
                offset=offset,
            )
            metadatas = batch.get("metadatas") or []
            if not metadatas:
                break

            for md in metadatas:
                if not md:
                    continue
                key = md.get(DOC_KEY_FIELD)
                if key:
                    existing_keys.add(key)

            if len(metadatas) < page_size:
                break

            offset += page_size

    except Exception as e:
        print(f"  Note: could not read existing keys (maybe empty DB): {e}")
        return set()

    return existing_keys


# --------------------- Ingestion Orchestration ---------------------


def get_paper_files(data_dir: str, only_file: Optional[str] = None) -> List[str]:
    """
    List JSON paper files in data_dir.
    If only_file is provided, return just that file path.
    """
    if only_file:
        path = os.path.join(data_dir, only_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Paper file not found: {path}")
        return [path]

    files: List[str] = []
    if os.path.exists(data_dir):
        for name in os.listdir(data_dir):
            if name.lower().endswith(".json"):
                files.append(os.path.join(data_dir, name))

    files.sort()
    return files


def ingest_papers(only_file: Optional[str] = None) -> None:
    """
    Ingest all (or one) JSON papers into Chroma, only adding new questions.

    - Uses HuggingFaceEmbeddings + Chroma
    - One question+solution+metadata = one chunk
    - Re-running only adds new questions; old embeddings are preserved
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Data directory '{DATA_DIR}' does not exist. "
            "Create it and add your paper JSON files."
        )

    print("Loading ChromaDB vectorstore...")
    vectorstore = load_or_create_vectorstore()

    print("Checking existing questions in vectorstore...")
    existing_keys = get_existing_doc_keys(vectorstore)
    print(f"  Found {len(existing_keys)} question(s) already ingested.")

    paper_files = get_paper_files(DATA_DIR, only_file=only_file)
    if not paper_files:
        print(f"No JSON files found in '{DATA_DIR}'. Nothing to ingest.")
        return

    print(f"\nFound {len(paper_files)} paper file(s) to process.")

    total_new_docs = 0
    total_skipped = 0

    for paper_path in paper_files:
        paper_name = os.path.basename(paper_path)
        print(f"\n{'=' * 60}")
        print(f"Processing: {paper_name}")
        print(f"{'=' * 60}")

        try:
            paper_json = load_paper_json(paper_path)
            docs = build_documents_for_paper(paper_json, source_file=paper_name)

            new_docs: List[Document] = []
            skipped_count = 0

            for doc in docs:
                doc_key = doc.metadata.get(DOC_KEY_FIELD)
                if not doc_key:
                    print(
                        f"  Warning: Question missing doc_key, skipping: "
                        f"{doc.metadata.get('question_id')}"
                    )
                    skipped_count += 1
                    continue

                if doc_key in existing_keys:
                    skipped_count += 1
                    continue

                new_docs.append(doc)
                existing_keys.add(doc_key)

            if skipped_count > 0:
                print(f"  Skipped {skipped_count} question(s) (already ingested).")

            if not new_docs:
                print("  All questions from this paper are already ingested.")
                total_skipped += skipped_count
                continue

            print(f"  Adding {len(new_docs)} new question(s) to ChromaDB...")
            vectorstore.add_documents(new_docs)
            total_new_docs += len(new_docs)
            total_skipped += skipped_count
            print(f"  ✓ Successfully added {len(new_docs)} question(s).")

        except Exception as e:
            print(f"  ✗ Error processing {paper_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if total_new_docs > 0:
        print(f"\n{'=' * 60}")
        print(f"Persisting ChromaDB with {total_new_docs} new document(s)...")
        vectorstore.persist()
        print("✓ Ingestion complete!")
        print(f"  - Added: {total_new_docs} new question(s)")
        print(f"  - Skipped: {total_skipped} question(s) (already present)")
    else:
        print(f"\n{'=' * 60}")
        print("No new questions found to ingest. Vectorstore unchanged.")
        print(f"  - Skipped: {total_skipped} question(s) (already present)")


def main() -> None:
    """
    CLI entry point for the ingestion pipeline.

    Usage from project root:

        python -m src.ingestion_pipeline
            -> ingest all JSON papers in data/

        python -m src.ingestion_pipeline jm_2024_s1_p1.json
            -> ingest only that specific paper
    """
    import sys

    only_file = sys.argv[1] if len(sys.argv) > 1 else None

    if only_file:
        print(f"Ingesting single paper: {only_file}")
    else:
        print("Ingesting all papers from data/ folder")

    ingest_papers(only_file=only_file)


if __name__ == "__main__":
    main()