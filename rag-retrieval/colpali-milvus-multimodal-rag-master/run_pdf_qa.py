import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from middleware import Middleware
from rag import Rag


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = BASE_DIR.parent / "test.pdf"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_MAX_PAGES = 2
DEFAULT_RETRIEVAL_ONLY = False
DEFAULT_GLM_RETRIES = 5
DEFAULT_GLM_RETRY_DELAY = 8.0
DEFAULT_QUESTION_DELAY = 6.0
DEFAULT_QUESTIONS = [
    "What is the title of this supplementary material?",
    "Which conference submission is this supplementary material for?",
    "What datasets are listed in Table 4?",
    "What four metric panels are shown in Figure 10?",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PDF indexing and multimodal Q&A without launching the Gradio app."
    )
    parser.add_argument(
        "--pdf",
        default=str(DEFAULT_PDF_PATH),
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Maximum number of PDF pages to index.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output JSON file. Defaults to outputs/<pdf-stem>_qa_results.json.",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=None,
        help="Question to ask. Can be repeated multiple times.",
    )
    parser.add_argument(
        "--questions-file",
        default=None,
        help="Optional text file with one question per line.",
    )
    parser.add_argument(
        "--glm-retries",
        type=int,
        default=DEFAULT_GLM_RETRIES,
        help="Number of retries for GLM 429/rate-limit responses.",
    )
    parser.add_argument(
        "--glm-retry-delay",
        type=float,
        default=DEFAULT_GLM_RETRY_DELAY,
        help="Initial retry delay in seconds for GLM 429/rate-limit responses.",
    )
    parser.add_argument(
        "--question-delay",
        type=float,
        default=DEFAULT_QUESTION_DELAY,
        help="Delay in seconds between questions.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        default=DEFAULT_RETRIEVAL_ONLY,
        help="Run only local PDF indexing and retrieval without calling the cloud VLM.",
    )
    return parser.parse_args()


def load_questions(args):
    if args.question:
        return args.question

    if args.questions_file:
        questions_file = Path(args.questions_file).expanduser().resolve()
        return [line.strip() for line in questions_file.read_text().splitlines() if line.strip()]

    return DEFAULT_QUESTIONS


def build_output_path(pdf_path: Path, output_arg: str | None, retrieval_only: bool) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_retrieval_only_results.json" if retrieval_only else "_qa_results.json"
    return (DEFAULT_OUTPUT_DIR / f"{pdf_path.stem}{suffix}").resolve()


def ask_with_retry(rag: Rag, question: str, image_path: str, retries: int, retry_delay: float):
    delay = retry_delay
    last_answer = None

    for attempt in range(1, retries + 1):
        answer = rag.get_answer_from_glm(question, [image_path])
        last_answer = answer

        if "GLM API returned 429" not in answer:
            return answer, attempt

        if attempt < retries:
            print(f"GLM rate-limited on attempt {attempt}, retrying in {delay:.1f}s")
            time.sleep(delay)
            delay *= 2

    return last_answer, retries


def should_delay_between_questions(question_index: int, total_questions: int, delay: float) -> bool:
    return delay > 0 and question_index < total_questions - 1


def main():
    args = parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    output_path = build_output_path(pdf_path, args.output, args.retrieval_only)
    questions = load_questions(args)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not questions:
        raise ValueError("No questions provided.")

    run_id = f"{pdf_path.stem}-{uuid.uuid4().hex[:8]}"
    rag = None if args.retrieval_only else Rag()
    middleware = Middleware(run_id, create_collection=True)

    print(f"Indexing PDF: {pdf_path}")
    indexed_pages = middleware.index(
        pdf_path=str(pdf_path),
        id=run_id,
        max_pages=args.max_pages,
    )
    print(f"Indexed {len(indexed_pages)} page(s)")

    results = []
    for index, question in enumerate(questions):
        print(f"Asking: {question}")
        search_result = middleware.search([question])[0]
        top_score, top_doc_id = search_result[0]
        page_num = top_doc_id + 1
        image_path = BASE_DIR / "pages" / run_id / f"page_{page_num}.png"
        if args.retrieval_only:
            answer = None
            attempts = 0
        else:
            answer, attempts = ask_with_retry(
                rag,
                question,
                str(image_path),
                retries=args.glm_retries,
                retry_delay=args.glm_retry_delay,
            )

        results.append(
            {
                "question": question,
                "answer": answer,
                "page_num": int(page_num),
                "search_score": float(top_score),
                "image_path": str(image_path.resolve()),
                "glm_attempts": int(attempts),
            }
        )

        if should_delay_between_questions(index, len(questions), args.question_delay):
            print(f"Waiting {args.question_delay:.1f}s before next question")
            time.sleep(args.question_delay)

    payload = {
        "pdf_path": str(pdf_path),
        "run_id": run_id,
        "indexed_pages": indexed_pages,
        "max_pages": args.max_pages,
        "questions": questions,
        "retrieval_only": args.retrieval_only,
        "question_delay": args.question_delay,
        "results": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
