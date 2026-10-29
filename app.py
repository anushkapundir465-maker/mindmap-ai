from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.analysis import analyze_answer, load_semantic_model
from src.demo import DEMO_ANALYSIS, demo_attempts
from src.model import load_trained_model, predict_text
from src.quiz import generate_quiz, grade_quiz
from src.recommendations import TOPIC_RECOMMENDATIONS, recommendations_for
from src.storage import load_attempts, save_attempt


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "mindmap_text_classifier.pt"
PAGES = [
    "Dashboard",
    "Analyze an answer",
    "Practice quiz",
    "Progress & reports",
    "Recommendations",
    "About",
]
TOPICS = list(TOPIC_RECOMMENDATIONS)
NAVY = "#0B1220"
PANEL = "#121D31"
ACCENT = "#72E4C5"
TEXT = "#F1F5F9"
MUTED = "#9BAAC0"


st.set_page_config(
    page_title="MindMap AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background: {NAVY}; color: {TEXT}; }}
    [data-testid="stSidebar"] {{ background: #0E1728; border-right: 1px solid #26344C; }}
    .hero {{
        padding: 1.5rem 1.75rem; border: 1px solid #26344C; border-radius: 18px;
        background: linear-gradient(135deg, #14233B, {PANEL}); margin-bottom: 1rem;
    }}
    .hero p {{ color: {MUTED}; margin-bottom: 0; }}
    .stButton > button, .stFormSubmitButton > button {{
        background: {ACCENT}; color: #07131D; border: 0; border-radius: 10px;
        font-weight: 700; min-height: 2.6rem;
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{
        background: #A0F2DC; color: #07131D;
    }}
    div[data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid #26344C; padding: 1rem;
        border-radius: 14px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("quiz_items", [])
st.session_state.setdefault("quiz_result", None)
st.session_state.setdefault("quiz_saved", False)
st.session_state.setdefault("analysis_result", None)
st.session_state.setdefault("navigation", "Dashboard")
st.session_state.setdefault("demo_mode", True)


@st.cache_resource(show_spinner="Loading the trained text classifier…")
def cached_classifier(model_path: Path) -> dict[str, Any]:
    return load_trained_model(model_path)


@st.cache_resource(show_spinner="Loading the semantic similarity model…")
def cached_semantic_model() -> tuple[Any | None, str]:
    return load_semantic_model()


def current_attempts() -> list[dict[str, Any]]:
    if st.session_state.demo_mode:
        return demo_attempts()
    try:
        return load_attempts()
    except RuntimeError as error:
        st.error(f"Could not load saved progress: {error}")
        return []


def save_personal_attempt(attempt: dict[str, Any]) -> bool:
    if st.session_state.demo_mode:
        return True
    try:
        save_attempt(attempt)
    except (OSError, RuntimeError, TypeError) as error:
        st.error(f"Your result was calculated, but progress could not be saved: {error}")
        return False
    return True


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def score_frame(attempts: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "Topic": str(item.get("topic", "Unspecified")),
            "Score (%)": float(item.get("score", 0)),
            "Activity": str(item.get("kind", "Activity")).replace("_", " ").title(),
            "Date": str(item.get("timestamp", "")),
        }
        for item in attempts
    ]
    return pd.DataFrame(rows, columns=["Topic", "Score (%)", "Activity", "Date"])


def render_dashboard(attempts: list[dict[str, Any]]) -> None:
    render_hero("MindMap AI", "Understand learning gaps. Practice with purpose.")
    quiz_attempts = [item for item in attempts if item.get("kind") == "quiz"]
    average = (
        sum(float(item.get("score", 0)) for item in quiz_attempts) / len(quiz_attempts)
        if quiz_attempts
        else None
    )
    first, second, third = st.columns(3)
    first.metric("Activities completed", len(attempts))
    second.metric("Average quiz score", f"{average:.0f}%" if average is not None else "—")
    third.metric("Learning mode", "Demo" if st.session_state.demo_mode else "Personal")

    left, right = st.columns([1.2, 1])
    with left:
        with st.container(border=True):
            st.subheader("Quick start")
            st.write("Compare an explanation with your notes, or create a quiz from study material.")
            st.caption("Choose **Analyze an answer** or **Practice quiz** from the navigation menu.")
    with right:
        with st.container(border=True):
            st.subheader("Recent learning activity")
            if attempts:
                st.dataframe(
                    score_frame(attempts).tail(5),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("Your completed quizzes will appear here.")

    if quiz_attempts:
        by_topic = (
            pd.DataFrame(
                [
                    {"Topic": item.get("topic", "Unspecified"), "Score (%)": float(item.get("score", 0))}
                    for item in quiz_attempts
                ]
            )
            .groupby("Topic", as_index=False)["Score (%)"]
            .mean()
        )
        if not by_topic.empty:
            st.subheader("Quiz scores by topic")
            st.bar_chart(by_topic, x="Topic", y="Score (%)")


def render_analysis() -> None:
    render_hero("Analyze an answer", "Compare an explanation with reference material and find concepts to revisit.")
    with st.form("answer_analysis_form", border=True):
        topic = st.selectbox(
            "Topic", TOPICS, key="analysis_topic", persist_state="session"
        )
        reference = st.text_area(
            "Reference material",
            value=DEMO_ANALYSIS["reference"] if st.session_state.demo_mode else "",
            height=170,
            key="analysis_reference",
            persist_state="session",
            placeholder="Paste notes or an expected explanation…",
        )
        answer = st.text_area(
            "Your answer",
            value=DEMO_ANALYSIS["answer"] if st.session_state.demo_mode else "",
            height=150,
            key="analysis_answer",
            persist_state="session",
            placeholder="Write or paste your explanation…",
        )
        enable_semantic = st.checkbox(
            "Include semantic similarity (may download a model on first use)",
            value=False,
            key="enable_semantic",
            persist_state="session",
        )
        submitted = st.form_submit_button("Analyze answer", type="primary")

    if submitted:
        try:
            result = analyze_answer(reference, answer, topic)
        except ValueError as error:
            st.session_state.analysis_result = None
            st.error(str(error))
        else:
            result["topic"] = topic
            result["answer"] = answer
            if enable_semantic:
                semantic_model, model_error = cached_semantic_model()
                if semantic_model is None:
                    st.warning(f"Semantic similarity is unavailable. TF-IDF results are still shown. {model_error}")
                else:
                    try:
                        embeddings = semantic_model.encode([reference, answer], normalize_embeddings=True)
                        result["semantic_similarity"] = float(
                            max(0.0, min(1.0, embeddings[0] @ embeddings[1]))
                        )
                    except (OSError, RuntimeError, ValueError) as error:
                        st.warning(f"Semantic similarity could not be calculated: {error}")
            st.session_state.analysis_result = result

    result = st.session_state.analysis_result
    if not result:
        return

    st.subheader("Analysis")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Keyword coverage", f"{result['keyword_coverage']:.0%}")
    metric_columns[1].metric("TF-IDF similarity", f"{result['tfidf_similarity']:.0%}")
    if "semantic_similarity" in result:
        metric_columns[2].metric("Semantic similarity", f"{result['semantic_similarity']:.0%}")
    else:
        metric_columns[2].metric("Semantic similarity", "Not run")

    missing = result["missing_concepts"]
    if missing:
        st.warning("Consider reviewing: " + ", ".join(missing))
    else:
        st.success("No mapped topic concepts were missing from the answer.")
    for recommendation in recommendations_for(str(result["topic"]), missing):
        st.markdown(f"- {recommendation}")

    if MODEL_PATH.exists():
        try:
            classifier = cached_classifier(MODEL_PATH)
            label, confidence = predict_text(classifier, str(result["answer"]))
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
            st.warning(f"The trained classifier could not analyze this answer: {error}")
        else:
            st.info(f"Classifier signal: **{label.title()}** ({confidence:.0%} confidence). This is not a mastery score.")
    else:
        st.info("The trained classifier file was not found; answer analysis is available without it.")

    report_rows = [
        {"Measure": "Keyword coverage", "Result": f"{result['keyword_coverage']:.4f}"},
        {"Measure": "TF-IDF similarity", "Result": f"{result['tfidf_similarity']:.4f}"},
        {"Measure": "Missing concepts", "Result": ", ".join(missing) or "None"},
        {"Measure": "Recommendations", "Result": " | ".join(recommendations_for(str(result["topic"]), missing))},
    ]
    if "semantic_similarity" in result:
        report_rows.append(
            {"Measure": "Semantic similarity", "Result": f"{result['semantic_similarity']:.4f}"}
        )
    report = pd.DataFrame(report_rows)
    st.download_button(
        "Download analysis report",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="mindmap_answer_analysis.csv",
        mime="text/csv",
    )


def extract_uploaded_text(uploaded_file: Any) -> str:
    suffix = Path(uploaded_file.name).suffix.casefold()
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8-sig")
    if suffix != ".pdf":
        raise ValueError("Upload a PDF or UTF-8 text file.")
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except (OSError, PdfReadError, ValueError) as error:
        raise ValueError(f"Could not read the uploaded PDF: {error}") from error


def render_quiz() -> None:
    render_hero("Practice quiz", "Generate experimental multiple-choice questions from notes or a text-based PDF.")
    uploaded_file = st.file_uploader(
        "Optional study file",
        type=["pdf", "txt"],
        help="PDFs must contain selectable text; scanned PDFs need OCR, which is not included.",
    )
    with st.form("quiz_generation_form", border=True):
        topic = st.selectbox(
            "Quiz topic", TOPICS, key="quiz_topic", persist_state="session"
        )
        notes = st.text_area(
            "Notes or study text",
            height=220,
            key="quiz_notes",
            persist_state="session",
            placeholder="Paste notes here, or upload a PDF/text file above…",
        )
        question_count = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key="quiz_question_count",
            persist_state="session",
        )
        generate = st.form_submit_button("Generate quiz", type="primary")

    if generate:
        try:
            source_text = extract_uploaded_text(uploaded_file) if uploaded_file is not None else notes
        except (UnicodeDecodeError, ValueError) as error:
            st.session_state.quiz_items = []
            st.session_state.quiz_result = None
            st.error(str(error))
        else:
            for index in range(10):
                st.session_state.pop(f"quiz_answer_{index}", None)
            questions = generate_quiz(source_text, int(question_count))
            st.session_state.quiz_items = questions
            st.session_state.quiz_result = None
            st.session_state.quiz_saved = False
            st.session_state.quiz_topic_value = topic
            st.session_state.quiz_source = (
                uploaded_file.name if uploaded_file is not None else "pasted notes"
            )
            if not questions:
                st.info(
                    "No questions could be generated. Add more explanatory text with at least "
                    "six words per sentence and enough distinct terms for answer choices."
                )

    questions = st.session_state.quiz_items
    if not questions:
        return

    st.subheader(f"Your quiz · {len(questions)} questions")
    for index, question in enumerate(questions):
        st.radio(
            f"{index + 1}. {question['question']}",
            options=question["options"],
            index=None,
            key=f"quiz_answer_{index}",
            persist_state="session",
        )

    if st.button("Check answers", type="primary", key="grade_quiz"):
        selected_answers = [
            st.session_state.get(f"quiz_answer_{index}") for index in range(len(questions))
        ]
        try:
            score, feedback = grade_quiz(questions, selected_answers)
        except ValueError as error:
            st.warning(str(error))
        else:
            result = {"score": score, "feedback": feedback}
            st.session_state.quiz_result = result
            correct_count = sum(correct for correct, _ in feedback)
            if not st.session_state.quiz_saved:
                st.session_state.quiz_saved = save_personal_attempt(
                    {
                        "kind": "quiz",
                        "topic": st.session_state.get("quiz_topic_value", "Unspecified"),
                        "score": score,
                        "total": len(questions),
                        "correct": correct_count,
                        "source": st.session_state.get("quiz_source", "study notes"),
                    }
                )

    quiz_result = st.session_state.quiz_result
    if quiz_result:
        score = float(quiz_result["score"])
        st.metric("Quiz score", f"{score:.0f}%")
        for index, (correct, explanation) in enumerate(quiz_result["feedback"], start=1):
            if correct:
                st.success(f"Question {index}: Correct. {explanation}")
            else:
                st.error(f"Question {index}: Review this explanation. {explanation}")


def render_progress(attempts: list[dict[str, Any]]) -> None:
    render_hero("Progress & reports", "Review activity and export a copy of your learning history.")
    if not attempts:
        st.info("No activity is available yet. Complete a quiz to start tracking progress.")
        return
    frame = score_frame(attempts)
    quiz_frame = frame[frame["Activity"] == "Quiz"]
    first, second = st.columns(2)
    first.metric("Activities", len(attempts))
    second.metric(
        "Average quiz score",
        f"{quiz_frame['Score (%)'].mean():.0f}%" if not quiz_frame.empty else "—",
    )
    st.dataframe(
        frame.sort_values("Date", ascending=False),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download progress report",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="mindmap_progress.csv",
        mime="text/csv",
    )
    if not quiz_frame.empty:
        by_topic = quiz_frame.groupby("Topic", as_index=False)["Score (%)"].mean()
        st.subheader("Average quiz score by topic")
        st.bar_chart(by_topic, x="Topic", y="Score (%)")


def render_recommendations(attempts: list[dict[str, Any]]) -> None:
    render_hero("Recommendations", "Choose a topic to get focused next steps for your study session.")
    quiz_attempts = [item for item in attempts if item.get("kind") == "quiz"]
    if quiz_attempts:
        weakest = min(quiz_attempts, key=lambda item: float(item.get("score", 0)))
        default_topic = str(weakest.get("topic", TOPICS[0]))
        if default_topic not in TOPICS:
            default_topic = TOPICS[0]
        st.caption(
            f"Based on recent quiz activity, consider reviewing **{default_topic}** "
            f"(latest low score: {float(weakest.get('score', 0)):.0f}%)."
        )
    else:
        default_topic = TOPICS[0]
        st.caption("Complete a quiz to personalize recommendations from your progress.")
    topic = st.selectbox(
        "Study topic",
        TOPICS,
        index=TOPICS.index(default_topic),
        key="recommendation_topic",
        persist_state="session",
    )
    for recommendation in recommendations_for(topic):
        st.markdown(f"- {recommendation}")


def render_about() -> None:
    render_hero("About MindMap AI", "A local learning dashboard for reflection, practice, and progress.")
    st.markdown(
        """
        - **Answer analysis:** keyword coverage, TF-IDF similarity, mapped missing concepts,
          optional sentence-transformer similarity, and the included PyTorch classifier.
        - **Practice quizzes:** experimental fill-in-the-blank questions from notes or
          selectable text in a PDF.
        - **Progress and recommendations:** demo data is read-only; Personal mode stores
          completed quizzes in `data/progress.json`.

        Similarity scores and classifier output are learning aids, not proof of correctness
        or a measure of mastery. PDF OCR is not included.
        """
    )


with st.sidebar:
    st.markdown("## 🧠 MindMap AI")
    st.caption("Student learning insights")
    st.radio("Navigation", PAGES, key="navigation", label_visibility="collapsed")
    st.divider()
    st.caption("Dark Navy · Premium AI")
    st.toggle("Demo mode", key="demo_mode")
    st.caption("Demo mode uses sample activity. Personal mode saves quiz progress locally.")

attempts = current_attempts()
page = st.session_state.navigation

if page == "Dashboard":
    render_dashboard(attempts)
elif page == "Analyze an answer":
    render_analysis()
elif page == "Practice quiz":
    render_quiz()
elif page == "Progress & reports":
    render_progress(attempts)
elif page == "Recommendations":
    render_recommendations(attempts)
else:
    render_about()
