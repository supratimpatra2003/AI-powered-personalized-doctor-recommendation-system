"""
Personalized Doctor Recommendation System
------------------------------------------

Local run:
    1. Create a folder and put app.py + requirements.txt inside it.
    2. Install dependencies:
           python -m pip install -r requirements.txt
    3. Start Streamlit:
           streamlit run app.py

Streamlit Community Cloud / GitHub:
    1. Create a GitHub repository and upload app.py + requirements.txt.
    2. Open https://share.streamlit.io/ (or Streamlit Community Cloud).
    3. Choose "Create app" and select your GitHub repository.
    4. Set the branch (normally main) and Main file path = app.py.
    5. Deploy.
    6. Streamlit Community Cloud reads requirements.txt and installs dependencies.

Important:
    This project uses a MOCK doctor dataset for demonstration only.
    It is not a medical diagnosis tool, and recommendations should not be
    treated as a substitute for professional medical advice.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CareMatch AI | Doctor Recommendation",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(62, 184, 174, 0.12), transparent 28%),
                radial-gradient(circle at 88% 15%, rgba(98, 111, 227, 0.11), transparent 24%),
                #f7f9fc;
        }

        .hero {
            padding: 1.3rem 1.45rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #0f172a 0%, #123b5d 58%, #0f766e 100%);
            color: white;
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.16);
            margin-bottom: 1.1rem;
        }

        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.15rem;
            letter-spacing: -0.03em;
        }

        .hero p {
            margin: 0;
            opacity: 0.88;
            font-size: 1rem;
        }

        .section-label {
            font-size: 0.82rem;
            font-weight: 800;
            color: #64748b;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.45rem 0 0.75rem 0;
        }

        .chip {
            display: inline-block;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            background: #eef6f6;
            color: #115e59;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid #d5eeee;
        }

        .muted {
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .match-caption {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 0.18rem;
        }

        .availability {
            margin-top: 0.2rem;
            color: #166534;
            font-weight: 700;
            font-size: 0.82rem;
        }

        .notice {
            padding: 0.75rem 0.9rem;
            border-radius: 12px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
            font-size: 0.88rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 0.55rem 0.7rem;
        }

        div[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }

        .small-help {
            font-size: 0.78rem;
            color: #64748b;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Mock data layer
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_doctors() -> pd.DataFrame:
    """Return a self-contained mock doctor dataset."""
    doctors = [
        {
            "doctor_id": "DOC-001",
            "name": "Dr. Ananya Sen",
            "specialty": "Cardiology",
            "location": "Kolkata",
            "language": "English, Bengali, Hindi",
            "years_experience": 17,
            "rating": 4.9,
            "bio": "Cardiologist focused on hypertension, preventive heart care, chest discomfort, and long-term cardiac risk management.",
            "available_days": ["Monday", "Wednesday", "Friday"],
        },
        {
            "doctor_id": "DOC-002",
            "name": "Dr. Rohan Mehta",
            "specialty": "Dermatology",
            "location": "Kolkata",
            "language": "English, Hindi",
            "years_experience": 11,
            "rating": 4.7,
            "bio": "Dermatologist treating acne, eczema, pigmentation, hair concerns, and common inflammatory skin conditions.",
            "available_days": ["Tuesday", "Thursday", "Saturday"],
        },
        {
            "doctor_id": "DOC-003",
            "name": "Dr. Arijit Mukherjee",
            "specialty": "Neurology",
            "location": "Kolkata",
            "language": "English, Bengali",
            "years_experience": 21,
            "rating": 4.8,
            "bio": "Neurologist with interests in headaches, migraine, neuropathy, seizures, and movement-related neurological disorders.",
            "available_days": ["Monday", "Tuesday", "Thursday"],
        },
        {
            "doctor_id": "DOC-004",
            "name": "Dr. Priya Nair",
            "specialty": "Endocrinology",
            "location": "Bengaluru",
            "language": "English, Hindi, Malayalam",
            "years_experience": 15,
            "rating": 4.9,
            "bio": "Endocrinologist focused on diabetes, thyroid disorders, metabolic health, and hormonal conditions.",
            "available_days": ["Monday", "Wednesday", "Saturday"],
        },
        {
            "doctor_id": "DOC-005",
            "name": "Dr. Vivek Sharma",
            "specialty": "Orthopedics",
            "location": "Delhi",
            "language": "English, Hindi, Punjabi",
            "years_experience": 19,
            "rating": 4.6,
            "bio": "Orthopedic physician managing joint pain, sports injuries, arthritis, back pain, and musculoskeletal rehabilitation.",
            "available_days": ["Tuesday", "Wednesday", "Friday"],
        },
        {
            "doctor_id": "DOC-006",
            "name": "Dr. Meera Iyer",
            "specialty": "Pediatrics",
            "location": "Chennai",
            "language": "English, Tamil, Hindi",
            "years_experience": 13,
            "rating": 4.9,
            "bio": "Pediatrician supporting childhood nutrition, fever, respiratory complaints, vaccinations, and preventive pediatric care.",
            "available_days": ["Monday", "Thursday", "Saturday"],
        },
        {
            "doctor_id": "DOC-007",
            "name": "Dr. Kunal Verma",
            "specialty": "Gastroenterology",
            "location": "Mumbai",
            "language": "English, Hindi, Marathi",
            "years_experience": 22,
            "rating": 4.8,
            "bio": "Gastroenterologist working with acidity, abdominal discomfort, reflux, liver health, and digestive disorders.",
            "available_days": ["Tuesday", "Thursday", "Friday"],
        },
        {
            "doctor_id": "DOC-008",
            "name": "Dr. Sneha Das",
            "specialty": "Gynecology",
            "location": "Kolkata",
            "language": "English, Bengali, Hindi",
            "years_experience": 16,
            "rating": 4.8,
            "bio": "Gynecologist providing care for menstrual health, PCOS, pregnancy support, and routine women's health needs.",
            "available_days": ["Monday", "Wednesday", "Saturday"],
        },
        {
            "doctor_id": "DOC-009",
            "name": "Dr. Aakash Kapoor",
            "specialty": "Psychiatry",
            "location": "Delhi",
            "language": "English, Hindi",
            "years_experience": 14,
            "rating": 4.7,
            "bio": "Psychiatrist supporting stress-related concerns, sleep difficulties, mood concerns, and general mental wellness care.",
            "available_days": ["Wednesday", "Friday", "Saturday"],
        },
        {
            "doctor_id": "DOC-010",
            "name": "Dr. Sayan Ghosh",
            "specialty": "ENT",
            "location": "Kolkata",
            "language": "English, Bengali, Hindi",
            "years_experience": 12,
            "rating": 4.6,
            "bio": "ENT specialist for sinus issues, sore throat, ear complaints, allergies, dizziness, and common head-and-neck conditions.",
            "available_days": ["Tuesday", "Thursday", "Sunday"],
        },
        {
            "doctor_id": "DOC-011",
            "name": "Dr. Kavya Rao",
            "specialty": "Pulmonology",
            "location": "Hyderabad",
            "language": "English, Telugu, Hindi",
            "years_experience": 18,
            "rating": 4.8,
            "bio": "Pulmonologist focused on asthma, cough, breathing difficulties, COPD, and respiratory wellness.",
            "available_days": ["Monday", "Wednesday", "Friday"],
        },
        {
            "doctor_id": "DOC-012",
            "name": "Dr. Neha Bansal",
            "specialty": "Ophthalmology",
            "location": "Pune",
            "language": "English, Hindi, Marathi",
            "years_experience": 10,
            "rating": 4.7,
            "bio": "Ophthalmologist providing comprehensive eye examinations, dry-eye care, vision screening, and common eye disease management.",
            "available_days": ["Tuesday", "Thursday", "Saturday"],
        },
        {
            "doctor_id": "DOC-013",
            "name": "Dr. Rahul Khanna",
            "specialty": "Urology",
            "location": "Delhi",
            "language": "English, Hindi, Punjabi",
            "years_experience": 20,
            "rating": 4.8,
            "bio": "Urologist working with urinary symptoms, kidney stone care, prostate health, and common urinary tract conditions.",
            "available_days": ["Monday", "Tuesday", "Friday"],
        },
        {
            "doctor_id": "DOC-014",
            "name": "Dr. Tanvi Chatterjee",
            "specialty": "General Medicine",
            "location": "Kolkata",
            "language": "English, Bengali, Hindi",
            "years_experience": 9,
            "rating": 4.9,
            "bio": "General physician for fever, fatigue, common infections, lifestyle-related concerns, and first-line adult care.",
            "available_days": ["Monday", "Wednesday", "Sunday"],
        },
        {
            "doctor_id": "DOC-015",
            "name": "Dr. Aditya Kulkarni",
            "specialty": "Cardiology",
            "location": "Pune",
            "language": "English, Hindi, Marathi",
            "years_experience": 24,
            "rating": 4.7,
            "bio": "Senior cardiologist with experience in coronary risk assessment, hypertension, preventive cardiology, and cardiac follow-up.",
            "available_days": ["Tuesday", "Thursday", "Saturday"],
        },
        {
            "doctor_id": "DOC-016",
            "name": "Dr. Ishita Roy",
            "specialty": "Endocrinology",
            "location": "Kolkata",
            "language": "English, Bengali, Hindi",
            "years_experience": 12,
            "rating": 4.8,
            "bio": "Endocrinologist focused on diabetes, insulin resistance, thyroid disorders, and personalized metabolic care.",
            "available_days": ["Monday", "Thursday", "Friday"],
        },
        {
            "doctor_id": "DOC-017",
            "name": "Dr. Harsh Patel",
            "specialty": "Orthopedics",
            "location": "Ahmedabad",
            "language": "English, Hindi, Gujarati",
            "years_experience": 16,
            "rating": 4.7,
            "bio": "Orthopedic specialist for knee pain, shoulder injuries, fractures, arthritis, and mobility-focused rehabilitation.",
            "available_days": ["Wednesday", "Friday", "Sunday"],
        },
        {
            "doctor_id": "DOC-018",
            "name": "Dr. Nandini Bose",
            "specialty": "Dermatology",
            "location": "Kolkata",
            "language": "English, Bengali",
            "years_experience": 8,
            "rating": 4.6,
            "bio": "Dermatologist focused on acne, pigmentation, sensitive skin, hair fall, and routine dermatological care.",
            "available_days": ["Monday", "Tuesday", "Saturday"],
        },
    ]

    df = pd.DataFrame(doctors)

    # Defensive cleanup: a mock-data typo should never crash the app.
    df["available_days"] = df["available_days"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    return df


# ---------------------------------------------------------------------------
# ML layer
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_tfidf_index(df: pd.DataFrame) -> tuple[TfidfVectorizer, object]:
    """Build and cache the TF-IDF representation of doctor profiles."""
    working = df.copy()

    working["document"] = (
        working["specialty"].fillna("").astype(str)
        + " "
        + working["location"].fillna("").astype(str)
        + " "
        + working["language"].fillna("").astype(str)
        + " "
        + working["bio"].fillna("").astype(str)
        + " "
        + working["available_days"]
        .apply(lambda days: " ".join(days) if isinstance(days, list) else str(days))
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(working["document"])
    return vectorizer, matrix


def _normalize(values: Iterable[float]) -> np.ndarray:
    """Min-max normalize an iterable, handling constant arrays safely."""
    arr = np.asarray(list(values), dtype=float)
    min_value = np.min(arr)
    max_value = np.max(arr)
    if np.isclose(max_value, min_value):
        return np.ones_like(arr)
    return (arr - min_value) / (max_value - min_value)


@st.cache_data(show_spinner=False)
def recommend_doctors(
    df: pd.DataFrame,
    query_text: str,
    preferred_language: str,
    location: str,
    min_rating: float,
) -> pd.DataFrame:
    """
    Rank doctors with a hybrid content + quality formula.

    Final score:
        65% TF-IDF cosine similarity
        10% exact language preference match
        10% exact location preference match
        10% historical rating
         5% experience
    """
    vectorizer, matrix = build_tfidf_index(df)

    clean_query = " ".join(query_text.strip().split())
    if clean_query:
        query_vector = vectorizer.transform([clean_query])
        cosine_scores = cosine_similarity(query_vector, matrix).ravel()
    else:
        cosine_scores = np.zeros(len(df), dtype=float)

    result = df.copy()
    result["cosine_similarity"] = cosine_scores

    result["language_match"] = np.where(
        preferred_language == "Any language",
        0.0,
        result["language"].str.contains(
            preferred_language, case=False, na=False, regex=False
        ).astype(float),
    )

    result["location_match"] = np.where(
        location == "Any location",
        0.0,
        (result["location"].str.casefold() == location.casefold()).astype(float),
    )

    result["rating_score"] = np.clip(result["rating"] / 5.0, 0.0, 1.0)
    result["experience_score"] = np.clip(result["years_experience"] / 25.0, 0.0, 1.0)

    result["match_score"] = (
        0.65 * result["cosine_similarity"]
        + 0.10 * result["language_match"]
        + 0.10 * result["location_match"]
        + 0.10 * result["rating_score"]
        + 0.05 * result["experience_score"]
    )

    result = result[result["rating"] >= min_rating].copy()
    result = result.sort_values(
        by=["match_score", "rating", "years_experience"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    result["match_percentage"] = np.clip(result["match_score"] * 100.0, 0.0, 100.0)
    return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def parse_languages(series: pd.Series) -> list[str]:
    """Extract unique language labels from comma-separated mock data."""
    languages: set[str] = set()
    for value in series.dropna().astype(str):
        for language in value.split(","):
            cleaned = language.strip()
            if cleaned:
                languages.add(cleaned)
    return sorted(languages)


def render_chip_row(items: Iterable[str]) -> None:
    """Render compact HTML chips."""
    chips = "".join(f'<span class="chip">{item}</span>' for item in items)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def build_query(
    target_text: str,
    specialty: str,
    preferred_language: str,
    location: str,
) -> str:
    """Create a natural-language search query for TF-IDF matching."""
    pieces: list[str] = []

    if target_text.strip():
        pieces.append(target_text.strip())

    if specialty != "Any specialty":
        pieces.append(specialty)

    if preferred_language != "Any language":
        pieces.append(preferred_language)

    if location != "Any location":
        pieces.append(location)

    return " ".join(pieces)


# ---------------------------------------------------------------------------
# Data + sidebar profile
# ---------------------------------------------------------------------------
doctors_df = load_doctors()

available_languages = ["Any language"] + parse_languages(doctors_df["language"])
available_locations = ["Any location"] + sorted(doctors_df["location"].unique())
available_specialties = ["Any specialty"] + sorted(doctors_df["specialty"].unique())
available_days = ["Any day"] + [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

with st.sidebar:
    st.markdown("## 🧑‍⚕️ Patient Profile")
    st.caption("Personalization controls")

    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    gender = st.selectbox(
        "Gender",
        ["Prefer not to say", "Female", "Male", "Non-binary", "Other"],
    )
    preferred_language = st.selectbox(
        "Preferred Language",
        available_languages,
        index=0,
    )
    location = st.selectbox(
        "Location",
        available_locations,
        index=0,
    )
    specialty = st.selectbox(
        "Target Specialty",
        available_specialties,
        index=0,
    )
    target_text = st.text_area(
        "Symptoms / Requirement",
        placeholder=(
            "Example: recurring headache, migraine, dizziness\n"
            "or: thyroid follow-up, diabetes management"
        ),
        height=120,
    )
    min_rating = st.slider(
        "Minimum Doctor Rating",
        min_value=3.0,
        max_value=5.0,
        value=4.0,
        step=0.1,
    )

    st.divider()
    st.markdown(
        '<div class="small-help"><b>Note:</b> Age and gender are collected as profile context but are not used to rank doctors in this demo. This keeps the recommendation logic focused on clinical-search preferences, language, location, ratings, and experience.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="section-label" style="color:#99f6e4;">CARE MATCH AI</div>
        <h1>Personalized Doctor Recommendation</h1>
        <p>Find doctors whose specialty, profile, language, location, ratings and experience align with your search.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

query = build_query(
    target_text=target_text,
    specialty=specialty,
    preferred_language=preferred_language,
    location=location,
)

if not query:
    st.warning(
        "Your search is empty. Add a specialty or symptoms/requirement to activate "
        "content-based matching. Until then, results are ordered mainly by quality signals."
    )

# Sidebar filters are applied first, then dashboard filters refine the visible list.
recommendations = recommend_doctors(
    doctors_df,
    query_text=query,
    preferred_language=preferred_language,
    location=location,
    min_rating=min_rating,
)

metric_a, metric_b, metric_c, metric_d = st.columns(4)
with metric_a:
    st.metric("Doctors in dataset", len(doctors_df))
with metric_b:
    st.metric("Above rating threshold", len(recommendations))
with metric_c:
    st.metric(
        "Avg. rating",
        f"{recommendations['rating'].mean():.2f}" if not recommendations.empty else "—",
    )
with metric_d:
    st.metric(
        "Top match",
        f"{recommendations['match_percentage'].iloc[0]:.0f}%"
        if not recommendations.empty
        else "—",
    )

st.markdown("### Refine Recommendations")

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

with filter_col1:
    selected_specialties = st.multiselect(
        "Specialty",
        options=sorted(doctors_df["specialty"].unique()),
        default=[],
        placeholder="All specialties",
    )

with filter_col2:
    selected_languages = st.multiselect(
        "Language",
        options=available_languages[1:],
        default=[],
        placeholder="All languages",
    )

with filter_col3:
    min_experience = st.slider(
        "Minimum Experience",
        min_value=0,
        max_value=int(doctors_df["years_experience"].max()),
        value=0,
        step=1,
        format="%d yrs",
    )

with filter_col4:
    selected_day = st.selectbox("Available Day", available_days, index=0)

filtered = recommendations.copy()

if selected_specialties:
    filtered = filtered[filtered["specialty"].isin(selected_specialties)]

if selected_languages:
    language_mask = filtered["language"].apply(
        lambda value: any(
            language.casefold() in value.casefold() for language in selected_languages
        )
    )
    filtered = filtered[language_mask]

filtered = filtered[filtered["years_experience"] >= min_experience]

if selected_day != "Any day":
    filtered = filtered[
        filtered["available_days"].apply(
            lambda days: selected_day in days if isinstance(days, list) else False
        )
    ]

view_col1, view_col2 = st.columns([1, 3])
with view_col1:
    top_n = st.slider("Show top N", min_value=1, max_value=12, value=6, step=1)
with view_col2:
    sort_mode = st.selectbox(
        "Sort cards by",
        ["Match score", "Rating", "Experience"],
        index=0,
    )

if sort_mode == "Rating":
    filtered = filtered.sort_values(["rating", "match_score"], ascending=[False, False])
elif sort_mode == "Experience":
    filtered = filtered.sort_values(
        ["years_experience", "match_score"], ascending=[False, False]
    )
else:
    filtered = filtered.sort_values(
        ["match_score", "rating", "years_experience"],
        ascending=[False, False, False],
    )

filtered = filtered.head(top_n).reset_index(drop=True)

if filtered.empty:
    st.warning(
        "No doctors match the current filters. Try lowering the minimum rating, "
        "reducing minimum experience, choosing another day, or clearing specialty/language filters."
    )
else:
    st.markdown(f"### {len(filtered)} Matching Doctors")

    # Render interactive doctor cards in two columns.
    for start in range(0, len(filtered), 2):
        card_row = st.columns(2)

        for offset, card_col in enumerate(card_row):
            idx = start + offset
            if idx >= len(filtered):
                continue

            doctor = filtered.iloc[idx]

            with card_col:
                with st.container(border=True):
                    title_col, score_col = st.columns([3.3, 1.2])
                    with title_col:
                        st.markdown(f"#### 🩺 {doctor['name']}")
                        st.caption(
                            f"{doctor['doctor_id']}  ·  {doctor['specialty']}  ·  {doctor['location']}"
                        )

                    with score_col:
                        st.metric(
                            "Match",
                            f"{doctor['match_percentage']:.0f}%",
                            delta=None,
                        )

                    st.progress(
                        int(round(float(doctor["match_percentage"]))),
                        text="Recommendation strength",
                    )

                    render_chip_row(
                        [
                            f"⭐ {doctor['rating']:.1f}/5",
                            f"🎓 {int(doctor['years_experience'])} yrs",
                            f"🗣️ {doctor['language']}",
                        ]
                    )

                    st.markdown(
                        f'<div class="muted">{doctor["bio"]}</div>',
                        unsafe_allow_html=True,
                    )

                    availability_text = ", ".join(doctor["available_days"])
                    st.markdown(
                        f'<div class="availability">● Available: {availability_text}</div>',
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        "View doctor details",
                        key=f"details_{doctor['doctor_id']}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_doctor_id"] = doctor["doctor_id"]

# ---------------------------------------------------------------------------
# Selected doctor detail panel
# ---------------------------------------------------------------------------
selected_id = st.session_state.get("selected_doctor_id")

if selected_id:
    selected = doctors_df[doctors_df["doctor_id"] == selected_id]

    if not selected.empty:
        doctor = selected.iloc[0]
        st.divider()
        st.markdown("### Selected Doctor Profile")

        left, right = st.columns([2.2, 1])
        with left:
            st.markdown(f"## {doctor['name']}")
            st.write(
                f"**{doctor['specialty']}** · {doctor['location']} · {doctor['doctor_id']}"
            )
            st.write(doctor["bio"])
            render_chip_row(
                [
                    f"⭐ {doctor['rating']:.1f}/5 rating",
                    f"🧠 {int(doctor['years_experience'])} years experience",
                    f"🗣️ {doctor['language']}",
                ]
            )

        with right:
            st.markdown("**Availability**")
            for day in doctor["available_days"]:
                st.write(f"✅ {day}")

# ---------------------------------------------------------------------------
# Explainability / technical details
# ---------------------------------------------------------------------------
with st.expander("🔍 How the recommendation engine works"):
    st.markdown(
        """
        **1. Content-based filtering**

        Each mock doctor gets a searchable text profile built from specialty,
        location, language, bio, and available days. The patient's search text
        is transformed with **TF-IDF**, then compared with doctor profiles using
        **cosine similarity**.

        **2. Hybrid ranking**

        The final recommendation score is:

        `0.65 × cosine similarity`
        `+ 0.10 × language match`
        `+ 0.10 × location match`
        `+ 0.10 × normalized rating`
        `+ 0.05 × normalized experience`

        **3. Real-time filtering**

        After ranking, the dashboard can further filter by specialty, language,
        minimum experience, availability day, and Top-N.

        **4. Caching**

        The mock dataset, TF-IDF index, and recommendation calculations are cached
        with `@st.cache_data` so repeated interactions avoid unnecessary recomputation.

        **5. Production extension**

        Replace `load_doctors()` with a database/API source and add a proper
        availability service, authentication, observability, data validation,
        privacy controls, and model evaluation before using this architecture
        in a real clinical product.
        """
    )

st.markdown(
    """
    <div class="notice">
        <b>Demo safety note:</b> This application uses synthetic doctor data and
        is intended for software/ML demonstration. It does not diagnose conditions,
        verify medical credentials, or replace professional medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)
