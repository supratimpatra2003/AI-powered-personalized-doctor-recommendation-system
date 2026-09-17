"""
CareMatch AI — Personalized Doctor Recommendation + Appointment Booking

LOCAL RUN
--------
1. Put app.py and requirements.txt in the same folder.
2. Install dependencies:
       python -m pip install -r requirements.txt
3. Run:
       streamlit run app.py

STREAMLIT COMMUNITY CLOUD / GITHUB
-----------------------------------
1. Push app.py and requirements.txt to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select the repository, branch (usually "main"), and main file "app.py".
5. Click Deploy.

IMPORTANT
---------
This application uses synthetic/mock doctor data, mock prices, mock availability,
and session-only patient/booking data for demonstration. It is NOT a real medical
booking service, does not verify medical credentials, and does not provide medical
diagnosis. A production system needs real authentication/OTP, a secure database,
payment processing, verified doctor data, real-time availability, audit logging,
privacy/security controls, and appropriate regulatory/compliance review.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable
import re
import uuid

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# APP CONFIG
# =============================================================================
st.set_page_config(
    page_title="CareMatch AI | Doctor Booking",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS / UI
# =============================================================================
st.markdown(
    """
    <style>
        :root {
            --ink: #0f172a;
            --muted: #64748b;
            --surface: #ffffff;
            --surface-2: #f8fafc;
            --line: #e2e8f0;
            --accent: #0f766e;
            --accent-2: #2563eb;
            --success: #166534;
            --danger: #b91c1c;
            --warning-bg: #fffbeb;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(20,184,166,0.13), transparent 25%),
                radial-gradient(circle at 92% 5%, rgba(37,99,235,0.12), transparent 24%),
                linear-gradient(180deg, #f8fbff 0%, #f7f9fc 55%, #f4f7fb 100%);
            color: var(--ink) !important;
        }

        .stApp,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp div,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp [data-testid="stMarkdownContainer"] {
            color: var(--ink);
        }

        .stApp [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
            border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: var(--ink) !important;
        }

        .stApp input,
        .stApp textarea,
        .stApp [data-baseweb="select"] > div,
        .stApp [data-baseweb="input"] > div {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-color: #cbd5e1 !important;
        }

        .stApp input::placeholder,
        .stApp textarea::placeholder {
            color: #94a3b8 !important;
            opacity: 1 !important;
        }

        .stApp [data-baseweb="select"] input,
        .stApp [data-baseweb="select"] span {
            color: var(--ink) !important;
        }

        [role="listbox"], [role="option"], [data-baseweb="popover"] {
            color: var(--ink) !important;
            background: #ffffff !important;
        }

        .stButton > button {
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
            background: #ffffff !important;
            color: #0f172a !important;
            font-weight: 700 !important;
            transition: transform 160ms ease, box-shadow 160ms ease,
                        border-color 160ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: #14b8a6 !important;
            box-shadow: 0 8px 22px rgba(20,184,166,0.13);
            color: #0f766e !important;
        }

        .stApp::before, .stApp::after {
            content: "";
            position: fixed;
            width: 30rem;
            height: 30rem;
            border-radius: 999px;
            pointer-events: none;
            filter: blur(72px);
            opacity: 0.15;
            z-index: 0;
            animation: auroraFloat 14s ease-in-out infinite alternate;
        }

        .stApp::before {
            top: -15rem;
            left: 10%;
            background: radial-gradient(circle, rgba(20,184,166,0.85), transparent 65%);
        }

        .stApp::after {
            right: -14rem;
            bottom: -14rem;
            background: radial-gradient(circle, rgba(37,99,235,0.75), transparent 65%);
            animation-delay: -6s;
        }

        @keyframes auroraFloat {
            0% { transform: translate3d(-2%, -1%, 0) scale(1); }
            50% { transform: translate3d(4%, 3%, 0) scale(1.08); }
            100% { transform: translate3d(-1%, 5%, 0) scale(0.96); }
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 1.65rem 1.7rem;
            border-radius: 24px;
            background:
                radial-gradient(circle at 90% 20%, rgba(45,212,191,0.2), transparent 30%),
                linear-gradient(135deg, #0b1730 0%, #123b5d 55%, #0f766e 100%);
            color: white !important;
            box-shadow: 0 18px 50px rgba(15,23,42,0.18);
            margin-bottom: 1.15rem;
            isolation: isolate;
        }

        .hero::before {
            content: "";
            position: absolute;
            inset: -50%;
            background: linear-gradient(
                110deg,
                transparent 35%,
                rgba(255,255,255,0.12) 50%,
                transparent 65%
            );
            animation: heroSweep 7s linear infinite;
            pointer-events: none;
        }

        @keyframes heroSweep {
            from { transform: translateX(-22%) rotate(7deg); }
            to { transform: translateX(22%) rotate(7deg); }
        }

        .hero h1 {
            position: relative;
            z-index: 1;
            color: #ffffff !important;
            margin: 0 0 0.35rem 0;
            font-size: 2.25rem;
            letter-spacing: -0.035em;
            text-shadow: 0 2px 18px rgba(0,0,0,0.17);
        }

        .hero p {
            position: relative;
            z-index: 1;
            color: rgba(255,255,255,0.9) !important;
            margin: 0;
            font-size: 1rem;
        }

        .hero .section-label {
            position: relative;
            z-index: 1;
            color: #99f6e4 !important;
        }

        .section-label {
            font-size: 0.82rem;
            font-weight: 800;
            color: #475569 !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .auth-card, .booking-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 1.25rem;
            box-shadow: 0 14px 36px rgba(15,23,42,0.08);
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.45rem 0 0.75rem 0;
        }

        .chip {
            display: inline-block;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: #ecfeff;
            color: #115e59 !important;
            font-size: 0.78rem;
            font-weight: 750;
            border: 1px solid #cceff0;
        }

        .muted {
            color: #475569 !important;
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .availability {
            margin-top: 0.2rem;
            color: #166534 !important;
            font-weight: 750;
            font-size: 0.82rem;
        }

        .nearby {
            color: #1d4ed8 !important;
            font-size: 0.82rem;
            font-weight: 700;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #e2e8f0 !important;
            border-radius: 18px !important;
            background: rgba(255,255,255,0.84) !important;
            box-shadow: 0 8px 24px rgba(15,23,42,0.045);
            transition: transform 180ms ease, box-shadow 180ms ease,
                        border-color 180ms ease;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-3px);
            border-color: #99f6e4 !important;
            box-shadow:
                0 16px 34px rgba(15,23,42,0.09),
                0 0 0 1px rgba(20,184,166,0.06);
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.88) !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px;
            padding: 0.65rem 0.8rem;
            box-shadow: 0 7px 20px rgba(15,23,42,0.035);
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--ink) !important;
        }

        .booking-pill {
            padding: 0.4rem 0.65rem;
            border-radius: 10px;
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            color: #166534 !important;
            display: inline-block;
            font-weight: 750;
            font-size: 0.82rem;
        }

        .warning-box {
            padding: 0.85rem 1rem;
            border-radius: 14px;
            background: var(--warning-bg);
            border: 1px solid #fde68a;
            color: #92400e !important;
        }

        .footer-note {
            padding: 0.9rem 1rem;
            border-radius: 14px;
            background: rgba(248,250,252,0.9);
            border: 1px solid #e2e8f0;
            color: #64748b !important;
            font-size: 0.8rem;
        }

        @media (max-width: 900px) {
            .hero h1 { font-size: 1.75rem; }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.001ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.001ms !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state() -> None:
    """Initialize state used across login, dashboard, and bookings."""
    defaults = {
        "authenticated": False,
        "patient": {},
        "selected_doctor_id": None,
        "booking_doctor_id": None,
        "bookings": [],
        "page": "Dashboard",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =============================================================================
# DATA LAYER
# =============================================================================
@st.cache_data(show_spinner=False)
def load_doctors() -> pd.DataFrame:
    """Return a self-contained synthetic doctor/clinic dataset."""
    doctors = [
        {
            "doctor_id": "DOC-001",
            "name": "Dr. Ananya Sen",
            "specialty": "Cardiology",
            "location": "Kolkata",
            "language": "English, Bengali, Hindi",
            "years_experience": 17,
            "rating": 4.9,
            "consultation_fee": 900,
            "visit_duration": "30 min",
            "clinic_name": "HeartCare Clinic",
            "clinic_area": "Park Street",
            "clinic_address": "Park Street, Kolkata",
            "distance_km": 3.8,
            "bio": "Cardiologist focused on hypertension, preventive heart care, chest discomfort, and long-term cardiac risk management.",
            "available_days": ["Monday", "Wednesday", "Friday"],
            "time_slots": ["09:30 AM", "11:00 AM", "02:30 PM", "05:00 PM"],
        },
        {
            "doctor_id": "DOC-002",
            "name": "Dr. Rohan Mehta",
            "specialty": "Dermatology",
            "location": "Kolkata",
            "language": "English, Hindi",
            "years_experience": 11,
            "rating": 4.7,
            "consultation_fee": 700,
            "visit_duration": "25 min",
            "clinic_name": "DermaPlus Centre",
            "clinic_area": "Salt Lake",
            "clinic_address": "Sector V, Salt Lake, Kolkata",
            "distance_km": 8.1,
            "bio": "Dermatologist treating acne, eczema, pigmentation, hair concerns, and common inflammatory skin conditions.",
            "available_days": ["Tuesday", "Thursday", "Saturday"],
            "time_slots": ["10:00 AM", "12:30 PM", "03:30 PM", "06:00 PM"],
        },
        {
            "doctor_id": "DOC-003",
            "name": "Dr. Arijit Mukherjee",
            "specialty": "Neurology",
            "location": "Kolkata",
            "language": "English, Bengali",
            "years_experience": 21,
            "rating": 4.8,
            "consultation_fee": 1100,
            "visit_duration": "30 min",
            "clinic_name": "NeuroCare Medical",
            "clinic_area": "Ballygunge",
            "clinic_address": "Ballygunge, Kolkata",
            "distance_km": 5.4,
            "bio": "Neurologist with interests in headaches, migraine, neuropathy, seizures, and movement-related neurological disorders.",
            "available_days": ["Monday", "Tuesday", "Thursday"],
            "time_slots": ["09:00 AM", "11:30 AM", "04:00 PM", "06:30 PM"],
        },
        {
            "doctor_id": "DOC-004",
            "name": "Dr. Priya Nair",
            "specialty": "Endocrinology",
            "location": "Bengaluru",
            "language": "English, Hindi, Malayalam",
            "years_experience": 15,
            "rating": 4.9,
            "consultation_fee": 1000,
            "visit_duration": "30 min",
            "clinic_name": "Metabolic Health Centre",
            "clinic_area": "Koramangala",
            "clinic_address": "Koramangala, Bengaluru",
            "distance_km": 4.6,
            "bio": "Endocrinologist focused on diabetes, thyroid disorders, metabolic health, and hormonal conditions.",
            "available_days": ["Monday", "Wednesday", "Saturday"],
            "time_slots": ["09:30 AM", "12:00 PM", "03:00 PM", "05:30 PM"],
        },
        {
            "doctor_id": "DOC-005",
            "name": "Dr. Vivek Sharma",
            "specialty": "Orthopedics",
            "location": "Delhi",
            "language": "English, Hindi, Punjabi",
            "years_experience": 19,
            "rating": 4.6,
            "consultation_fee": 850,
            "visit_duration": "30 min",
            "clinic_name": "OrthoMotion Hospital",
            "clinic_area": "South Delhi",
            "clinic_address": "Greater Kailash, New Delhi",
            "distance_km": 6.2,
            "bio": "Orthopedic physician managing joint pain, sports injuries, arthritis, back pain, and musculoskeletal rehabilitation.",
            "available_days": ["Tuesday", "Wednesday", "Friday"],
            "time_slots": ["10:00 AM", "01:00 PM", "04:30 PM", "07:00 PM"],
        },
        {
            "doctor_id": "DOC-006",
            "name": "Dr. Meera Iyer",
            "specialty": "Pediatrics",
            "location": "Chennai",
            "language": "English, Tamil, Hindi",
            "years_experience": 13,
            "rating": 4.9,
            "consultation_fee": 750,
            "visit_duration": "25 min",
            "clinic_name": "Little Steps Pediatrics",
            "clinic_area": "Adyar",
            "clinic_address": "Adyar, Chennai",
            "distance_km": 3.2,
            "bio": "Pediatrician supporting childhood nutrition, fever, respiratory complaints, vaccinations, and preventive pediatric care.",
            "available_days": ["Monday", "Thursday", "Saturday"],
            "time_slots": ["09:00 AM", "11:00 AM", "03:30 PM", "06:00 PM"],
        },
        {
            "doctor_id": "DOC-007",
            "name": "Dr. Kunal Verma",
            "specialty": "Gastroenterology",
            "location": "Mumbai",
            "language": "English, Hindi, Marathi",
            "years_experience": 22,
            "rating": 4.8,
            "consultation_fee": 1200,
            "visit_duration": "30 min",
            "clinic_name": "Digestive Health Clinic",
            "clinic_area": "Andheri",
            "clinic_address": "Andheri West, Mumbai",
            "distance_km": 7.1,
            "bio": "Gastroenterologist working with acidity, abdominal discomfort, reflux, liver health, and digestive disorders.",
            "available_days": ["Tuesday", "Thursday", "Friday"],
            "time_slots": ["09:30 AM", "12:00 PM", "04:00 PM", "06:30 PM"],
        },
        {
            "doctor_id": "DOC-008",
            "name": "Dr. Sneha Das",
            "specialty": "Gynecology",
            "location": "Kolkata",
            "language": "English, Bengali, Hindi",
            "years_experience": 16,
            "rating": 4.8,
            "consultation_fee": 900,
            "visit_duration": "30 min",
            "clinic_name": "WomenFirst Clinic",
            "clinic_area": "New Town",
            "clinic_address": "New Town, Kolkata",
            "distance_km": 10.4,
            "bio": "Gynecologist providing care for menstrual health, PCOS, pregnancy support, and routine women's health needs.",
            "available_days": ["Monday", "Wednesday", "Saturday"],
            "time_slots": ["10:00 AM", "12:30 PM", "03:30 PM", "05:30 PM"],
        },
        {
            "doctor_id": "DOC-009",
            "name": "Dr. Aakash Kapoor",
            "specialty": "Psychiatry",
            "location": "Delhi",
            "language": "English, Hindi",
            "years_experience": 14,
            "rating": 4.7,
            "consultation_fee": 1000,
            "visit_duration": "45 min",
            "clinic_name": "MindWell Centre",
            "clinic_area": "Vasant Kunj",
            "clinic_address": "Vasant Kunj, New Delhi",
            "distance_km": 8.8,
            "bio": "Psychiatrist supporting stress-related concerns, sleep difficulties, mood concerns, and general mental wellness care.",
            "available_days": ["Wednesday", "Friday", "Saturday"],
            "time_slots": ["11:00 AM", "02:00 PM", "05:00 PM", "07:00 PM"],
        },
        {
            "doctor_id": "DOC-010",
            "name": "Dr. Sayan Ghosh",
            "specialty": "ENT",
            "location": "Kolkata",
            "language": "English, Bengali, Hindi",
            "years_experience": 12,
            "rating": 4.6,
            "consultation_fee": 650,
            "visit_duration": "25 min",
            "clinic_name": "ENT & Hearing Care",
            "clinic_area": "Barasat",
            "clinic_address": "Barasat, Kolkata",
            "distance_km": 12.5,
            "bio": "ENT specialist for sinus issues, sore throat, ear complaints, allergies, dizziness, and common head-and-neck conditions.",
            "available_days": ["Tuesday", "Thursday", "Sunday"],
            "time_slots": ["09:30 AM", "12:00 PM", "04:00 PM", "06:00 PM"],
        },
        {
            "doctor_id": "DOC-011",
            "name": "Dr. Kavya Rao",
            "specialty": "Pulmonology",
            "location": "Hyderabad",
            "language": "English, Telugu, Hindi",
            "years_experience": 18,
            "rating": 4.8,
            "consultation_fee": 950,
            "visit_duration": "30 min",
            "clinic_name": "Respira Health",
            "clinic_area": "Hitech City",
            "clinic_address": "Hitech City, Hyderabad",
            "distance_km": 5.7,
            "bio": "Pulmonologist focused on asthma, cough, breathing difficulties, COPD, and respiratory wellness.",
            "available_days": ["Monday", "Wednesday", "Friday"],
            "time_slots": ["10:00 AM", "12:30 PM", "03:30 PM", "06:30 PM"],
        },
        {
            "doctor_id": "DOC-012",
            "name": "Dr. Neha Bansal",
            "specialty": "Ophthalmology",
            "location": "Pune",
            "language": "English, Hindi, Marathi",
            "years_experience": 10,
            "rating": 4.7,
            "consultation_fee": 700,
            "visit_duration": "25 min",
            "clinic_name": "ClearView Eye Centre",
            "clinic_area": "Kothrud",
            "clinic_address": "Kothrud, Pune",
            "distance_km": 4.9,
            "bio": "Ophthalmologist providing comprehensive eye examinations, dry-eye care, vision screening, and common eye disease management.",
            "available_days": ["Tuesday", "Thursday", "Saturday"],
            "time_slots": ["09:00 AM", "11:30 AM", "03:00 PM", "05:30 PM"],
        },
        {
            "doctor_id": "DOC-013",
            "name": "Dr. Rahul Khanna",
            "specialty": "Urology",
            "location": "Delhi",
            "language": "English, Hindi, Punjabi",
            "years_experience": 20,
            "rating": 4.8,
            "consultation_fee": 1050,
            "visit_duration": "30 min",
            "clinic_name": "UroCare Specialists",
            "clinic_area": "Rohini",
            "clinic_address": "Rohini, New Delhi",
            "distance_km": 7.7,
            "bio": "Urologist working with urinary symptoms, kidney stone care, prostate health, and common urinary tract conditions.",
            "available_days": ["Monday", "Tuesday", "Friday"],
            "time_slots": ["10:00 AM", "01:00 PM", "04:00 PM", "06:30 PM"],
        },
        {
            "doctor_id": "DOC-014",
            "name": "Dr. Tanvi Chatterjee",
            "specialty": "General Medicine",
            "location": "Kolkata",
            "language": "English, Bengali, Hindi",
            "years_experience": 9,
            "rating": 4.9,
            "consultation_fee": 600,
            "visit_duration": "20 min",
            "clinic_name": "CityCare General Practice",
            "clinic_area": "Dum Dum",
            "clinic_address": "Dum Dum, Kolkata",
            "distance_km": 6.9,
            "bio": "General physician for fever, fatigue, common infections, lifestyle-related concerns, and first-line adult care.",
            "available_days": ["Monday", "Wednesday", "Sunday"],
            "time_slots": ["09:00 AM", "11:00 AM", "03:00 PM", "05:00 PM"],
        },
        {
            "doctor_id": "DOC-015",
            "name": "Dr. Aditya Kulkarni",
            "specialty": "Cardiology",
            "location": "Pune",
            "language": "English, Hindi, Marathi",
            "years_experience": 24,
            "rating": 4.7,
            "consultation_fee": 1100,
            "visit_duration": "30 min",
            "clinic_name": "CardioLife Institute",
            "clinic_area": "Baner",
            "clinic_address": "Baner, Pune",
            "distance_km": 5.8,
            "bio": "Senior cardiologist with experience in coronary risk assessment, hypertension, preventive cardiology, and cardiac follow-up.",
            "available_days": ["Tuesday", "Thursday", "Saturday"],
            "time_slots": ["09:30 AM", "12:00 PM", "04:00 PM", "06:00 PM"],
        },
        {
            "doctor_id": "DOC-016",
            "name": "Dr. Ishita Roy",
            "specialty": "Endocrinology",
            "location": "Kolkata",
            "language": "English, Bengali, Hindi",
            "years_experience": 12,
            "rating": 4.8,
            "consultation_fee": 900,
            "visit_duration": "30 min",
            "clinic_name": "Thyroid & Diabetes Care",
            "clinic_area": "Rajarhat",
            "clinic_address": "Rajarhat, Kolkata",
            "distance_km": 11.2,
            "bio": "Endocrinologist focused on diabetes, insulin resistance, thyroid disorders, and personalized metabolic care.",
            "available_days": ["Monday", "Thursday", "Friday"],
            "time_slots": ["10:30 AM", "01:00 PM", "04:30 PM", "06:30 PM"],
        },
        {
            "doctor_id": "DOC-017",
            "name": "Dr. Harsh Patel",
            "specialty": "Orthopedics",
            "location": "Ahmedabad",
            "language": "English, Hindi, Gujarati",
            "years_experience": 16,
            "rating": 4.7,
            "consultation_fee": 800,
            "visit_duration": "30 min",
            "clinic_name": "Mobility & Joint Clinic",
            "clinic_area": "Satellite",
            "clinic_address": "Satellite, Ahmedabad",
            "distance_km": 4.4,
            "bio": "Orthopedic specialist for knee pain, shoulder injuries, fractures, arthritis, and mobility-focused rehabilitation.",
            "available_days": ["Wednesday", "Friday", "Sunday"],
            "time_slots": ["09:30 AM", "12:00 PM", "03:30 PM", "06:00 PM"],
        },
        {
            "doctor_id": "DOC-018",
            "name": "Dr. Nandini Bose",
            "specialty": "Dermatology",
            "location": "Kolkata",
            "language": "English, Bengali",
            "years_experience": 8,
            "rating": 4.6,
            "consultation_fee": 650,
            "visit_duration": "25 min",
            "clinic_name": "Skin & Hair Studio",
            "clinic_area": "Gariahat",
            "clinic_address": "Gariahat, Kolkata",
            "distance_km": 5.2,
            "bio": "Dermatologist focused on acne, pigmentation, sensitive skin, hair fall, and routine dermatological care.",
            "available_days": ["Monday", "Tuesday", "Saturday"],
            "time_slots": ["10:00 AM", "12:00 PM", "04:00 PM", "06:00 PM"],
        },
    ]

    df = pd.DataFrame(doctors)
    df["available_days"] = df["available_days"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df["time_slots"] = df["time_slots"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    return df


# =============================================================================
# ML LAYER
# =============================================================================
@st.cache_data(show_spinner=False)
def build_tfidf_index(df: pd.DataFrame) -> tuple[TfidfVectorizer, object]:
    """Build the cached TF-IDF matrix for doctor profile documents."""
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
        + working["clinic_area"].fillna("").astype(str)
        + " "
        + working["available_days"].apply(
            lambda days: " ".join(days) if isinstance(days, list) else str(days)
        )
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(working["document"])
    return vectorizer, matrix


@st.cache_data(show_spinner=False)
def recommend_doctors(
    df: pd.DataFrame,
    query_text: str,
    preferred_language: str,
    location: str,
    min_rating: float,
) -> pd.DataFrame:
    """
    Hybrid recommendation:
        65% TF-IDF cosine similarity
        10% language preference
        10% location preference
        10% rating
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
        result["language"]
        .str.contains(
            preferred_language,
            case=False,
            na=False,
            regex=False,
        )
        .astype(float),
    )

    result["location_match"] = np.where(
        location == "Any location",
        0.0,
        result["location"].str.casefold().eq(location.casefold()).astype(float),
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

    if not clean_query:
        result["match_score"] = (
            0.70 * result["rating_score"]
            + 0.30 * result["experience_score"]
        )

    result = result[result["rating"] >= min_rating].copy()
    result = result.sort_values(
        by=["match_score", "rating", "years_experience"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    result["match_percentage"] = np.clip(
        result["match_score"] * 100.0,
        0.0,
        100.0,
    )
    return result


# =============================================================================
# HELPERS
# =============================================================================
def parse_languages(series: pd.Series) -> list[str]:
    """Extract unique language labels from comma-separated values."""
    languages: set[str] = set()
    for value in series.dropna().astype(str):
        languages.update(part.strip() for part in value.split(",") if part.strip())
    return sorted(languages)


def render_chip_row(items: Iterable[str]) -> None:
    chips = "".join(
        f'<span class="chip">{item}</span>' for item in items
    )
    st.markdown(
        f'<div class="chip-row">{chips}</div>',
        unsafe_allow_html=True,
    )


def build_query(
    target_text: str,
    specialty: str,
    preferred_language: str,
    location: str,
) -> str:
    pieces = []

    if target_text.strip():
        pieces.append(target_text.strip())

    if specialty != "Any specialty":
        pieces.append(specialty)

    if preferred_language != "Any language":
        pieces.append(preferred_language)

    if location != "Any location":
        pieces.append(location)

    return " ".join(pieces)


def is_valid_phone(phone: str) -> bool:
    """Basic demo validation; use a real OTP/auth provider in production."""
    digits = re.sub(r"\D", "", phone)
    return 10 <= len(digits) <= 15


def get_doctor(df: pd.DataFrame, doctor_id: str) -> pd.Series | None:
    rows = df[df["doctor_id"] == doctor_id]
    if rows.empty:
        return None
    return rows.iloc[0]


def next_available_dates(
    doctor: pd.Series,
    start_date: date,
    days_ahead: int = 45,
) -> list[date]:
    """Generate future calendar dates that match the doctor's working days."""
    dates = []
    allowed_days = set(doctor["available_days"])

    for offset in range(days_ahead + 1):
        candidate = start_date + timedelta(days=offset)
        if candidate.strftime("%A") in allowed_days:
            dates.append(candidate)

    return dates


def format_currency(amount: float) -> str:
    return f"₹{amount:,.0f}"


# =============================================================================
# LOGIN / PATIENT PROFILE PAGE
# =============================================================================
def render_login_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="section-label">CARE MATCH AI</div>
            <h1>Patient Sign In & Profile</h1>
            <p>Enter your basic details to personalize recommendations and book a visit.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown("### 👤 Patient details")
        st.caption(
            "Demo mode: this profile is stored only in Streamlit session state."
        )

        name = st.text_input(
            "Full name",
            placeholder="Enter your name",
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25,
            step=1,
        )

        sex = st.selectbox(
            "Sex",
            ["Prefer not to say", "Female", "Male", "Other"],
        )

        location_options = [
            "Kolkata",
            "Delhi",
            "Bengaluru",
            "Chennai",
            "Mumbai",
            "Pune",
            "Hyderabad",
            "Ahmedabad",
        ]
        location = st.selectbox(
            "City / Location",
            location_options,
        )

        locality = st.text_input(
            "Nearby area / locality",
            placeholder="Example: Barasat, Salt Lake, New Town",
        )

        phone = st.text_input(
            "Phone number",
            placeholder="10-digit mobile number",
            max_chars=15,
        )

        agree = st.checkbox(
            "I understand this is a demo booking application using mock data.",
            value=False,
        )

        if st.button(
            "Continue to CareMatch →",
            type="primary",
            use_container_width=True,
        ):
            if not name.strip():
                st.error("Please enter your name.")
            elif not is_valid_phone(phone):
                st.error("Enter a valid phone number (10–15 digits).")
            elif not agree:
                st.warning("Please accept the demo notice to continue.")
            else:
                st.session_state["patient"] = {
                    "name": name.strip(),
                    "age": int(age),
                    "sex": sex,
                    "location": location,
                    "locality": locality.strip(),
                    "phone": re.sub(r"\D", "", phone),
                }
                st.session_state["authenticated"] = True
                st.session_state["page"] = "Dashboard"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("### What happens next?")
        st.markdown(
            """
            **1. Personalized search**  
            Use symptoms, specialty, language, rating, and location.

            **2. Nearby clinic details**  
            See clinic/area, address, availability, and a mock distance estimate.

            **3. Book a visit**  
            Choose an available date and time slot and see the consultation fee.

            **4. My Appointments**  
            Your demo bookings appear in one place during the current session.
            """
        )
        st.markdown(
            '<div class="warning-box"><b>Privacy note:</b> This demo does not send your phone number to an external service. For a real product, use secure authentication, encryption, access control, and a compliant backend.</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# BOOKING MODAL-LIKE SECTION
# =============================================================================
def render_booking_section(doctors_df: pd.DataFrame) -> None:
    doctor_id = st.session_state.get("booking_doctor_id")
    doctor = get_doctor(doctors_df, doctor_id) if doctor_id else None

    if doctor is None:
        return

    st.divider()
    st.markdown("### 📅 Book an appointment")

    with st.container(border=True):
        top_left, top_right = st.columns([2, 1])

        with top_left:
            st.markdown(f"#### {doctor['name']}")
            st.caption(
                f"{doctor['specialty']} · {doctor['clinic_name']} · {doctor['clinic_area']}"
            )
            st.markdown(
                f'<span class="booking-pill">Consultation fee: {format_currency(doctor["consultation_fee"])}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="nearby">📍 {doctor["clinic_address"]} · Mock distance: ~{doctor["distance_km"]:.1f} km</div>',
                unsafe_allow_html=True,
            )

        with top_right:
            if st.button("Close booking", use_container_width=True):
                st.session_state["booking_doctor_id"] = None
                st.rerun()

        patient = st.session_state["patient"]

        form_left, form_right = st.columns(2)

        valid_dates = next_available_dates(
            doctor,
            start_date=date.today() + timedelta(days=1),
        )

        if not valid_dates:
            st.warning("No future availability is configured for this doctor.")
            return

        with form_left:
            selected_date = st.selectbox(
                "Visit date",
                options=valid_dates,
                format_func=lambda d: f"{d.strftime('%A, %d %B %Y')}",
                key=f"booking_date_{doctor_id}",
            )

        with form_right:
            selected_time = st.selectbox(
                "Visit time",
                options=doctor["time_slots"],
                key=f"booking_time_{doctor_id}",
            )

        visit_reason = st.text_input(
            "Appointment note (optional)",
            placeholder="Example: follow-up, general consultation, recurring headache",
            key=f"booking_reason_{doctor_id}",
        )

        st.markdown("**Patient**")
        render_chip_row(
            [
                patient["name"],
                f"Age {patient['age']}",
                patient["sex"],
                patient["location"],
            ]
        )

        if st.button(
            f"Confirm booking · {format_currency(doctor['consultation_fee'])}",
            type="primary",
            use_container_width=True,
        ):
            booking = {
                "booking_id": f"APT-{uuid.uuid4().hex[:8].upper()}",
                "doctor_id": doctor["doctor_id"],
                "doctor_name": doctor["name"],
                "specialty": doctor["specialty"],
                "clinic_name": doctor["clinic_name"],
                "clinic_address": doctor["clinic_address"],
                "date": selected_date.isoformat(),
                "time": selected_time,
                "fee": float(doctor["consultation_fee"]),
                "patient_name": patient["name"],
                "patient_phone": patient["phone"],
                "note": visit_reason.strip(),
                "status": "Confirmed",
            }

            # Basic session-level duplicate guard for the same doctor/date/time.
            duplicate = any(
                item["doctor_id"] == booking["doctor_id"]
                and item["date"] == booking["date"]
                and item["time"] == booking["time"]
                and item["status"] == "Confirmed"
                for item in st.session_state["bookings"]
            )

            if duplicate:
                st.warning(
                    "That doctor/date/time is already booked in this demo session. "
                    "Please choose another slot."
                )
            else:
                st.session_state["bookings"].append(booking)
                st.session_state["booking_doctor_id"] = None
                st.success(
                    f"Appointment confirmed for {selected_date.strftime('%d %b %Y')} at {selected_time}."
                )
                st.rerun()


# =============================================================================
# MY APPOINTMENTS PAGE
# =============================================================================
def render_appointments_page(doctors_df: pd.DataFrame) -> None:
    patient = st.session_state["patient"]

    st.markdown(
        """
        <div class="hero">
            <div class="section-label">PATIENT PORTAL</div>
            <h1>My Appointments</h1>
            <p>Review your confirmed demo appointments and visit details.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state["bookings"]:
        st.info("No appointments booked yet. Go to Dashboard and choose a doctor.")
        return

    for booking in st.session_state["bookings"]:
        with st.container(border=True):
            left, mid, right = st.columns([2.2, 1.5, 1.1])

            with left:
                st.markdown(f"### 🩺 {booking['doctor_name']}")
                st.caption(
                    f"{booking['specialty']} · {booking['clinic_name']}"
                )
                st.write(booking["clinic_address"])
                st.caption(f"Booking ID: {booking['booking_id']}")

            with mid:
                st.markdown("**Visit**")
                st.write(
                    date.fromisoformat(booking["date"]).strftime("%A, %d %B %Y")
                )
                st.write(f"⏰ {booking['time']}")
                st.write(f"💳 {format_currency(booking['fee'])}")

            with right:
                st.markdown("**Status**")
                st.success(booking["status"])

                if st.button(
                    "Cancel",
                    key=f"cancel_{booking['booking_id']}",
                    use_container_width=True,
                ):
                    for item in st.session_state["bookings"]:
                        if item["booking_id"] == booking["booking_id"]:
                            item["status"] = "Cancelled"
                    st.rerun()

        if booking.get("note"):
            st.caption(f"Note: {booking['note']}")

    st.markdown(
        '<div class="footer-note">Demo reminder: bookings exist only in the current Streamlit session. Refreshing/restarting the app can clear them.</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR AFTER LOGIN
# =============================================================================
def render_sidebar(doctors_df: pd.DataFrame) -> tuple[str, str, str, str, str, float]:
    patient = st.session_state["patient"]

    available_languages = ["Any language"] + parse_languages(doctors_df["language"])
    available_locations = ["Any location"] + sorted(
        doctors_df["location"].unique()
    )
    available_specialties = ["Any specialty"] + sorted(
        doctors_df["specialty"].unique()
    )

    with st.sidebar:
        st.markdown("## 👤 Patient")
        st.markdown(f"**{patient['name']}**")
        st.caption(
            f"Age {patient['age']} · {patient['sex']} · {patient['location']}"
        )
        if patient.get("locality"):
            st.caption(f"📍 {patient['locality']}")
        st.caption(f"📱 {patient['phone']}")

        st.divider()

        page = st.radio(
            "Navigation",
            ["Dashboard", "My Appointments"],
            index=0 if st.session_state["page"] == "Dashboard" else 1,
        )
        st.session_state["page"] = page

        if page == "Dashboard":
            st.markdown("### 🔎 Patient preferences")

            preferred_language = st.selectbox(
                "Preferred Language",
                available_languages,
                index=0,
            )

            location = st.selectbox(
                "Location",
                available_locations,
                index=(
                    available_locations.index(patient["location"])
                    if patient["location"] in available_locations
                    else 0
                ),
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

            if st.button("Edit patient profile", use_container_width=True):
                st.session_state["authenticated"] = False
                st.session_state["page"] = "Dashboard"
                st.rerun()

            if st.button("Log out", use_container_width=True):
                st.session_state["authenticated"] = False
                st.session_state["patient"] = {}
                st.session_state["selected_doctor_id"] = None
                st.session_state["booking_doctor_id"] = None
                st.session_state["page"] = "Dashboard"
                st.rerun()

            st.markdown(
                '<div class="small-help">Age and sex are shown as patient-profile information in this demo; they are not used in the recommendation score.</div>',
                unsafe_allow_html=True,
            )

            return (
                preferred_language,
                location,
                specialty,
                target_text,
                "Dashboard",
                min_rating,
            )

        return (
            "Any language",
            "Any location",
            "Any specialty",
            "",
            "My Appointments",
            3.0,
        )


# =============================================================================
# DASHBOARD
# =============================================================================
def render_dashboard(
    doctors_df: pd.DataFrame,
    preferred_language: str,
    location: str,
    specialty: str,
    target_text: str,
    min_rating: float,
) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="section-label">CARE MATCH AI</div>
            <h1>Personalized Doctor Recommendation</h1>
            <p>Search, compare, view nearby clinic details, and book an available visit time.</p>
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
            "content-based matching. Results currently use quality-based fallback ranking."
        )

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
            f"{recommendations['rating'].mean():.2f}"
            if not recommendations.empty
            else "—",
        )
    with metric_d:
        st.metric(
            "Top match",
            (
                f"{recommendations['match_percentage'].iloc[0]:.0f}%"
                if not recommendations.empty
                else "—"
            ),
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
            options=parse_languages(doctors_df["language"]),
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
        selected_day = st.selectbox(
            "Available Day",
            [
                "Any day",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )

    filtered = recommendations.copy()

    if selected_specialties:
        filtered = filtered[
            filtered["specialty"].isin(selected_specialties)
        ]

    if selected_languages:
        language_mask = filtered["language"].apply(
            lambda value: any(
                language.casefold() in value.casefold()
                for language in selected_languages
            )
        )
        filtered = filtered[language_mask]

    filtered = filtered[
        filtered["years_experience"] >= min_experience
    ]

    if selected_day != "Any day":
        filtered = filtered[
            filtered["available_days"].apply(
                lambda days: selected_day in days
                if isinstance(days, list)
                else False
            )
        ]

    view_col1, view_col2 = st.columns([1, 3])
    with view_col1:
        top_n = st.slider(
            "Show top N",
            min_value=1,
            max_value=12,
            value=6,
            step=1,
        )

    with view_col2:
        sort_mode = st.selectbox(
            "Sort cards by",
            ["Match score", "Rating", "Experience"],
            index=0,
        )

    if sort_mode == "Rating":
        filtered = filtered.sort_values(
            ["rating", "match_score"],
            ascending=[False, False],
        )
    elif sort_mode == "Experience":
        filtered = filtered.sort_values(
            ["years_experience", "match_score"],
            ascending=[False, False],
        )
    else:
        filtered = filtered.sort_values(
            ["match_score", "rating", "years_experience"],
            ascending=[False, False, False],
        )

    filtered = filtered.head(top_n).reset_index(drop=True)

    if filtered.empty:
        st.warning(
            "No doctors match the current filters. Try lowering the rating/experience "
            "filters, changing the location, or clearing filters."
        )
        return

    st.markdown(f"### {len(filtered)} Matching Doctors")

    for start in range(0, len(filtered), 2):
        row = st.columns(2)

        for offset, column in enumerate(row):
            idx = start + offset
            if idx >= len(filtered):
                continue

            doctor = filtered.iloc[idx]

            with column:
                with st.container(border=True):
                    title_col, score_col = st.columns([3.1, 1.25])

                    with title_col:
                        st.markdown(f"#### 🩺 {doctor['name']}")
                        st.caption(
                            f"{doctor['doctor_id']} · {doctor['specialty']} · {doctor['location']}"
                        )

                    with score_col:
                        st.metric(
                            "Match",
                            f"{doctor['match_percentage']:.0f}%",
                        )

                    st.progress(
                        int(round(float(doctor["match_percentage"]))),
                        text="Recommendation strength",
                    )

                    render_chip_row(
                        [
                            f"⭐ {doctor['rating']:.1f}/5",
                            f"🎓 {int(doctor['years_experience'])} yrs",
                            f"💳 {format_currency(doctor['consultation_fee'])}",
                        ]
                    )

                    st.markdown(
                        f'<div class="muted">{doctor["bio"]}</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f'<div class="nearby">📍 {doctor["clinic_name"]} · {doctor["clinic_area"]} · ~{doctor["distance_km"]:.1f} km (demo estimate)</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f'<div class="availability">● Available: {", ".join(doctor["available_days"])}</div>',
                        unsafe_allow_html=True,
                    )

                    action_a, action_b = st.columns(2)

                    with action_a:
                        if st.button(
                            "View details",
                            key=f"details_{doctor['doctor_id']}",
                            use_container_width=True,
                        ):
                            st.session_state["selected_doctor_id"] = doctor["doctor_id"]
                            st.session_state["booking_doctor_id"] = None
                            st.rerun()

                    with action_b:
                        if st.button(
                            "📅 Book visit",
                            key=f"book_{doctor['doctor_id']}",
                            type="primary",
                            use_container_width=True,
                        ):
                            st.session_state["booking_doctor_id"] = doctor["doctor_id"]
                            st.session_state["selected_doctor_id"] = doctor["doctor_id"]
                            st.rerun()

    # Details section
    selected_id = st.session_state.get("selected_doctor_id")
    selected = get_doctor(doctors_df, selected_id) if selected_id else None

    if selected is not None:
        st.divider()
        st.markdown("### 👨‍⚕️ Doctor details")
        left, right = st.columns([2.1, 1])

        with left:
            st.markdown(f"## {selected['name']}")
            st.write(
                f"**{selected['specialty']}** · "
                f"{selected['location']} · "
                f"{selected['doctor_id']}"
            )

            render_chip_row(
                [
                    f"⭐ {selected['rating']:.1f}/5 rating",
                    f"🧠 {int(selected['years_experience'])} years",
                    f"💳 {format_currency(selected['consultation_fee'])}",
                    f"⏱️ {selected['visit_duration']}",
                ]
            )

            st.write(selected["bio"])

        with right:
            st.markdown("**📍 Nearby clinic details**")
            st.write(selected["clinic_name"])
            st.write(selected["clinic_address"])
            st.write(
                f"Demo distance estimate: ~{selected['distance_km']:.1f} km"
            )
            st.caption(
                "Distance is synthetic demo data, not live GPS/directions."
            )

            st.markdown("**🗓️ Availability**")
            for day in selected["available_days"]:
                st.write(f"✅ {day}")

            if st.button(
                "Book this doctor",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["booking_doctor_id"] = selected["doctor_id"]
                st.rerun()

    render_booking_section(doctors_df)

    with st.expander("🔍 How the recommendation + booking engine works"):
        st.markdown(
            """
            **Recommendation**

            `0.65 × TF-IDF cosine similarity`
            `+ 0.10 × language preference`
            `+ 0.10 × location preference`
            `+ 0.10 × normalized rating`
            `+ 0.05 × normalized experience`

            **Booking**

            - The mock doctor dataset contains consultation price, clinic details,
              available weekdays, and sample time slots.
            - The date selector only offers future dates matching the doctor's
              configured working days.
            - Confirmed bookings are stored in `st.session_state`, so they are
              visible in **My Appointments** during the current session.
            - Production deployment should replace this with a transactional
              database and server-side booking/availability logic.
            """
        )


# =============================================================================
# MAIN
# =============================================================================
doctors_df = load_doctors()

if not st.session_state["authenticated"]:
    render_login_page()
else:
    (
        preferred_language,
        location,
        specialty,
        target_text,
        current_page,
        min_rating,
    ) = render_sidebar(doctors_df)

    if current_page == "My Appointments":
        render_appointments_page(doctors_df)
    else:
        render_dashboard(
            doctors_df=doctors_df,
            preferred_language=preferred_language,
            location=location,
            specialty=specialty,
            target_text=target_text,
            min_rating=min_rating,
        )

    st.markdown(
        """
        <div class="footer-note">
            <b>Demo safety notice:</b> Synthetic doctor data, prices, clinic details,
            distances, and availability are for software demonstration only.
            This application is not a medical diagnosis service and does not verify
            doctors or provide emergency care.
        </div>
        """,
        unsafe_allow_html=True,
    )
