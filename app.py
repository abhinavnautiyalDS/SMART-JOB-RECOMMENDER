"""
Smart Job Recommender - Streamlit Cloud Deployment Version
=============================================

Fixed and improved version with robust apply-link handling and other bug fixes.

Main fixes:
- Robust fallback for application links (checks many possible fields and company/website fallbacks)
- Fix scoping bug where `apply_link` was referenced but not defined for internships
- Safer PDF text extraction (handles None)
- Safer duplicate removal (handles missing company/title)
- Proper URL encoding for Google search fallback links
- Minor defensive checks and clearer logging/messages

Author: AI Assistant (updated)
Version: 2.1 (Bugfix)
"""

import streamlit as st
import pandas as pd
import requests
import time
import json
import os
from typing import List, Dict, Any
import tempfile
from urllib.parse import quote_plus

# Import AI libraries with error handling
try:
    import google.generativeai as genai_old
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.error("Google Generative AI not available. Please install: pip install google-generativeai")

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    st.error("PyPDF not available. Please install: pip install pypdf")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="Smart Job Recommender",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CORE RAG SYSTEM CLASS
# ============================================================================

class SmartJobRecommenderRAG:
    """Enhanced RAG system for job recommendations using Gemini Flash"""

    def __init__(self):
        self.gemini_client = None
        self.initialize_gemini()

    def initialize_gemini(self) -> bool:
        """Initialize Gemini AI client"""
        try:
            # Get API key from Streamlit secrets or environment
            try:
                gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            except Exception:
                gemini_key = os.environ.get("GEMINI_API_KEY")

            if gemini_key and GEMINI_AVAILABLE:
                # Use Gemini 1.5 Flash (free tier)
                try:
                    genai_old.configure(api_key=gemini_key)
                    # note: API surface may differ depending on package version
                    self.gemini_client = genai_old.GenerativeModel('gemini-1.5-flash')
                    st.success("AI system initialized successfully")
                    return True
                except Exception as e:
                    st.error(f"❌ Error initializing Gemini client: {e}")
                    return False
            else:
                st.error("❌ Gemini API key required. Please add GEMINI_API_KEY to your Streamlit secrets.")
                return False
        except Exception as e:
            st.error(f"❌ Error initializing Gemini: {e}")
            return False

    def load_document_with_pypdf(self, uploaded_file) -> List:
        """Load PDF document using PyPDF (defensive against None pages)"""
        if not PYPDF_AVAILABLE:
            st.error("PyPDF not available for document processing")
            return []

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name

            # Read PDF
            documents = []
            reader = PdfReader(temp_file_path)

            for page_num, page in enumerate(reader.pages):
                # Some pages may return None from extract_text
                text = page.extract_text() if hasattr(page, 'extract_text') else None
                if text and text.strip():
                    doc_obj = type('Document', (), {
                        'page_content': text,
                        'metadata': {'page': page_num + 1}
                    })()
                    documents.append(doc_obj)

            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

            st.success(f"✅ Loaded {len(documents)} pages from PDF")
            return documents

        except Exception as e:
            st.error(f"❌ Error loading PDF: {e}")
            return []

    def call_direct_gemini(self, prompt: str) -> Dict[str, Any]:
        """Call Gemini directly for text analysis"""
        if not self.gemini_client:
            return {"skills": [], "job_interests": [], "experience_level": "entry"}

        try:
            response = self.gemini_client.generate_content(prompt)
            # Response parsing may differ depending on the client version
            response_text = getattr(response, 'text', str(response))

            # Parse response
            skills = []
            job_interests = []
            experience_level = "entry"

            lines = response_text.split('\n')
            for line in lines:
                if line.startswith('SKILLS:'):
                    skills_text = line.replace('SKILLS:', '').strip()
                    skills = [s.strip() for s in skills_text.split(',') if s.strip()]
                elif line.startswith('JOB_INTERESTS:'):
                    interests_text = line.replace('JOB_INTERESTS:', '').strip()
                    job_interests = [i.strip() for i in interests_text.split(',') if i.strip()]
                elif line.startswith('EXPERIENCE_LEVEL:'):
                    experience_level = line.replace('EXPERIENCE_LEVEL:', '').strip().lower()

            return {
                "skills": skills[:10],  # Limit to top 10 skills
                "job_interests": job_interests[:5],  # Limit to top 5 interests
                "experience_level": experience_level
            }

        except Exception as e:
            st.error(f"❌ Error calling Gemini: {e}")
            return {"skills": [], "job_interests": [], "experience_level": "entry"}

    # -------------------- New helper utilities --------------------
    def sanitize_link(self, link: Any) -> str:
        """Return cleaned link or empty string if invalid"""
        if not link:
            return ""
        try:
            cleaned = str(link).strip()
            if cleaned == "#" or cleaned.lower() == "none":
                return ""
            return cleaned
        except Exception:
            return ""

    def get_best_apply_link(self, job: Dict[str, Any], response_data: Dict[str, Any] = None) -> str:
        """Try many possible fields for an application/website link.
        If nothing found, return empty string (UI will show a search fallback).
        """
        candidates = [
            'apply_link', 'application_link', 'apply_url', 'url', 'link', 'job_posting_url',
            'canonical_url', 'destination', 'job_link', 'website', 'company_website', 'company_url'
        ]

        for key in candidates:
            if key in job:
                s = self.sanitize_link(job.get(key))
                if s:
                    return s

        # Try nested known patterns
        # Some SerpAPI responses include a top-level website link or source URL
        if response_data:
            maybe = response_data.get('website_link') or response_data.get('website')
            if maybe:
                s = self.sanitize_link(maybe)
                if s:
                    return s
            # Try search_metadata or related
            search_meta = response_data.get('search_metadata') if isinstance(response_data, dict) else None
            if search_meta and isinstance(search_meta, dict):
                for k in ['source', 'website', 'source_url']:
                    if k in search_meta:
                        s = self.sanitize_link(search_meta.get(k))
                        if s:
                            return s

        return ""

    # ---------------------------------------------------------------
    def search_jobs_with_serpapi(self, skills: List[str], job_interests: List[str]) -> Dict[str, List]:
        """Search jobs from multiple sources using SerpAPI"""
        try:
            # Get SerpAPI key
            try:
                serpapi_key = st.secrets.get("SERPAPI_API_KEY") or os.environ.get("SERPAPI_API_KEY")
            except Exception:
                serpapi_key = os.environ.get("SERPAPI_API_KEY")

            if not serpapi_key:
                st.error("❌ SerpAPI key required for job search. Please add SERPAPI_API_KEY to your Streamlit secrets.")
                return {"jobs": [], "internships": [], "search_queries": []}

            # Create search queries
            search_queries = []

            if skills:
                primary_skills = skills[:2]
                for skill in primary_skills:
                    search_queries.extend([
                        f"{skill} developer jobs",
                        f"{skill} engineer jobs"
                    ])

            if job_interests:
                for interest in job_interests[:2]:
                    search_queries.append(f"{interest} jobs")

            if not search_queries:
                search_queries = ["software developer jobs", "python developer jobs"]

            all_jobs = []
            all_internships = []

            # Search using Google Jobs engine (most reliable)
            sources = [
                {"engine": "google_jobs", "source_name": "Google Jobs"},
                {"engine": "google_jobs", "source_name": "LinkedIn (via Google)"}
            ]

            for source in sources[:2]:
                for query in search_queries[:3]:
                    try:
                        url = "https://serpapi.com/search"

                        enhanced_query = query
                        if "LinkedIn" in source["source_name"]:
                            enhanced_query += " site:linkedin.com"

                        params = {
                            "engine": source["engine"],
                            "q": enhanced_query,
                            "location": "United States",
                            "api_key": serpapi_key,
                            "num": 8
                        }

                        st.info(f"🔍 Searching {source['source_name']} for '{query}'...")

                        response = requests.get(url, params=params, timeout=15)
                        data = response.json()

                        jobs_key = "jobs_results"

                        if jobs_key in data and data[jobs_key]:
                            for job in data[jobs_key]:
                                # Get best available application link, fallback to website link
                                apply_link = self.get_best_apply_link(job, response_data=data)

                                job_data = {
                                    "title": job.get("title", "Unknown Title") or "Unknown Title",
                                    "company": job.get("company_name", job.get("company", "Unknown Company")) or "Unknown Company",
                                    "location": job.get("location", "Unknown Location") or "Unknown Location",
                                    "description": job.get("description", job.get("snippet", "No description")) or "No description",
                                    "apply_link": apply_link,
                                    "salary": job.get("salary", job.get("salary_range", "Not specified")) or "Not specified",
                                    "source": source["source_name"],
                                    "match_score": self.calculate_match_score(skills, job.get("description", job.get("snippet", ""))),
                                    "required_skills": self.extract_skills_from_description(job.get("description", job.get("snippet", "")))
                                }

                                title_lower = (job_data["title"] or "").lower()
                                if any(word in title_lower for word in ["intern", "internship", "trainee"]):
                                    all_internships.append(job_data)
                                else:
                                    all_jobs.append(job_data)

                        time.sleep(0.5)

                    except Exception as e:
                        st.warning(f"⚠️ Error searching {source['source_name']}: {str(e)}")
                        continue

            # Remove duplicates and sort by match score
            unique_jobs = self.remove_duplicates(all_jobs)
            unique_internships = self.remove_duplicates(all_internships)

            unique_jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            unique_internships.sort(key=lambda x: x.get("match_score", 0), reverse=True)

            st.success(f"✅ Found {len(unique_jobs)} jobs and {len(unique_internships)} internships from multiple sources")

            return {
                "jobs": unique_jobs[:15],
                "internships": unique_internships[:8],
                "search_queries": search_queries[:6]
            }

        except Exception as e:
            st.error(f"❌ Error with job search: {e}")
            return {"jobs": [], "internships": [], "search_queries": []}

    def search_jobs_with_serpapi_location(self, skills: List[str], job_interests: List[str], location: str) -> Dict[str, List]:
        """Search jobs with location preference"""
        try:
            # Get SerpAPI key
            try:
                serpapi_key = st.secrets.get("SERPAPI_API_KEY") or os.environ.get("SERPAPI_API_KEY")
            except Exception:
                serpapi_key = os.environ.get("SERPAPI_API_KEY")

            if not serpapi_key:
                st.error("❌ SerpAPI key required for job search. Please add SERPAPI_API_KEY to your Streamlit secrets.")
                return {"jobs": [], "internships": [], "search_queries": []}

            # Create location-specific search queries
            search_queries = []

            if skills:
                primary_skills = skills[:2]
                for skill in primary_skills:
                    search_queries.extend([
                        f"{skill} jobs {location}",
                        f"{skill} developer {location}"
                    ])

            if job_interests:
                for interest in job_interests[:2]:
                    search_queries.append(f"{interest} {location}")

            sources = [
                {"engine": "google_jobs", "source_name": "Google Jobs"},
                {"engine": "google_jobs", "source_name": "Indeed (via Google)"}
            ]

            # Add India-specific searches if location suggests India
            if any(word in location.lower() for word in ["india", "mumbai", "delhi", "bangalore", "chennai", "pune", "hyderabad"]):
                search_queries.append(f"internship {location}")

            all_jobs = []
            all_internships = []

            for source in sources:
                for query in search_queries[:3]:
                    try:
                        url = "https://serpapi.com/search"

                        enhanced_query = query
                        if "Indeed" in source["source_name"]:
                            enhanced_query += " site:indeed.com"

                        params = {
                            "engine": source["engine"],
                            "q": enhanced_query,
                            "location": location,
                            "api_key": serpapi_key,
                            "num": 8
                        }

                        st.info(f"🔍 Searching {source['source_name']} in {location} for '{query}'...")

                        response = requests.get(url, params=params, timeout=15)
                        data = response.json()

                        jobs_key = "jobs_results"

                        if jobs_key in data and data[jobs_key]:
                            for job in data[jobs_key]:
                                apply_link = self.get_best_apply_link(job, response_data=data)

                                job_data = {
                                    "title": job.get("title", "Unknown Title") or "Unknown Title",
                                    "company": job.get("company_name", job.get("company", "Unknown Company")) or "Unknown Company",
                                    "location": job.get("location", location) or location,
                                    "description": job.get("description", job.get("snippet", "No description")) or "No description",
                                    "apply_link": apply_link,
                                    "salary": job.get("salary", job.get("salary_range", "Not specified")) or "Not specified",
                                    "source": source["source_name"],
                                    "match_score": self.calculate_match_score(skills, job.get("description", job.get("snippet", ""))),
                                    "required_skills": self.extract_skills_from_description(job.get("description", job.get("snippet", "")))
                                }

                                title_lower = (job_data["title"] or "").lower()
                                if any(word in title_lower for word in ["intern", "internship", "trainee"]):
                                    all_internships.append(job_data)
                                else:
                                    all_jobs.append(job_data)

                        time.sleep(0.5)

                    except Exception as e:
                        st.warning(f"⚠️ Error searching {source['source_name']}: {str(e)}")
                        continue

            # Remove duplicates and sort
            unique_jobs = self.remove_duplicates(all_jobs)
            unique_internships = self.remove_duplicates(all_internships)

            unique_jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            unique_internships.sort(key=lambda x: x.get("match_score", 0), reverse=True)

            st.success(f"✅ Found {len(unique_jobs)} jobs and {len(unique_internships)} internships in {location}")

            return {
                "jobs": unique_jobs[:12],
                "internships": unique_internships[:8],
                "search_queries": search_queries[:4]
            }

        except Exception as e:
            st.error(f"❌ Error with location-based job search: {e}")
            return {"jobs": [], "internships": [], "search_queries": []}

    def calculate_match_score(self, user_skills: List[str], job_description: str) -> int:
        """Calculate match percentage between user skills and job requirements"""
        if not user_skills or not job_description:
            return 0

        job_desc_lower = (job_description or "").lower()
        matched_skills = 0

        for skill in user_skills:
            if skill and skill.lower() in job_desc_lower:
                matched_skills += 1

        try:
            return int((matched_skills / len(user_skills)) * 100) if user_skills else 0
        except Exception:
            return 0

    def extract_skills_from_description(self, description: str) -> List[str]:
        """Extract skills from job description"""
        if not description:
            return []

        # Extended list of technical skills
        tech_skills = [
            # Programming Languages
            "python", "java", "javascript", "typescript", "c++", "c#", "php", "ruby", "go", "rust", "swift", "kotlin",

            # Web Technologies
            "react", "angular", "vue.js", "node.js", "express", "django", "flask", "fastapi", "spring boot",
            "html", "css", "sass", "bootstrap", "tailwind", "jquery",

            # Databases
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle", "sqlite",

            # Cloud & DevOps
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "github", "gitlab",
            "terraform", "ansible", "linux", "bash",

            # Data & Analytics
            "machine learning", "ai", "data analysis", "pandas", "numpy", "tensorflow", "pytorch",
            "tableau", "power bi", "excel", "r", "spark",

            # Soft Skills
            "communication", "leadership", "project management", "agile", "scrum", "problem solving",
            "teamwork", "time management"
        ]

        found_skills = []
        desc_lower = description.lower()

        for skill in tech_skills:
            if skill in desc_lower:
                found_skills.append(skill.title())

        return list(set(found_skills))[:8]  # Remove duplicates and limit

    def remove_duplicates(self, jobs: List[Dict]) -> List[Dict]:
        """Remove duplicate jobs based on title and company (defensive)"""
        seen = set()
        unique_jobs = []

        for job in jobs:
            title = (job.get("title") or "").lower()
            company = (job.get("company") or "").lower()
            key = (title, company)
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)

        return unique_jobs

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================

def main():
    """Main application function"""

    # Initialize RAG system
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = SmartJobRecommenderRAG()

    # App header
    st.title("💼 Smart Job Recommender")
    st.markdown("### AI-Powered Job Matching with Real-Time Search")
    st.markdown("---")

    # Sidebar for information
    with st.sidebar:
        st.header("🔧 Configuration")

        # API Status
        st.subheader("API Status")

        # Check for required API keys
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if gemini_key:
                st.success("✅ Gemini AI: Connected")
            else:
                st.error("❌ Gemini AI: API key required")
        except Exception:
            st.error("❌ Gemini AI: API key required")

        try:
            serpapi_key = st.secrets.get("SERPAPI_API_KEY") or os.environ.get("SERPAPI_API_KEY")
            if serpapi_key:
                st.success("✅ SerpAPI: Connected")
            else:
                st.error("❌ SerpAPI: API key required")
        except Exception:
            st.error("❌ SerpAPI: API key required")

        st.markdown("---")
        st.subheader("📋 Instructions")
        st.markdown("""
        **Setup Required:**
        1. Add GEMINI_API_KEY to Streamlit secrets
        2. Add SERPAPI_API_KEY to Streamlit secrets

        **How to Use:**
        1. Upload your resume PDF, OR
        2. Enter your skills manually
        3. Get personalized job recommendations
        4. Click 'Apply Now' to apply directly
        """)

        st.markdown("---")
        st.subheader("🎯 Features")
        st.markdown("""
        - Resume PDF analysis
        - Manual skill entry
        - Multi-source job search
        - Real-time matching scores
        - Clickable application links
        - Location-based search
        """)

    # Main content tabs
    tab1, tab2 = st.tabs(["📄 Resume Upload", "✍️ Manual Entry"])

    with tab1:
        st.header("📄 Upload Your Resume")
        st.markdown("Upload your resume in PDF format for AI-powered skill extraction and job matching.")

        uploaded_file = st.file_uploader(
            "Choose your resume PDF file",
            type="pdf",
            help="Upload a clear, text-readable PDF resume for best results."
        )

        if uploaded_file is not None:
            st.success(f"✅ Uploaded: {uploaded_file.name}")

            if st.button("🚀 Analyze Resume & Find Jobs", type="primary"):
                process_resume_and_find_jobs(uploaded_file)

    with tab2:
        st.header("✍️ Manual Skills Entry")
        st.markdown("Enter your skills and preferences manually to find matching job opportunities.")

        with st.form("manual_skills_form"):
            skills_input = st.text_area(
                "Your Skills (comma-separated)",
                placeholder="e.g., Python, React, Machine Learning, SQL, Project Management",
                height=100,
                help="Enter your technical and soft skills separated by commas"
            )

            col1, col2 = st.columns(2)

            with col1:
                job_interests = st.text_input(
                    "Job Interests (comma-separated)",
                    placeholder="e.g., Software Developer, Data Scientist, Product Manager",
                    help="Enter job titles or fields you're interested in"
                )

                experience_level = st.selectbox(
                    "Experience Level",
                    ["entry", "mid", "senior"],
                    help="Select your current experience level"
                )

            with col2:
                location_pref = st.text_input(
                    "Preferred Location (Optional)",
                    placeholder="e.g., United States, Remote, New York",
                    help="Enter your preferred job location"
                )

            submitted = st.form_submit_button("🔍 Find Matching Jobs", type="primary")

            if submitted:
                if skills_input.strip():
                    skills_list = [skill.strip() for skill in skills_input.split(',') if skill.strip()]
                    interests_list = [interest.strip() for interest in job_interests.split(',') if interest.strip()]

                    manual_data = {
                        "skills": skills_list,
                        "job_interests": interests_list,
                        "experience_level": experience_level
                    }

                    process_manual_skills_and_find_jobs(manual_data, location_pref)
                else:
                    st.error("Please enter at least some skills to find matching jobs.")


def process_resume_and_find_jobs(uploaded_file):
    """Process uploaded resume and find matching jobs"""

    rag_system = st.session_state.rag_system

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Load document
        status_text.text("📄 Loading PDF document...")
        progress_bar.progress(20)
        documents = rag_system.load_document_with_pypdf(uploaded_file)

        if not documents:
            st.error("❌ Failed to load PDF. Please check the file format.")
            return

        # Step 2: Analyze resume
        status_text.text("📝 Analyzing resume content...")
        progress_bar.progress(50)

        all_text = "\n\n".join([doc.page_content for doc in documents])

        final_prompt = f"""
Based on the following resume content, extract relevant information:

RESUME CONTENT:
{all_text}

Extract:
1. Technical skills (programming languages, frameworks, tools)
2. Soft skills 
3. Job preferences or career interests
4. Experience level

Format your response as:
SKILLS: [comma-separated list of skills]
JOB_INTERESTS: [comma-separated job titles/fields]
EXPERIENCE_LEVEL: [entry/mid/senior]
"""

        # Step 3: Call Gemini
        status_text.text("🤖 Analyzing with Gemini AI...")
        progress_bar.progress(80)
        extracted_data = rag_system.call_direct_gemini(final_prompt)

        # Step 4: Search jobs
        status_text.text("🔍 Searching for matching jobs...")
        progress_bar.progress(90)
        job_results = rag_system.search_jobs_with_serpapi(
            extracted_data["skills"],
            extracted_data["job_interests"]
        )

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(1)

        # Clear progress
        progress_bar.empty()
        status_text.empty()

        # Display results
        display_results(extracted_data, job_results)

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
        progress_bar.empty()
        status_text.empty()


def process_manual_skills_and_find_jobs(manual_data: Dict[str, Any], location_pref: str):
    """Process manually entered skills and find matching jobs"""

    rag_system = st.session_state.rag_system

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("📝 Processing your skills...")
        progress_bar.progress(20)

        st.success(f"✅ Skills processed: {len(manual_data['skills'])} skills found")

        status_text.text("🔍 Searching for matching jobs...")
        progress_bar.progress(60)

        # Search with location if provided
        if location_pref.strip():
            job_results = rag_system.search_jobs_with_serpapi_location(
                manual_data["skills"],
                manual_data["job_interests"],
                location_pref
            )
        else:
            job_results = rag_system.search_jobs_with_serpapi(
                manual_data["skills"],
                manual_data["job_interests"]
            )

        progress_bar.progress(100)
        status_text.text("✅ Search complete!")
        time.sleep(1)

        # Clear progress
        progress_bar.empty()
        status_text.empty()

        # Display results
        display_results(manual_data, job_results)

    except Exception as e:
        st.error(f"❌ Error during job search: {e}")
        progress_bar.empty()
        status_text.empty()


def display_results(extracted_data: Dict[str, Any], job_results: Dict[str, List]):
    """Display analysis results and job recommendations"""

    st.markdown("---")
    st.header("📊 Analysis Results")

    # Display extracted information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🛠️ Skills Found")
        if extracted_data["skills"]:
            for skill in extracted_data["skills"]:
                st.markdown(f"• {skill}")
        else:
            st.info("No specific skills detected")

    with col2:
        st.subheader("💼 Job Interests")
        if extracted_data["job_interests"]:
            for interest in extracted_data["job_interests"]:
                st.markdown(f"• {interest}")
        else:
            st.info("No specific interests detected")

    with col3:
        st.subheader("📈 Experience Level")
        level = extracted_data["experience_level"].title()
        st.markdown(f"**{level}**")

    # Display job recommendations
    st.markdown("---")
    st.header("💼 Job Recommendations")

    jobs = job_results.get("jobs", []) if isinstance(job_results, dict) else []
    internships = job_results.get("internships", []) if isinstance(job_results, dict) else []

    if jobs:
        st.subheader(f"🎯 Found {len(jobs)} Job Matches")

        for i, job in enumerate(jobs, 1):
            with st.expander(f"#{i} {job['title']} at {job['company']} - {job.get('match_score', 0)}% Match"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Company:** {job['company']}")
                    st.write(f"**Location:** {job['location']}")
                    st.write(f"**Salary:** {job.get('salary', 'Not specified')}")
                    st.write(f"**Description:** {job.get('description','')[:200]}...")

                    if job.get('required_skills'):
                        st.write("**Required Skills:**")
                        for skill in job.get('required_skills', []):
                            st.markdown(f"• {skill}")

                with col2:
                    st.metric("Match Score", f"{job.get('match_score',0)}%")
                    st.write(f"**Source:** {job.get('source', 'Unknown')}")

                    # Enhanced apply link display
                    apply_link_local = (job.get('apply_link') or '').strip()
                    if apply_link_local and apply_link_local != "#":
                        st.link_button("🚀 Apply Now", apply_link_local, type="primary")
                        st.caption("Click to apply on the job site")
                    else:
                        st.warning("No direct apply link available")
                        if job.get('company') and job.get('title'):
                            q = quote_plus(f"{job.get('company')} {job.get('title')} jobs")
                            search_url = f"https://www.google.com/search?q={q}"
                            st.link_button("🔍 Search on Google", search_url)

    # Display internship recommendations
    if internships:
        st.markdown("---")
        st.subheader(f"🎓 Found {len(internships)} Internship Matches")

        for i, internship in enumerate(internships, 1):
            with st.expander(f"#{i} {internship['title']} at {internship['company']} - {internship.get('match_score',0)}% Match"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Company:** {internship['company']}")
                    st.write(f"**Location:** {internship.get('location','')}")
                    st.write(f"**Description:** {internship.get('description','')[:200]}...")

                    if internship.get('required_skills'):
                        st.write("**Required Skills:**")
                        for skill in internship.get('required_skills', []):
                            st.markdown(f"• {skill}")

                with col2:
                    st.metric("Match Score", f"{internship.get('match_score',0)}%")
                    st.write(f"**Source:** {internship.get('source', 'Unknown')}")

                    # Use the apply_link present on the internship dict (fixed scoping)
                    internship_apply = (internship.get('apply_link') or '').strip()
                    if internship_apply and internship_apply != "#":
                        st.link_button("🚀 Apply Now", internship_apply, type="primary")
                        st.caption("Click to apply on the internship site")
                    else:
                        st.warning("No direct apply link available")
                        if internship.get('company') and internship.get('title'):
                            q = quote_plus(f"{internship.get('company')} {internship.get('title')} internship")
                            search_url = f"https://www.google.com/search?q={q}"
                            st.link_button("🔍 Search on Google", search_url)

    # No results message
    if not jobs and not internships:
        st.info("🔍 No job matches found. This could be due to:")
        st.markdown("""
        - API configuration issues
        - No matching jobs available
        - Skills extraction needs improvement

        Please check your API keys and try again.
        """)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
