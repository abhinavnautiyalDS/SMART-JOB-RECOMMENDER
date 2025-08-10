"""
SMART JOB RECOMMENDER - DEPLOYMENT READY VERSION
Custom RAG with Gemini Integration for Streamlit Cloud Deployment

Pipeline: PyPDF → SemanticChunker → Vector Store → MMR → Direct Gemini → SerpAPI

To deploy on Streamlit Cloud:
1. Push this code to GitHub
2. Connect GitHub repo to Streamlit Cloud  
3. Add secrets in Streamlit Cloud dashboard:
   - GEMINI_API_KEY
   - SERPAPI_API_KEY
"""

import streamlit as st
import pandas as pd
import os
import tempfile
import requests
import json
from typing import List, Dict, Any
from io import BytesIO

# Streamlit page config
st.set_page_config(
    page_title="Smart Job Recommender",
    page_icon="🤖", 
    layout="wide"
)

# Try importing required packages with error handling
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain.schema import Document
    import google.generativeai as genai
    
    PACKAGES_AVAILABLE = True
except ImportError as e:
    st.error(f"Missing required packages. Install: {str(e)}")
    PACKAGES_AVAILABLE = False

# ============================================================================
# CUSTOM RAG SYSTEM CLASS
# ============================================================================

class GeminiRAGSystem:
    """Custom RAG system using Gemini instead of OpenAI"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.semantic_chunker = None
        self.gemini_client = None
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all components with Gemini"""
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
            
            if gemini_key:
                # Configure Gemini
                genai.configure(api_key=gemini_key)
                self.gemini_client = genai.GenerativeModel('gemini-pro')
                
                # Initialize embeddings for LangChain
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=gemini_key
                )
                
                # Initialize SemanticChunker
                self.semantic_chunker = SemanticChunker(
                    embeddings=self.embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=95
                )
                
                return True
            else:
                st.warning("Gemini API key not found")
                return False
                
        except Exception as e:
            st.error(f"Error initializing Gemini RAG: {e}")
            return False
    
    def load_document_with_pypdf(self, uploaded_file):
        """Step 1: Load document using PyPDF"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            os.unlink(tmp_file_path)
            
            st.success(f"✅ Loaded {len(documents)} pages using PyPDF")
            return documents
            
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
            return []
    
    def split_with_semantic_chunker(self, documents: List[Document]) -> List[Document]:
        """Step 2: Split using SemanticChunker"""
        try:
            if not self.semantic_chunker:
                st.error("SemanticChunker not initialized")
                return documents
            
            chunks = self.semantic_chunker.split_documents(documents)
            st.success(f"✅ Created {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            st.error(f"Error with SemanticChunker: {e}")
            return documents
    
    def create_vector_store(self, chunks: List[Document]) -> bool:
        """Step 3: Create vector store"""
        try:
            if not self.embeddings:
                st.error("Embeddings not initialized")
                return False
            
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=None
            )
            
            st.success(f"✅ Vector store created with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return False
    
    def retrieve_with_mmr(self, query: str, k: int = 5) -> List[Document]:
        """Step 4: Maximum Margin Relevance retrieval"""
        try:
            if not self.vectorstore:
                st.error("Vector store not available")
                return []
            
            mmr_retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "lambda_mult": 0.5,
                    "fetch_k": k * 2
                }
            )
            
            relevant_docs = mmr_retriever.get_relevant_documents(query)
            st.success(f"✅ Retrieved {len(relevant_docs)} documents using MMR")
            return relevant_docs
            
        except Exception as e:
            st.error(f"Error with MMR retrieval: {e}")
            return []
    
    def combine_docs_with_query(self, relevant_docs: List[Document], user_query: str) -> str:
        """Step 5: Combine documents with query"""
        try:
            doc_contents = [doc.page_content.strip() for doc in relevant_docs if doc.page_content.strip()]
            combined_context = "\n\n".join(doc_contents)
            
            final_prompt = f"""
Based on the following resume content, extract relevant information:

RESUME CONTENT:
{combined_context}

USER QUERY: {user_query}

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
            
            st.success(f"✅ Combined {len(relevant_docs)} documents with query")
            return final_prompt
            
        except Exception as e:
            st.error(f"Error combining documents: {e}")
            return user_query
    
    def call_direct_gemini(self, prompt: str) -> Dict[str, Any]:
        """Step 6: Direct Gemini LLM call (not LangChain)"""
        try:
            if not self.gemini_client:
                st.error("Gemini client not initialized")
                return {"skills": [], "job_interests": [], "experience_level": "entry"}
            
            response = self.gemini_client.generate_content(prompt)
            llm_output = response.text
            
            parsed_data = self.parse_gemini_response(llm_output)
            st.success(f"✅ Gemini extracted {len(parsed_data['skills'])} skills")
            return parsed_data
            
        except Exception as e:
            st.error(f"Error with Gemini call: {e}")
            return {"skills": [], "job_interests": [], "experience_level": "entry"}
    
    def parse_gemini_response(self, llm_output: str) -> Dict[str, Any]:
        """Parse Gemini response"""
        try:
            skills = []
            job_interests = []
            experience_level = "entry"
            
            lines = llm_output.split('\n')
            for line in lines:
                if 'SKILLS:' in line:
                    skills_text = line.split('SKILLS:')[1].strip()
                    skills = [skill.strip() for skill in skills_text.split(',') if skill.strip()]
                elif 'JOB_INTERESTS:' in line:
                    interests_text = line.split('JOB_INTERESTS:')[1].strip()
                    job_interests = [interest.strip() for interest in interests_text.split(',') if interest.strip()]
                elif 'EXPERIENCE_LEVEL:' in line:
                    experience_level = line.split('EXPERIENCE_LEVEL:')[1].strip().lower()
            
            return {
                "skills": skills,
                "job_interests": job_interests,
                "experience_level": experience_level
            }
            
        except Exception as e:
            st.error(f"Error parsing Gemini response: {e}")
            return {"skills": [], "job_interests": [], "experience_level": "entry"}
    
    def search_jobs_with_serpapi(self, skills: List[str], job_interests: List[str]) -> Dict[str, List]:
        """Step 7: Search jobs with SerpAPI"""
        try:
            serpapi_key = st.secrets.get("SERPAPI_API_KEY", os.environ.get("SERPAPI_API_KEY"))
            
            if not serpapi_key:
                st.warning("SerpAPI key not found - using sample data")
                return self.get_sample_jobs(skills)
            
            # Create search queries
            search_queries = []
            if skills:
                skills_query = " ".join(skills[:3]) + " jobs"
                search_queries.append(skills_query)
            
            for interest in job_interests[:2]:
                search_queries.append(f"{interest} jobs")
            
            if not search_queries:
                search_queries = ["software developer jobs"]
            
            all_jobs = []
            all_internships = []
            
            for query in search_queries[:2]:  # Limit queries
                try:
                    url = "https://serpapi.com/search"
                    params = {
                        "engine": "google_jobs",
                        "q": query,
                        "location": "United States",
                        "api_key": serpapi_key,
                        "num": 10
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    data = response.json()
                    
                    if "jobs_results" in data:
                        for job in data["jobs_results"]:
                            job_data = {
                                "title": job.get("title", "Unknown Title"),
                                "company": job.get("company_name", "Unknown Company"),
                                "location": job.get("location", "Unknown Location"),
                                "description": job.get("description", "No description"),
                                "apply_link": job.get("apply_link", "#"),
                                "salary": job.get("salary", "Not specified"),
                                "source": "Google Jobs (SerpAPI)",
                                "match_score": self.calculate_match_score(skills, job.get("description", "")),
                                "required_skills": self.extract_skills_from_description(job.get("description", ""))
                            }
                            
                            if "intern" in job.get("title", "").lower():
                                all_internships.append(job_data)
                            else:
                                all_jobs.append(job_data)
                
                except Exception as e:
                    st.error(f"SerpAPI error for query '{query}': {e}")
                    continue
            
            # Remove duplicates
            unique_jobs = self.remove_duplicates(all_jobs)
            unique_internships = self.remove_duplicates(all_internships)
            
            st.success(f"✅ Found {len(unique_jobs)} jobs and {len(unique_internships)} internships")
            
            return {
                "jobs": unique_jobs[:10],
                "internships": unique_internships[:5],
                "search_queries": search_queries[:2]
            }
            
        except Exception as e:
            st.error(f"Error with SerpAPI: {e}")
            return self.get_sample_jobs(skills)
    
    def calculate_match_score(self, user_skills: List[str], job_description: str) -> int:
        """Calculate match percentage"""
        if not user_skills or not job_description:
            return 0
        
        job_desc_lower = job_description.lower()
        matches = sum(1 for skill in user_skills if skill.lower() in job_desc_lower)
        return min(100, int((matches / len(user_skills)) * 100))
    
    def extract_skills_from_description(self, description: str) -> List[str]:
        """Extract skills from job description"""
        common_skills = [
            "Python", "JavaScript", "Java", "React", "Node.js", "SQL", "AWS",
            "Docker", "Kubernetes", "Git", "Machine Learning", "Data Science"
        ]
        
        found_skills = []
        desc_lower = description.lower()
        
        for skill in common_skills:
            if skill.lower() in desc_lower:
                found_skills.append(skill)
        
        return found_skills[:6]
    
    def remove_duplicates(self, jobs: List[Dict]) -> List[Dict]:
        """Remove duplicate jobs"""
        seen = set()
        unique_jobs = []
        
        for job in jobs:
            identifier = f"{job['title'].lower()}_{job['company'].lower()}"
            if identifier not in seen:
                seen.add(identifier)
                unique_jobs.append(job)
        
        return unique_jobs
    
    def get_sample_jobs(self, skills: List[str]) -> Dict[str, List]:
        """Sample jobs fallback"""
        return {
            "jobs": [
                {
                    "title": "Software Developer",
                    "company": "Tech Company",
                    "location": "Remote",
                    "description": f"Looking for developer with {', '.join(skills[:3])} experience",
                    "apply_link": "#",
                    "salary": "$70,000 - $90,000",
                    "source": "Sample Data",
                    "match_score": 85,
                    "required_skills": skills[:5] if skills else ["Python", "JavaScript"]
                }
            ],
            "internships": [
                {
                    "title": "Software Engineering Intern",
                    "company": "Startup Inc",
                    "location": "San Francisco, CA",
                    "description": f"Internship requiring {', '.join(skills[:2]) if skills else 'programming'} knowledge",
                    "apply_link": "#",
                    "salary": "$20/hour",
                    "source": "Sample Data",
                    "match_score": 75,
                    "required_skills": skills[:3] if skills else ["Python", "Git"]
                }
            ],
            "search_queries": [" ".join(skills[:3]) if skills else "software developer"]
        }
    
    def execute_complete_pipeline(self, uploaded_file, user_query: str) -> Dict[str, Any]:
        """Execute complete RAG pipeline"""
        try:
            st.info("🚀 Starting Gemini RAG Pipeline...")
            
            # Step 1: PyPDF loading
            documents = self.load_document_with_pypdf(uploaded_file)
            if not documents:
                return {"error": "Failed to load document"}
            
            # Step 2: SemanticChunker
            chunks = self.split_with_semantic_chunker(documents)
            
            # Step 3: Vector store
            if not self.create_vector_store(chunks):
                return {"error": "Failed to create vector store"}
            
            # Step 4: MMR retrieval
            relevant_docs = self.retrieve_with_mmr(user_query)
            
            # Step 5: Combine docs with query
            final_prompt = self.combine_docs_with_query(relevant_docs, user_query)
            
            # Step 6: Direct Gemini call
            extraction_result = self.call_direct_gemini(final_prompt)
            
            # Step 7: SerpAPI search
            job_results = self.search_jobs_with_serpapi(
                extraction_result["skills"],
                extraction_result["job_interests"]
            )
            
            st.success("✅ Complete Gemini RAG pipeline executed!")
            
            return {
                "skills": extraction_result["skills"],
                "job_interests": extraction_result["job_interests"],
                "experience_level": extraction_result["experience_level"],
                "jobs": job_results["jobs"],
                "internships": job_results["internships"],
                "search_queries": job_results.get("search_queries", []),
                "pipeline_completed": True
            }
            
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return {"error": str(e)}

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'extracted_skills' not in st.session_state:
    st.session_state.extracted_skills = []
if 'job_interests' not in st.session_state:
    st.session_state.job_interests = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {'jobs': [], 'internships': []}
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False

# ============================================================================
# INITIALIZE RAG SYSTEM
# ============================================================================

@st.cache_resource
def load_gemini_rag_system():
    """Initialize Gemini RAG system"""
    return GeminiRAGSystem()

if PACKAGES_AVAILABLE:
    gemini_rag = load_gemini_rag_system()

# Check API availability
gemini_available = bool(st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY")))
serpapi_available = bool(st.secrets.get("SERPAPI_API_KEY", os.environ.get("SERPAPI_API_KEY")))

# ============================================================================
# MAIN UI
# ============================================================================

st.title("🤖 Smart Job Recommender - Gemini RAG System")
st.markdown("**Custom RAG Pipeline**: PyPDF → SemanticChunker → Vector Store → MMR → Direct Gemini → SerpAPI")

# API Status
st.subheader("API Status")
col1, col2 = st.columns(2)

with col1:
    if gemini_available:
        st.success("✅ Gemini API Connected")
    else:
        st.error("❌ Gemini API Key Required")

with col2:
    if serpapi_available:
        st.success("✅ SerpAPI Connected")
    else:
        st.warning("⚠️ SerpAPI Key Missing (will use sample data)")

# Main interface
col_left, col_right = st.columns([1, 1])

# Left column: Upload
with col_left:
    st.header("📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=['pdf'],
        help="Upload PDF resume to execute complete RAG pipeline"
    )
    
    if uploaded_file and PACKAGES_AVAILABLE:
        st.success(f"📄 File uploaded: {uploaded_file.name}")
        
        # Pipeline steps preview
        st.subheader("Pipeline Steps")
        steps = [
            "1. PyPDF LangChain Loader",
            "2. SemanticChunker Text Splitting",
            "3. ChromaDB Vector Store",
            "4. Maximum Margin Relevance",
            "5. Combine Docs + Query",
            "6. Direct Gemini LLM Call",
            "7. SerpAPI Job Search"
        ]
        
        for step in steps:
            st.write(step)
        
        # Execute button
        if st.button("🚀 Execute RAG Pipeline", type="primary"):
            if not gemini_available:
                st.error("❌ Gemini API key required")
            else:
                with st.spinner("Executing Gemini RAG pipeline..."):
                    result = gemini_rag.execute_complete_pipeline(
                        uploaded_file,
                        "Extract all technical skills, programming languages, frameworks, and career interests from this resume"
                    )
                    
                    if "error" in result:
                        st.error(f"Pipeline failed: {result['error']}")
                    else:
                        st.session_state.extracted_skills = result.get("skills", [])
                        st.session_state.job_interests = result.get("job_interests", [])
                        st.session_state.recommendations = {
                            "jobs": result.get("jobs", []),
                            "internships": result.get("internships", [])
                        }
                        st.session_state.search_performed = True
                        st.rerun()

# Right column: Results
with col_right:
    st.header("📊 Results")
    
    # Skills display
    if st.session_state.extracted_skills:
        st.subheader("🔧 Extracted Skills")
        skills_df = pd.DataFrame({"Skills": st.session_state.extracted_skills})
        st.dataframe(skills_df, use_container_width=True)
        
        if st.session_state.job_interests:
            st.subheader("💼 Career Interests")
            interests_df = pd.DataFrame({"Interests": st.session_state.job_interests})
            st.dataframe(interests_df, use_container_width=True)
    
    # Job recommendations
    if st.session_state.search_performed and st.session_state.recommendations:
        st.subheader("💼 Job Recommendations")
        jobs = st.session_state.recommendations['jobs']
        
        if jobs:
            for i, job in enumerate(jobs[:5], 1):
                with st.expander(f"🏢 {job['title']} at {job['company']}"):
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        st.write(f"**Location:** {job['location']}")
                        st.write(f"**Salary:** {job.get('salary', 'Not specified')}")
                        
                        description = job.get('description', 'No description')
                        if len(description) > 200:
                            description = description[:200] + "..."
                        st.write(f"**Description:** {description}")
                        
                        if job.get('source') == 'Google Jobs (SerpAPI)':
                            st.info("🌐 Real job from Google Jobs")
                    
                    with col_b:
                        st.write("**Required Skills:**")
                        for skill in job.get('required_skills', [])[:4]:
                            if skill in st.session_state.extracted_skills:
                                st.success(f"✅ {skill}")
                            else:
                                st.info(f"📝 {skill}")
                        
                        match_score = job.get('match_score', 0)
                        st.metric("Match Score", f"{match_score}%")
                    
                    apply_link = job.get('apply_link', '#')
                    if apply_link and apply_link != '#':
                        st.link_button("Apply Now", apply_link)
        
        # Internships
        internships = st.session_state.recommendations['internships']
        if internships:
            st.subheader("🎓 Internship Opportunities")
            for i, internship in enumerate(internships[:3], 1):
                with st.expander(f"🎓 {internship['title']} at {internship['company']}"):
                    st.write(f"**Location:** {internship['location']}")
                    st.write(f"**Compensation:** {internship.get('salary', 'Not specified')}")
                    
                    description = internship.get('description', 'No description')
                    if len(description) > 150:
                        description = description[:150] + "..."
                    st.write(f"**Description:** {description}")
                    
                    apply_link = internship.get('apply_link', '#')
                    if apply_link and apply_link != '#':
                        st.link_button("Apply Now", apply_link)
    
    elif not st.session_state.extracted_skills:
        st.info("👆 Upload a PDF resume to start the RAG pipeline")
    
    elif not st.session_state.search_performed:
        st.info("Click 'Execute RAG Pipeline' to find recommendations")

# Footer
st.markdown("---")
if not gemini_available or not serpapi_available:
    st.warning("⚠️ Add API keys in Streamlit Cloud secrets for full functionality")

st.markdown(
    "<div style='text-align: center; color: gray;'>Smart Job Recommender | Gemini RAG Pipeline | Ready for Streamlit Cloud</div>",
    unsafe_allow_html=True
)
