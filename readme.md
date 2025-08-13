
<img width="1008" height="647" alt="ChatGPT Image Aug 11, 2025, 12_53_45 PM" src="https://github.com/user-attachments/assets/37d6a2ea-7023-434d-a2e6-b7451cc4cb03" />

# Smart Job Recommender - Gemini RAG System


## Project Overview
Finding a job or internship is time-consuming and frustrating for many students:You have to search on multiple websites (Google Jobs, LinkedIn, etc.).You see lots of irrelevant job posts that don’t match your skills.You waste time reading through dozens of descriptions.
What if you could have an AI assistant that:
- Reads your resume or skills.
- Searches for jobs online.
- Picks the ones that match you best.
- Explains why they’re a good fit.


## 🎯 Problem Statement

This project aims to build an AI-powered job recommendation web application that helps students and fresh graduates quickly find the most relevant job or internship opportunities.
The app will use LangChain, Google’s Gemini LLM, and Retrieval-Augmented Generation (RAG) techniques to:

- Understand the user’s skills, experience, and preferences from either a PDF resume or manual input.
- Retrieve relevant job postings from Google Jobs or LinkedIn Jobs using SerpAPI.
- Match and rank jobs based on skill relevance.
- Provide short, personalized explanations for each recommendation.

The application will be developed with Streamlit to deliver an interactive and user-friendly experience, making job searching smarter, faster, and more personalized.

## 🚀 Quick Deploy on Streamlit Cloud

1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)** 
3. **Connect your GitHub account** and select this repository
4. **Set the main file path** to `streamlit_app.py`
5. **Add secrets** in Advanced Settings:
   ```
   GEMINI_API_KEY = "your_gemini_api_key_here"
   SERPAPI_API_KEY = "your_serpapi_key_here"  
   ```
6. **Click Deploy**

## 🔧 RAG Pipeline Architecture

**Complete 7-Step Pipeline:**
1. **PyPDF LangChain Loader** - Document loading
2. **SemanticChunker** - Intelligent text splitting  
3. **ChromaDB Vector Store** - Vector storage with Gemini embeddings
4. **Maximum Margin Relevance** - Diverse document retrieval
5. **Prompt Engineering** - Combine context with user query
6. **Direct Gemini LLM** - Skill extraction (not via LangChain)
7. **SerpAPI Integration** - Real Google Jobs search

## 🔑 Required API Keys

### Gemini API Key (Required)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create new API key
3. Add to Streamlit secrets as `GEMINI_API_KEY`

### SerpAPI Key (Optional - for real jobs)
1. Sign up at [SerpAPI](https://serpapi.com/)
2. Get your API key from dashboard
3. Add to Streamlit secrets as `SERPAPI_API_KEY`

**Without SerpAPI:** App will use sample job data with clear labeling

## 💻 Local Development

```bash
# Clone repository
git clone <your-repo-url>
cd smart-job-recommender

# Install dependencies
pip install -r requirements_streamlit.txt

# Set environment variables
export GEMINI_API_KEY="your_key_here"
export SERPAPI_API_KEY="your_key_here"

# Run locally
streamlit run streamlit_app.py
```

## 📋 Features

- **PDF Resume Upload** - Uses PyPDF for document processing
- **Intelligent Skill Extraction** - Gemini-powered analysis
- **Real Job Search** - Google Jobs via SerpAPI
- **Match Scoring** - AI-powered compatibility analysis  
- **Interactive UI** - Clean, responsive interface
- **Error Handling** - Graceful fallbacks for missing APIs

## 🔍 How It Works

1. **Upload PDF Resume** → System loads document with PyPDF
2. **Semantic Processing** → Text split into meaningful chunks
3. **Vector Search** → Create embeddings and vector store
4. **Smart Retrieval** → Find relevant resume sections
5. **AI Analysis** → Gemini extracts skills and preferences
6. **Job Matching** → Search real jobs matching your profile
7. **Results Display** → View ranked opportunities with insights

## 📊 Sample Output

The system extracts:
- Technical skills (Python, JavaScript, React, etc.)
- Soft skills (Communication, Leadership, etc.) 
- Career interests (Software Developer, Data Scientist, etc.)
- Experience level (Entry, Mid, Senior)

Then finds matching jobs with:
- Real job postings from Google Jobs
- Compatibility scores
- Required vs. your skills comparison
- Direct application links

## 🛠️ Customization

**Modify Search Parameters:**
- Edit `search_jobs_with_serpapi()` function
- Adjust location, job types, or number of results

**Change Gemini Model:**
- Update model name in `GeminiRAGSystem.__init__()`
- Available models: `gemini-pro`, `gemini-pro-vision`

**Adjust RAG Settings:**
- Modify chunk size in `SemanticChunker`
- Change MMR parameters for retrieval diversity
- Update embedding model for different languages

## 📝 Deployment Checklist

- [ ] Repository contains `streamlit_app.py`
- [ ] Repository contains `requirements_streamlit.txt`  
- [ ] Gemini API key added to Streamlit secrets
- [ ] SerpAPI key added to Streamlit secrets (optional)
- [ ] App deployed on Streamlit Cloud
- [ ] Test with sample PDF resume

## 🔒 Security Notes

- API keys stored securely in Streamlit Cloud secrets
- No API keys exposed in code
- Temporary files automatically cleaned up
- No user data stored permanently

## 🐛 Troubleshooting

**"Missing required packages" error:**
- Check `requirements_streamlit.txt` is properly formatted
- Verify all package names are correct

**"Gemini API key not found" warning:**
- Ensure `GEMINI_API_KEY` is set in Streamlit secrets
- Check API key is valid and has quota

**"SerpAPI error" messages:**  
- Verify `SERPAPI_API_KEY` in secrets
- Check SerpAPI account has remaining credits
- App will use sample data as fallback

## 📈 Performance Tips

- Upload PDF files under 10MB for faster processing
- Ensure stable internet for API calls
- Use specific job titles in resume for better matching
- Include relevant technical skills prominently

---

**Ready to deploy?** Just push to GitHub and connect to Streamlit Cloud!


https://github.com/user-attachments/assets/46dc4b1b-d58f-44b5-a65a-2913105f832e


https://github.com/user-attachments/assets/b4c9c715-661a-49d4-b222-213202b4dfa5





