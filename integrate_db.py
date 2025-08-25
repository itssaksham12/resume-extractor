#!/usr/bin/env python3
"""
Integration script for ChromaDB with existing Gradio app
Shows how to integrate the ResultsDatabase with the resume analysis workflow
"""

import gradio as gr
from resultsDB import ResultsDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the database
db = ResultsDatabase()

def analyze_and_store_resume(job_description, resume_text, resume_pdf=None, candidate_name="Unknown"):
    """
    Analyze resume and store results in ChromaDB
    This function can be integrated with your existing Gradio app
    """
    try:
        # Handle PDF upload if provided
        if resume_pdf is not None:
            pdf_text = db.extract_text_from_pdf(resume_pdf.name)
            if pdf_text.strip():
                resume_text = pdf_text
            elif not resume_text.strip():
                return "Please provide either resume text or a valid PDF file."
        elif not resume_text.strip():
            return "Please enter resume text or upload a PDF file."
        
        # Add resume analysis to database
        resume_id = db.add_resume_analysis(
            resume_text=resume_text,
            job_description=job_description,
            resume_name=candidate_name,
            pdf_path=resume_pdf.name if resume_pdf else None
        )
        
        if resume_id:
            # Get the stored analysis
            analysis = db.get_resume_by_id(resume_id)
            if analysis:
                metadata = analysis['metadata']
                
                # Format results for display
                result_text = f"""
## 📊 Resume Analysis Results (Stored in Database)

**Candidate**: {metadata['resume_name']}
**Analysis ID**: {resume_id}
**Match Score**: {metadata['match_score']:.1f}%
**Recommendation**: {metadata['recommendation']}

### 🎯 Skills Analysis
**Matching Skills**: {metadata['matching_skills']}
**Missing Skills**: {metadata['missing_skills']}
**Skills Match Percentage**: {metadata['skills_match_percentage']:.1f}%

### 📝 Resume Summary
{metadata['resume_summary']}

### 📈 Analysis Details
- **Total Resume Skills**: {metadata['total_resume_skills']}
- **Total JD Skills**: {metadata['total_jd_skills']}
- **Analysis Method**: {metadata['analysis_method']}
- **Processed Date**: {metadata['processed_date']}

### 🔍 Database Operations Available
- Search resumes: `db.search_resumes("query")`
- Get analytics: `db.get_analytics_summary()`
- Get all resumes: `db.get_all_resumes()`
"""
                return result_text
            else:
                return f"✅ Analysis stored with ID: {resume_id}, but could not retrieve details."
        else:
            return "❌ Failed to store resume analysis in database."
            
    except Exception as e:
        logger.error(f"Error in analyze_and_store_resume: {e}")
        return f"❌ Error: {str(e)}"

def search_stored_resumes(query, min_match_score=None):
    """Search resumes stored in the database"""
    try:
        filter_criteria = {}
        if min_match_score:
            filter_criteria["min_match_score"] = float(min_match_score)
        
        results = db.search_resumes(query, n_results=10, filter_criteria=filter_criteria)
        
        if results['total_results'] == 0:
            return "No matching resumes found."
        
        result_text = f"## 🔍 Search Results for: '{query}'\n\n"
        result_text += f"**Found {results['total_results']} matching resumes:**\n\n"
        
        for i, result in enumerate(results['results'], 1):
            result_text += f"### {i}. {result['resume_name']}\n"
            result_text += f"- **Match Score**: {result['match_score']:.1f}%\n"
            result_text += f"- **Recommendation**: {result['recommendation']}\n"
            result_text += f"- **Matching Skills**: {', '.join(result['matching_skills'])}\n"
            result_text += f"- **Missing Skills**: {', '.join(result['missing_skills'])}\n"
            result_text += f"- **Summary**: {result['summary'][:100]}...\n\n"
        
        return result_text
        
    except Exception as e:
        logger.error(f"Error in search_stored_resumes: {e}")
        return f"❌ Error: {str(e)}"

def get_database_analytics():
    """Get analytics summary of the database"""
    try:
        analytics = db.get_analytics_summary()
        
        if 'error' in analytics:
            return f"❌ Error: {analytics['error']}"
        
        result_text = f"""
## 📊 Database Analytics

**Total Resumes**: {analytics['total_resumes']}
**Average Match Score**: {analytics['average_match_score']:.1f}%
**Recommended Candidates**: {analytics['recommended_candidates']}
**Candidates Needing Training**: {analytics['candidates_needing_training']}
**High Performers (≥80%)**: {analytics['high_performers']}

### 🏆 Most Common Matching Skills
"""
        for skill, count in analytics['most_common_matching_skills']:
            result_text += f"- {skill}: {count} times\n"
        
        result_text += "\n### ❌ Most Common Missing Skills\n"
        for skill, count in analytics['most_common_missing_skills']:
            result_text += f"- {skill}: {count} times\n"
        
        return result_text
        
    except Exception as e:
        logger.error(f"Error in get_database_analytics: {e}")
        return f"❌ Error: {str(e)}"

def list_all_resumes():
    """List all resumes in the database"""
    try:
        all_resumes = db.get_all_resumes(limit=50)
        
        if all_resumes['total_count'] == 0:
            return "No resumes found in database."
        
        result_text = f"## 📋 All Resumes in Database ({all_resumes['total_count']} total)\n\n"
        
        for i, resume in enumerate(all_resumes['resumes'], 1):
            result_text += f"### {i}. {resume['resume_name']}\n"
            result_text += f"- **ID**: {resume['id']}\n"
            result_text += f"- **Match Score**: {resume['match_score']:.1f}%\n"
            result_text += f"- **Recommendation**: {resume['recommendation']}\n"
            result_text += f"- **Processed**: {resume['processed_date']}\n\n"
        
        return result_text
        
    except Exception as e:
        logger.error(f"Error in list_all_resumes: {e}")
        return f"❌ Error: {str(e)}"

# Create Gradio interface for database operations
def create_database_interface():
    """Create a Gradio interface for database operations"""
    
    with gr.Blocks(title="Resume Database Manager") as interface:
        gr.Markdown("# 🗄️ Resume Analysis Database Manager")
        gr.Markdown("Manage and search resume analysis results stored in ChromaDB")
        
        with gr.Tabs():
            # Tab 1: Add and Analyze Resumes
            with gr.TabItem("📄 Add Resume Analysis"):
                gr.Markdown("### Add new resume analysis to database")
                
                with gr.Row():
                    with gr.Column():
                        job_desc_input = gr.Textbox(
                            label="Job Description",
                            placeholder="Paste the job description here...",
                            lines=5
                        )
                        candidate_name_input = gr.Textbox(
                            label="Candidate Name",
                            placeholder="Enter candidate name",
                            value="Unknown"
                        )
                    
                    with gr.Column():
                        resume_text_input = gr.Textbox(
                            label="Resume Text",
                            placeholder="Paste resume text here...",
                            lines=10
                        )
                        resume_pdf_input = gr.File(
                            label="Or Upload PDF",
                            file_types=[".pdf"]
                        )
                
                analyze_btn = gr.Button("🔍 Analyze & Store", variant="primary")
                analyze_output = gr.Markdown(label="Analysis Results")
                
                analyze_btn.click(
                    fn=analyze_and_store_resume,
                    inputs=[job_desc_input, resume_text_input, resume_pdf_input, candidate_name_input],
                    outputs=analyze_output
                )
            
            # Tab 2: Search Resumes
            with gr.TabItem("🔍 Search Resumes"):
                gr.Markdown("### Search resumes in database")
                
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter search terms (e.g., 'python machine learning')",
                    lines=2
                )
                min_score = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=5,
                    label="Minimum Match Score (%)"
                )
                
                search_btn = gr.Button("🔍 Search", variant="primary")
                search_output = gr.Markdown(label="Search Results")
                
                search_btn.click(
                    fn=search_stored_resumes,
                    inputs=[search_query, min_score],
                    outputs=search_output
                )
            
            # Tab 3: Analytics
            with gr.TabItem("📊 Analytics"):
                gr.Markdown("### Database analytics and insights")
                
                analytics_btn = gr.Button("📊 Get Analytics", variant="primary")
                analytics_output = gr.Markdown(label="Analytics Results")
                
                analytics_btn.click(
                    fn=get_database_analytics,
                    inputs=[],
                    outputs=analytics_output
                )
            
            # Tab 4: List All Resumes
            with gr.TabItem("📋 All Resumes"):
                gr.Markdown("### View all resumes in database")
                
                list_btn = gr.Button("📋 List All", variant="primary")
                list_output = gr.Markdown(label="All Resumes")
                
                list_btn.click(
                    fn=list_all_resumes,
                    inputs=[],
                    outputs=list_output
                )
    
    return interface

# Example usage functions
def example_usage():
    """Example of how to use the database programmatically"""
    print("🚀 Example Database Usage")
    print("=" * 50)
    
    # Example 1: Add a resume analysis
    job_desc = """
    We are seeking a Senior Data Scientist with 5+ years of experience in machine learning, 
    Python programming, and statistical analysis. Experience with TensorFlow, PyTorch, SQL, 
    and cloud platforms (AWS/Azure) is required.
    """
    
    resume_text = """
    Jane Smith
    Senior Data Scientist
    
    EXPERIENCE:
    TechCorp (2019-2023)
    - Developed ML models using Python and TensorFlow
    - Worked with SQL databases and AWS cloud platform
    - Led data analysis projects for 3 years
    
    SKILLS: Python, Machine Learning, TensorFlow, SQL, AWS, Data Analysis
    EDUCATION: Master's in Computer Science
    """
    
    # Add to database
    resume_id = db.add_resume_analysis(
        resume_text=resume_text,
        job_description=job_desc,
        resume_name="Jane Smith"
    )
    
    print(f"✅ Added resume analysis: {resume_id}")
    
    # Example 2: Search resumes
    search_results = db.search_resumes("python machine learning", n_results=5)
    print(f"🔍 Found {search_results['total_results']} matching resumes")
    
    # Example 3: Get analytics
    analytics = db.get_analytics_summary()
    print(f"📊 Database has {analytics['total_resumes']} resumes")
    print(f"📊 Average match score: {analytics['average_match_score']:.1f}%")

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_database_interface()
    interface.launch(share=False, server_name="0.0.0.0", server_port=7861)
    
    # Uncomment to run example usage
    # example_usage()
