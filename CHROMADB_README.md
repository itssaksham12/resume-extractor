# 🗄️ ChromaDB Results Database

A comprehensive ChromaDB implementation for storing and managing resume analysis results with the following schema:

## 📋 Database Schema

### Core Fields
- **`resume_name`** - Candidate name or resume identifier
- **`job_description`** - Original job posting text
- **`matching_skills`** - Skills found in both resume and job description (comma-separated)
- **`missing_skills`** - Required skills missing from resume (comma-separated)
- **`resume_summary`** - AI-generated summary of the resume
- **`match_score`** - Overall match percentage (0-100)
- **`recommendation`** - Hiring recommendation based on match score
- **`analysis_method`** - Method used for analysis (LSTM AI Model / Skills-based)

### Additional Metadata
- **`extra_skills`** - Skills in resume not required by job
- **`skills_match_percentage`** - Percentage of required skills matched
- **`total_resume_skills`** - Total number of skills found in resume
- **`total_jd_skills`** - Total number of skills required by job
- **`prediction_confidence`** - Confidence level of the analysis
- **`processed_timestamp`** - Unix timestamp of processing
- **`processed_date`** - Human-readable processing date
- **`pdf_path`** - Path to original PDF file (if applicable)
- **`resume_text_length`** - Length of resume text in characters

### Analysis Flags
- **`is_recommended`** - Boolean flag for recommended candidates (≥65% match)
- **`needs_training`** - Boolean flag for candidates needing training (50-65% match)
- **`has_critical_gaps`** - Boolean flag for candidates with >5 missing skills
- **`analysis_status`** - Status of analysis (completed/error)

## 🚀 Quick Start

### 1. Installation
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (already done)
pip install -r requirements.txt
```

### 2. Basic Usage
```python
from resultsDB import ResultsDatabase

# Initialize database
db = ResultsDatabase()

# Add resume analysis
resume_id = db.add_resume_analysis(
    resume_text="Resume content here...",
    job_description="Job description here...",
    resume_name="John Doe",
    pdf_path="path/to/resume.pdf"  # Optional
)

# Search resumes
results = db.search_resumes("python machine learning", n_results=5)

# Get analytics
analytics = db.get_analytics_summary()
```

### 3. Launch Database Manager Interface
```bash
python integrate_db.py
```
This launches a Gradio interface at `http://localhost:7861` with tabs for:
- 📄 Add Resume Analysis
- 🔍 Search Resumes
- 📊 Analytics
- 📋 All Resumes

## 🔧 API Reference

### ResultsDatabase Class

#### Initialization
```python
db = ResultsDatabase(db_path="resume_results_db")
```

#### Core Methods

##### `add_resume_analysis(resume_text, job_description, resume_name, pdf_path=None)`
Adds a comprehensive resume analysis to the database.

**Parameters:**
- `resume_text` (str): Text content of the resume
- `job_description` (str): Job description to compare against
- `resume_name` (str): Name of the candidate/resume
- `pdf_path` (str, optional): Path to the PDF file

**Returns:**
- `str`: Unique ID of the stored analysis or `None` if failed

##### `search_resumes(query, n_results=5, filter_criteria=None)`
Searches resumes using semantic search.

**Parameters:**
- `query` (str): Search query
- `n_results` (int): Number of results to return
- `filter_criteria` (dict, optional): Filter criteria

**Returns:**
- `dict`: Search results with metadata

##### `get_resume_by_id(resume_id)`
Gets complete resume analysis by ID.

**Parameters:**
- `resume_id` (str): Unique ID of the resume

**Returns:**
- `dict`: Complete resume analysis or `None` if not found

##### `get_all_resumes(limit=100)`
Gets all resumes from the database.

**Parameters:**
- `limit` (int): Maximum number of resumes to return

**Returns:**
- `dict`: List of all resumes with metadata

##### `get_analytics_summary()`
Gets analytics summary of all resumes in database.

**Returns:**
- `dict`: Analytics data including statistics and insights

##### `delete_resume(resume_id)`
Deletes a resume analysis from the database.

**Parameters:**
- `resume_id` (str): Unique ID of the resume

**Returns:**
- `bool`: Success status

##### `clear_database()`
Clears all data from the database.

**Returns:**
- `bool`: Success status

## 🔍 Search and Filtering

### Basic Search
```python
# Simple search
results = db.search_resumes("python machine learning")

# Search with result limit
results = db.search_resumes("data scientist", n_results=10)
```

### Advanced Filtering
```python
# Filter by minimum match score
filter_criteria = {"min_match_score": 70}
results = db.search_resumes("python", filter_criteria=filter_criteria)

# Filter for recommended candidates only
filter_criteria = {"is_recommended": True}
results = db.search_resumes("developer", filter_criteria=filter_criteria)
```

## 📊 Analytics Features

### Database Statistics
```python
analytics = db.get_analytics_summary()

print(f"Total resumes: {analytics['total_resumes']}")
print(f"Average match score: {analytics['average_match_score']:.1f}%")
print(f"Recommended candidates: {analytics['recommended_candidates']}")
print(f"High performers (≥80%): {analytics['high_performers']}")
```

### Skill Analysis
```python
# Most common matching skills
for skill, count in analytics['most_common_matching_skills']:
    print(f"{skill}: {count} times")

# Most common missing skills
for skill, count in analytics['most_common_missing_skills']:
    print(f"{skill}: {count} times")
```

## 🔄 Integration with Existing App

### Integration with Gradio App
```python
# In your existing app.py
from resultsDB import ResultsDatabase

# Initialize database
db = ResultsDatabase()

# Add to your analysis function
def analyze_resume(job_description, resume_text, resume_pdf=None):
    # Your existing analysis logic here...
    
    # Store results in database
    resume_id = db.add_resume_analysis(
        resume_text=resume_text,
        job_description=job_description,
        resume_name="Candidate",  # You can get this from user input
        pdf_path=resume_pdf.name if resume_pdf else None
    )
    
    # Return both analysis results and database ID
    return analysis_results, resume_id
```

### Adding Database Tab to Existing App
```python
# Add this to your existing Gradio interface
with gr.TabItem("🗄️ Database"):
    gr.Markdown("### Resume Database Operations")
    
    # Add your database operations here
    search_query = gr.Textbox(label="Search Resumes")
    search_btn = gr.Button("Search")
    search_output = gr.Markdown()
    
    search_btn.click(
        fn=lambda q: db.search_resumes(q) if q else "Enter search query",
        inputs=search_query,
        outputs=search_output
    )
```

## 🛠️ Configuration

### Database Path
```python
# Custom database path
db = ResultsDatabase(db_path="custom_db_path")
```

### Model Configuration
The database automatically initializes:
- BERT Summarizer (for resume summaries)
- Skills Extractor (for skill analysis)
- Resume Matcher (for match scoring)

Models are loaded from:
- `bert_summarizer_model.pth`
- `lstm_resume_matcher_best.h5`

### Embedding Configuration
The database uses sentence-transformers for embeddings:
- Model: `all-MiniLM-L6-v2`
- Fallback: Simple hash-based embedding if sentence-transformers fails

## 📁 File Structure

```
resume-reviewer/
├── resultsDB.py              # Main ChromaDB implementation
├── integrate_db.py           # Gradio interface for database management
├── CHROMADB_README.md        # This documentation
├── resume_results_db/        # ChromaDB data directory (auto-created)
│   └── chroma.sqlite3        # Database file
└── requirements.txt          # Dependencies (updated)
```

## 🚨 Error Handling

The implementation includes comprehensive error handling:

### Model Loading Failures
- Graceful fallback to rule-based methods
- Warning messages for missing models
- Continued operation with reduced functionality

### Database Errors
- Connection error handling
- Metadata validation
- Transaction rollback on failures

### Embedding Generation
- Multiple fallback methods
- Simple hash-based embeddings as last resort
- Warning messages for embedding failures

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Check if model files exist
   ls -la *.pth *.h5
   ```

2. **ChromaDB Connection Issues**
   ```python
   # Check database path
   print(db.db_path)
   # Check if collection exists
   print(db.collection.name)
   ```

3. **Embedding Generation Failures**
   ```python
   # Install tf-keras for better compatibility
   pip install tf-keras
   ```

### Performance Optimization

1. **Batch Operations**
   ```python
   # For multiple resumes, process in batches
   for batch in resume_batches:
       for resume in batch:
           db.add_resume_analysis(...)
   ```

2. **Search Optimization**
   ```python
   # Use specific filters to reduce search space
   filter_criteria = {"is_recommended": True, "min_match_score": 70}
   results = db.search_resumes("query", filter_criteria=filter_criteria)
   ```

## 📈 Future Enhancements

### Planned Features
- [ ] Batch import/export functionality
- [ ] Advanced filtering options
- [ ] Resume comparison tools
- [ ] Automated backup and restore
- [ ] Performance metrics dashboard
- [ ] Integration with external HR systems

### Customization Options
- [ ] Custom embedding models
- [ ] Configurable skill categories
- [ ] Custom scoring algorithms
- [ ] Multi-language support

## 🤝 Contributing

To contribute to the ChromaDB implementation:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This ChromaDB implementation is part of the Resume Reviewer AI project and follows the same license terms.
