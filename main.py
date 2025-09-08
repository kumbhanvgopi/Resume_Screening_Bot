import os
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
import logging
import io
import csv
import json
import time

# Import the enhanced accuracy-focused classes and dynamic config
from vectors import AccuracyVectorManager
from chatbot import AccuracyScreeningChatbot
from config import get_config, update_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress unnecessary logs from specific libraries
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced AI Resume Screening API - FULLY DYNAMIC",
    description="Enhanced Accuracy-First AI Resume Screening with Specialized Agents & Dynamic Configuration",
    version="4.2.0"
)

# Configure CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Jinja2 templates
try:
    os.makedirs("templates", exist_ok=True)
    templates = Jinja2Templates(directory="templates")
except Exception as e:
    logger.error(f"Failed to set up templates directory: {str(e)}")
    raise

# Mount static files directory
try:
    os.makedirs("static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.error(f"Failed to mount static directory: {str(e)}")
    raise

# Enhanced Pydantic models for request/response
class StructuredScores(BaseModel):
    mandatory_skills: Dict[str, Any] = Field(default_factory=dict)
    total_experience: Dict[str, Any] = Field(default_factory=dict)
    relevant_experience: Dict[str, Any] = Field(default_factory=dict)
    project_exposure: Dict[str, Any] = Field(default_factory=dict)

class CandidateScore(BaseModel):
    name: str
    score: float
    reason: str
    overall_rating: str = "Unknown"
    key_strengths: List[str] = Field(default_factory=list)
    key_gaps: List[str] = Field(default_factory=list)
    recommendation: str = ""
    structured_scores: StructuredScores = Field(default_factory=StructuredScores)
    personal_info: Dict[str, Any] = Field(default_factory=dict)
    confidence_level: str = "UNKNOWN"
    evaluation_timestamp: float = 0.0

class EnhancedMatchingResult(BaseModel):
    success: bool
    message: str
    candidates: List[CandidateScore] = Field(default_factory=list)
    total_candidates: int = 0
    candidates_above_threshold: int = 0
    processing_time: float = 0.0
    accuracy_stats: Dict[str, Any] = Field(default_factory=dict)
    jd_requirements: Dict[str, Any] = Field(default_factory=dict)
    dynamic_config: Dict[str, Any] = Field(default_factory=dict)  # NEW: Include config info

class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_details: Optional[str] = None

# Global variables for services
vector_manager: Optional[AccuracyVectorManager] = None
screening_chatbot: Optional[AccuracyScreeningChatbot] = None
dynamic_config = get_config()

# Thread-safe storage for latest results (will be replaced with proper session management in production)
latest_output_rows = []

# Helper functions for file conversion
async def convert_uploadfile_to_filelike(upload_file: UploadFile):
    """Convert FastAPI UploadFile to a file-like object with .name attribute."""
    await upload_file.seek(0)
    content = await upload_file.read()
    file_obj = io.BytesIO(content)
    file_obj.name = upload_file.filename
    return file_obj

async def convert_multiple_uploadfiles(upload_files: List[UploadFile]):
    """Convert multiple FastAPI UploadFiles to file-like objects."""
    converted_files = []
    for upload_file in upload_files:
        file_obj = await convert_uploadfile_to_filelike(upload_file)
        converted_files.append(file_obj)
    return converted_files

def format_structured_scores_for_csv_dynamic(structured_scores: Dict[str, Any], jd_requirements: Dict = None) -> Dict[str, str]:
    """Format structured scores for CSV export with FULLY DYNAMIC validation"""
    try:
        csv_scores = {}
        config = get_config()
        
        # Helper function to safely get nested values with dynamic keys
        def safe_get_dynamic(obj, path, defaultValue='N/A'):
            try:
                keys = path.split('.')
                result = obj
                for key in keys:
                    if isinstance(result, dict) and key in result:
                        result = result[key]
                    else:
                        return defaultValue
                return result if result is not None else defaultValue
            except:
                return defaultValue

        # Helper function to safely process arrays with dynamic handling
        def safe_array_dynamic(arr, defaultValue=None):
            if defaultValue is None:
                defaultValue = []
            try:
                return arr if isinstance(arr, list) else defaultValue
            except:
                return defaultValue

        # DYNAMIC Mandatory Skills Processing
        mandatory_skills = structured_scores.get("mandatory_skills", {})
        if mandatory_skills and isinstance(mandatory_skills, dict):
            csv_scores.update({
                "Skills_Score": str(safe_get_dynamic(mandatory_skills, 'score')),
                "Required_Skills": "; ".join(safe_array_dynamic(mandatory_skills.get("required_skills", []))),
                "Matched_Skills": "; ".join(safe_array_dynamic(mandatory_skills.get("matched_skills", []))),
                "Missing_Skills": "; ".join(safe_array_dynamic(mandatory_skills.get("missing_skills", []))),
                "Skills_Calculation": str(safe_get_dynamic(mandatory_skills, 'calculation')),
                "Required_Skills_Count": str(len(safe_array_dynamic(mandatory_skills.get("required_skills", [])))),
                "Matched_Skills_Count": str(len(safe_array_dynamic(mandatory_skills.get("matched_skills", [])))),
                "Missing_Skills_Count": str(len(safe_array_dynamic(mandatory_skills.get("missing_skills", []))))
            })
        else:
            # Dynamic defaults based on JD requirements if available
            default_skills_count = len(jd_requirements.get("technical_skills", [])) if jd_requirements else 0
            csv_scores.update({
                "Skills_Score": "0/50",
                "Required_Skills": "None extracted",
                "Matched_Skills": "None found",
                "Missing_Skills": f"All {default_skills_count} skills missing" if default_skills_count > 0 else "Unknown",
                "Skills_Calculation": "0/0 × 50 = 0",
                "Required_Skills_Count": str(default_skills_count),
                "Matched_Skills_Count": "0",
                "Missing_Skills_Count": str(default_skills_count)
            })

        # DYNAMIC Total Experience Processing
        total_exp = structured_scores.get("total_experience", {})
        if total_exp and isinstance(total_exp, dict):
            employment_periods = safe_array_dynamic(total_exp.get("employment_periods", []))
            csv_scores.update({
                "Total_Experience_Score": str(safe_get_dynamic(total_exp, 'score')),
                "Total_Experience_Years": str(safe_get_dynamic(total_exp, 'total_experience')),
                "Experience_Required": str(safe_get_dynamic(total_exp, 'required')),
                "Meets_Experience_Requirement": "Yes" if total_exp.get("meets_requirement", False) else "No",
                "Employment_Periods": "; ".join(employment_periods)[:500] if employment_periods else "None listed",
                "Employment_Periods_Count": str(len(employment_periods))
            })
        else:
            # Dynamic defaults
            required_exp = jd_requirements.get("min_experience_years", 0) if jd_requirements else 0
            csv_scores.update({
                "Total_Experience_Score": "0/10",
                "Total_Experience_Years": "0 years",
                "Experience_Required": f"{required_exp} years",
                "Meets_Experience_Requirement": "No",
                "Employment_Periods": "None found",
                "Employment_Periods_Count": "0"
            })

        # DYNAMIC Relevant Experience Processing
        relevant_exp = structured_scores.get("relevant_experience", {})
        if relevant_exp and isinstance(relevant_exp, dict):
            relevant_companies = safe_array_dynamic(relevant_exp.get("relevant_companies", []))
            csv_scores.update({
                "Relevant_Experience_Score": str(safe_get_dynamic(relevant_exp, 'score')),
                "Relevant_Experience_Years": str(safe_get_dynamic(relevant_exp, 'total_relevant_experience')),
                "Relevant_Experience_Threshold": str(safe_get_dynamic(relevant_exp, 'threshold')),
                "Meets_Relevant_Requirement": "Yes" if relevant_exp.get("meets_requirement", False) else "No",
                "Relevant_Companies": "; ".join(relevant_companies)[:500] if relevant_companies else "None",
                "Relevant_Companies_Count": str(len(relevant_companies))
            })
        else:
            # Dynamic defaults
            threshold = config.calculate_relevant_experience_threshold(jd_requirements.get("min_experience_years", 0)) if jd_requirements else 1
            csv_scores.update({
                "Relevant_Experience_Score": "0/20",
                "Relevant_Experience_Years": "0 years",
                "Relevant_Experience_Threshold": f"{threshold} years",
                "Meets_Relevant_Requirement": "No",
                "Relevant_Companies": "None identified",
                "Relevant_Companies_Count": "0"
            })

        # DYNAMIC Project Exposure Processing
        project_exp = structured_scores.get("project_exposure", {})
        if project_exp and isinstance(project_exp, dict):
            e2e_projects = safe_array_dynamic(project_exp.get("e2e_projects", []))
            support_projects = safe_array_dynamic(project_exp.get("support_projects", []))
            academic_projects = safe_array_dynamic(project_exp.get("academic_unrelated", []))
            
            csv_scores.update({
                "Project_Score": str(safe_get_dynamic(project_exp, 'score')),
                "E2E_Projects_Count": str(len(e2e_projects)),
                "Support_Projects_Count": str(len(support_projects)),
                "Academic_Projects_Count": str(len(academic_projects)),
                "E2E_Projects": "; ".join(e2e_projects)[:500] if e2e_projects else "None found",
                "Support_Projects": "; ".join(support_projects)[:500] if support_projects else "None found",
                "Academic_Projects": "; ".join(academic_projects)[:500] if academic_projects else "None found",
                "Project_Scoring_Logic": str(safe_get_dynamic(project_exp, 'scoring_logic')),
                "Total_Projects_Count": str(len(e2e_projects) + len(support_projects) + len(academic_projects))
            })
        else:
            # Dynamic defaults
            scoring_weights = config.get_scoring_weights()
            csv_scores.update({
                "Project_Score": f"0/{scoring_weights.project_exposure_max}",
                "E2E_Projects_Count": "0",
                "Support_Projects_Count": "0",
                "Academic_Projects_Count": "0",
                "E2E_Projects": "No end-to-end projects found",
                "Support_Projects": "No support projects found",
                "Academic_Projects": "No academic projects found",
                "Project_Scoring_Logic": "No projects identified",
                "Total_Projects_Count": "0"
            })
        
        return csv_scores
        
    except Exception as e:
        logger.error(f"Error formatting structured scores for CSV (DYNAMIC): {e}")
        # Even error responses are dynamic
        scoring_weights = get_config().get_scoring_weights()
        return {
            "Error": f"Failed to format scores: {str(e)}",
            "Skills_Score": f"ERROR/{scoring_weights.skills_max}",
            "Total_Experience_Score": f"ERROR/{scoring_weights.total_experience_max}",
            "Relevant_Experience_Score": f"ERROR/{scoring_weights.relevant_experience_max}",
            "Project_Score": f"ERROR/{scoring_weights.project_exposure_max}",
            "Processing_Status": "Dynamic formatting failed"
        }

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced accuracy-focused services on startup with dynamic configuration."""
    global vector_manager, screening_chatbot, dynamic_config
    
    try:
        # Initialize Enhanced Accuracy Vector Manager
        vector_manager = AccuracyVectorManager()
        logger.info("Enhanced AccuracyVectorManager initialized successfully (DYNAMIC)")
        
        # Initialize Enhanced Accuracy Screening Chatbot with AI agents
        screening_chatbot = AccuracyScreeningChatbot()
        logger.info("Enhanced AccuracyScreeningChatbot initialized successfully (DYNAMIC)")
        
        # Log dynamic configuration status
        scoring_weights = dynamic_config.get_scoring_weights()
        logger.info(f"DYNAMIC Configuration: Skills({scoring_weights.skills_max}), "
                   f"TotalExp({scoring_weights.total_experience_max}), "
                   f"RelevantExp({scoring_weights.relevant_experience_max}), "
                   f"Projects({scoring_weights.project_exposure_max})")
        
        logger.info("Enhanced accuracy-focused screening system ready with DYNAMIC configuration")
        
    except Exception as e:
        logger.error(f"Failed to initialize enhanced accuracy services: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the main frontend HTML page with dynamic configuration."""
    try:
        # Pass dynamic config to template
        config_data = {
            "scoring_weights": dynamic_config.get_scoring_weights().dict(),
            "display_labels": dynamic_config.get_display_labels().dict()
        }
        return templates.TemplateResponse("index.html", {
            "request": request,
            "dynamic_config": config_data
        })
    except Exception as e:
        logger.error(f"Error serving frontend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve frontend: {str(e)}")

@app.get("/ui", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Alternative route to serve the UI."""
    return await serve_frontend(request)

@app.get("/api/health")
async def health_check():
    """Perform a comprehensive health check on the enhanced accuracy system with dynamic config info."""
    config_info = {
        "scoring_weights": dynamic_config.get_scoring_weights().dict(),
        "experience_thresholds": dynamic_config.get_experience_thresholds().dict(),
        "display_labels": dynamic_config.get_display_labels().dict()
    }
    
    return {
        "status": "healthy",
        "vector_manager_ready": vector_manager is not None,
        "screening_chatbot_ready": screening_chatbot is not None,
        "version": "4.2.0",
        "architecture": "enhanced_accuracy_focused_ai_agents_dynamic",
        "dynamic_configuration": config_info,
        "features": [
            "fully_dynamic_configuration",
            "enhanced_specialized_ai_agents",
            "candidate_summary_generation_with_specifics", 
            "deterministic_calculations", 
            "evidence_based_matching",
            "search_index_caching",
            "structured_validation",
            "improved_date_parsing",
            "dynamic_structured_ui_data",
            "project_validation_fixes",
            "configurable_scoring_weights"
        ]
    }

@app.get("/api/config")
async def get_dynamic_config():
    """Get current dynamic configuration."""
    return {
        "success": True,
        "config": {
            "scoring_weights": dynamic_config.get_scoring_weights().dict(),
            "experience_thresholds": dynamic_config.get_experience_thresholds().dict(),
            "project_categories": dynamic_config.get_project_categories().dict(),
            "display_labels": dynamic_config.get_display_labels().dict()
        }
    }

@app.post("/api/config")
async def update_dynamic_config(
    skills_max: Optional[float] = Form(None),
    total_experience_max: Optional[float] = Form(None),
    relevant_experience_max: Optional[float] = Form(None),
    project_exposure_max: Optional[float] = Form(None),
    experience_buffer_low: Optional[int] = Form(None),
    experience_buffer_high: Optional[int] = Form(None)
):
    """Update dynamic configuration."""
    try:
        config_updates = {}
        if skills_max is not None:
            config_updates['skills_max'] = skills_max
        if total_experience_max is not None:
            config_updates['total_experience_max'] = total_experience_max
        if relevant_experience_max is not None:
            config_updates['relevant_experience_max'] = relevant_experience_max
        if project_exposure_max is not None:
            config_updates['project_exposure_max'] = project_exposure_max
        if experience_buffer_low is not None:
            config_updates['experience_buffer_low'] = experience_buffer_low
        if experience_buffer_high is not None:
            config_updates['experience_buffer_high'] = experience_buffer_high
        
        if config_updates:
            updated_config = update_config(**config_updates)
            logger.info(f"Dynamic configuration updated: {config_updates}")
            return {
                "success": True,
                "message": "Dynamic configuration updated successfully",
                "updated_config": {
                    "scoring_weights": updated_config.get_scoring_weights().dict(),
                    "experience_thresholds": updated_config.get_experience_thresholds().dict()
                }
            }
        else:
            return {
                "success": False,
                "message": "No configuration parameters provided"
            }
    except Exception as e:
        logger.error(f"Error updating dynamic configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.post("/api/match-candidates", response_model=EnhancedMatchingResult)
async def match_candidates(
    background_tasks: BackgroundTasks,
    job_description: str = Form(..., description="Job description text"),
    threshold: float = Form(0.0, description="Minimum matching score threshold (0-100)"),
    resume_files: List[UploadFile] = File(..., description="Resume files (PDF or DOCX)")
):
    """Match candidates using enhanced accuracy-first AI agents with DYNAMIC evaluation."""
    try:
        # Validate inputs
        if not job_description.strip():
            raise HTTPException(status_code=400, detail="Job description cannot be empty")
        
        if not resume_files:
            raise HTTPException(status_code=400, detail="At least one resume file must be uploaded")
        
        # Validate file types
        allowed_types = {"pdf", "docx"}
        for file in resume_files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="All files must have filenames")
            
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in allowed_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} has unsupported format. Only PDF and DOCX are allowed"
                )
        
        logger.info(f"Starting DYNAMIC accuracy-focused screening for {len(resume_files)} resumes")
        
        # Step 1: Process resume files using Enhanced Accuracy Vector Manager
        try:
            converted_files = await convert_multiple_uploadfiles(resume_files)
            documents_with_metadata = vector_manager.process_multiple_documents(converted_files)
            
            logger.info(f"Successfully processed {len(resume_files)} resume files")
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")
        
        if not documents_with_metadata:
            raise HTTPException(status_code=400, detail="No text found in the uploaded files")
        
        # Step 2: Index documents in search index (replaces blob storage)
        try:
            indexing_message = vector_manager.create_embeddings_from_files(converted_files)
            logger.info(f"Search index updated: {indexing_message}")
        except Exception as e:
            logger.error(f"Error updating search index: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error updating search index: {str(e)}")
        
        # Step 3: Run enhanced DYNAMIC accuracy-focused screening with AI agents
        try:
            screening_result = await screening_chatbot.process_screening_job(
                job_description=job_description,
                documents_with_metadata=documents_with_metadata,
                vector_manager=vector_manager,
                threshold=threshold
            )
            
            if not screening_result["success"]:
                raise HTTPException(status_code=500, detail=screening_result["message"])
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in enhanced DYNAMIC accuracy screening process: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Enhanced DYNAMIC accuracy screening process failed: {str(e)}")
        
        # Step 4: Prepare enhanced results for frontend and CSV export with DYNAMIC data
        candidates = screening_result.get("candidates", [])
        
        # Clear previous results (will be replaced with proper session management)
        latest_output_rows.clear()
        
        # Extract and format candidate data for enhanced CSV export with DYNAMIC processing
        output_rows = []
        formatted_candidates = []
        
        for candidate in candidates:
            personal_info = candidate.get("personal_info", {})
            structured_scores = candidate.get("structured_scores", {})
            
            # Enhanced CSV row with DYNAMIC structured data and safe access
            row = {
                "Candidate_Name": str(candidate.get("name", "Unknown")),
                "Total_Score": float(candidate.get("score", 0)),
                "Max_Possible_Score": float(dynamic_config.get_scoring_weights().get_total_max()),
                "Score_Percentage": f"{(candidate.get('score', 0) / dynamic_config.get_scoring_weights().get_total_max()) * 100:.1f}%" if dynamic_config.get_scoring_weights().get_total_max() > 0 else "0%",
                "Overall_Rating": str(candidate.get("overall_rating", "Unknown")),
                "Confidence_Level": str(candidate.get("confidence_level", "UNKNOWN")),
                "Summary": str(candidate.get("reason", "No reason provided")),
                "Key_Strengths": "; ".join(candidate.get("key_strengths", [])) if candidate.get("key_strengths") else "None identified",
                "Key_Gaps": "; ".join(candidate.get("key_gaps", [])) if candidate.get("key_gaps") else "None identified",
                "Recommendation": str(candidate.get("recommendation", "")),
                "Evaluation_Method": "Dynamic Enhanced AI Agents with Specific Summaries",
                "Processing_Time": float(screening_result.get("processing_time", 0))
            }
            
            # Add personal info dynamically
            if personal_info.get("full_name"):
                row["Full_Name"] = str(personal_info.get("full_name"))
            if personal_info.get("email_address"):
                row["Email_Address"] = str(personal_info.get("email_address"))
            if personal_info.get("mobile_number"):
                row["Mobile_Number"] = str(personal_info.get("mobile_number"))
            if personal_info.get("location"):
                row["Location"] = str(personal_info.get("location"))
            
            # Add DYNAMIC structured scores to CSV with proper validation
            jd_requirements = screening_result.get("jd_requirements", {})
            csv_structured_scores = format_structured_scores_for_csv_dynamic(structured_scores, jd_requirements)
            row.update(csv_structured_scores)
            
            output_rows.append(row)
            
            # Enhanced API response format with validation
            try:
                formatted_candidates.append(CandidateScore(
                    name=str(candidate.get("name", "Unknown")),
                    score=float(candidate.get("score", 0)),
                    reason=str(candidate.get("reason", "No summary provided")),
                    overall_rating=str(candidate.get("overall_rating", "Unknown")),
                    key_strengths=candidate.get("key_strengths", []) if isinstance(candidate.get("key_strengths"), list) else [],
                    key_gaps=candidate.get("key_gaps", []) if isinstance(candidate.get("key_gaps"), list) else [],
                    recommendation=str(candidate.get("recommendation", "")),
                    structured_scores=StructuredScores(**structured_scores) if structured_scores else StructuredScores(),
                    personal_info=personal_info if isinstance(personal_info, dict) else {},
                    confidence_level=str(candidate.get("confidence_level", "UNKNOWN")),
                    evaluation_timestamp=float(candidate.get("evaluation_timestamp", 0.0))
                ))
            except Exception as e:
                logger.error(f"Error formatting candidate {candidate.get('name', 'Unknown')}: {e}")
                # Add minimal candidate data on error with DYNAMIC info
                formatted_candidates.append(CandidateScore(
                    name=str(candidate.get("name", "Unknown")),
                    score=float(candidate.get("score", 0)),
                    reason="Error formatting candidate data - manual review recommended",
                    overall_rating="Error",
                    key_strengths=[],
                    key_gaps=["Data formatting error occurred"],
                    recommendation="Manual review required due to dynamic processing error",
                    structured_scores=StructuredScores(),
                    personal_info={},
                    confidence_level="ERROR",
                    evaluation_timestamp=float(candidate.get("evaluation_timestamp", 0.0))
                ))
        
        latest_output_rows.extend(output_rows)
        
        # Include dynamic configuration in response
        config_info = {
            "scoring_weights": dynamic_config.get_scoring_weights().dict(),
            "max_total_score": dynamic_config.get_scoring_weights().get_total_max()
        }
        
        return EnhancedMatchingResult(
            success=True,
            message=screening_result.get("message", "Enhanced DYNAMIC accuracy screening completed successfully"),
            candidates=formatted_candidates,
            total_candidates=screening_result.get("total_candidates", 0),
            candidates_above_threshold=screening_result.get("candidates_above_threshold", 0),
            processing_time=screening_result.get("processing_time", 0.0),
            accuracy_stats=screening_result.get("accuracy_stats", {}),
            jd_requirements=screening_result.get("jd_requirements", {}),
            dynamic_config=config_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in enhanced DYNAMIC accuracy matching: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/match-candidates-with-jd-file", response_model=EnhancedMatchingResult)
async def match_candidates_with_jd_file(
    background_tasks: BackgroundTasks,
    job_description_file: UploadFile = File(..., description="Job description file (PDF or DOCX)"),
    threshold: float = Form(0.0, description="Minimum matching score threshold (0-100)"),
    resume_files: List[UploadFile] = File(..., description="Resume files (PDF or DOCX)")
):
    """Match candidates against a job description file using enhanced DYNAMIC accuracy-first AI agents."""
    try:
        # Validate job description file
        if not job_description_file.filename:
            raise HTTPException(status_code=400, detail="Job description file must have a filename")
        
        jd_file_extension = job_description_file.filename.split(".")[-1].lower()
        if jd_file_extension not in {"pdf", "docx"}:
            raise HTTPException(
                status_code=400, 
                detail="Job description file must be PDF or DOCX format"
            )
        
        # Extract text from job description file using Enhanced Accuracy Vector Manager
        try:
            jd_file_obj = await convert_uploadfile_to_filelike(job_description_file)
            job_description = vector_manager.extract_text_from_file(jd_file_obj)
            logger.info(f"Extracted JD from file: {len(job_description)} characters")
        except Exception as e:
            logger.error(f"Error extracting text from job description file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing job description file: {str(e)}")
        
        if not job_description.strip():
            raise HTTPException(status_code=400, detail="Job description file appears to be empty")
        
        # Call the main matching function with extracted text
        return await match_candidates(
            background_tasks=background_tasks,
            job_description=job_description,
            threshold=threshold,
            resume_files=resume_files
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in DYNAMIC JD file matching: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/download-shortlisted")
async def download_shortlisted():
    """Download enhanced CSV of all processed candidates with DYNAMIC structured accuracy metrics."""
    global latest_output_rows

    if not latest_output_rows:
        raise HTTPException(status_code=404, detail="No candidates processed yet")

    # Generate DYNAMIC CSV columns based on actual data
    if latest_output_rows:
        # Get first row to determine available columns dynamically
        sample_row = latest_output_rows[0]
        dynamic_columns = dynamic_config.get_dynamic_csv_columns(sample_row)
        
        # Ensure all rows have consistent columns
        all_keys = set()
        for row in latest_output_rows:
            all_keys.update(row.keys())
        
        # Use dynamic columns if available, otherwise use discovered keys
        final_columns = list(all_keys) if not dynamic_columns else dynamic_columns
        
        # Fill missing keys with appropriate defaults
        for row in latest_output_rows:
            for key in final_columns:
                if key not in row:
                    row[key] = "N/A"

    # Create CSV output with DYNAMIC error handling
    output = io.StringIO()
    try:
        fieldnames = final_columns if latest_output_rows else []
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write rows with additional DYNAMIC validation
        for row in latest_output_rows:
            # Convert any complex values to strings for CSV compatibility
            clean_row = {}
            for key, value in row.items():
                if value is None:
                    clean_row[key] = "N/A"
                elif isinstance(value, (list, dict)):
                    clean_row[key] = str(value)
                elif isinstance(value, float):
                    clean_row[key] = f"{value:.2f}" if not (value != value) else "0.0"  # Handle NaN
                else:
                    clean_row[key] = str(value)
            writer.writerow(clean_row)
        
        output.seek(0)
        
        # Dynamic filename with timestamp
        timestamp = int(time.time())
        filename = f"dynamic_accuracy_screening_results_{timestamp}.csv"
        
        return StreamingResponse(
            io.StringIO(output.getvalue()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Error creating DYNAMIC CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating CSV file: {str(e)}")

@app.get("/api/stats")
async def get_processing_stats():
    """Get enhanced processing statistics and DYNAMIC accuracy metrics."""
    try:
        if vector_manager:
            stats = vector_manager.get_stats()
            config_info = {
                "scoring_weights": dynamic_config.get_scoring_weights().dict(),
                "experience_thresholds": dynamic_config.get_experience_thresholds().dict(),
                "max_total_score": dynamic_config.get_scoring_weights().get_total_max()
            }
            
            return {
                "success": True,
                "stats": stats,
                "dynamic_config": config_info,
                "enhanced_accuracy_features": {
                    "dynamic_configuration": True,
                    "configurable_scoring_weights": True,
                    "enhanced_ai_agents": True,
                    "candidate_summary_generation_with_specifics": True,
                    "evidence_based_matching": True,
                    "deterministic_calculations": True,
                    "search_index_caching": True,
                    "structured_validation": True,
                    "improved_date_parsing": True,
                    "dynamic_structured_ui_data": True,
                    "project_validation_fixes": True,
                    "improved_error_handling": True,
                    "fully_dynamic_processing": True
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Vector manager not initialized")
    except Exception as e:
        logger.error(f"Error getting enhanced DYNAMIC stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting enhanced DYNAMIC stats: {str(e)}")

@app.post("/api/validate-evaluation")
async def validate_evaluation(
    candidate_name: str = Form(..., description="Candidate name to validate"),
    jd_text: str = Form(..., description="Job description used for evaluation")
):
    """Validate a specific candidate evaluation for enhanced accuracy and consistency with DYNAMIC config."""
    try:
        if not screening_chatbot or not vector_manager:
            raise HTTPException(status_code=500, detail="Enhanced services not initialized")
        
        # Get JD requirements
        jd_requirements = vector_manager.get_cached_jd_requirements(jd_text)
        if not jd_requirements:
            raise HTTPException(status_code=404, detail="JD requirements not found in cache")
        
        # Get cached evaluation  
        import hashlib
        jd_hash = hashlib.md5(json.dumps(jd_requirements).encode()).hexdigest()
        evaluation = vector_manager.get_cached_evaluation(candidate_name, jd_hash)
        
        if not evaluation:
            raise HTTPException(status_code=404, detail="Candidate evaluation not found in cache")
        
        # Include dynamic configuration in validation response
        config_info = {
            "scoring_weights": dynamic_config.get_scoring_weights().dict(),
            "max_total_score": dynamic_config.get_scoring_weights().get_total_max()
        }
        
        return {
            "success": True,
            "candidate_name": candidate_name,
            "evaluation_found": True,
            "confidence_level": evaluation.get("confidence_level", "UNKNOWN"),
            "total_score": evaluation.get("total_score", 0),
            "max_possible_score": config_info["max_total_score"],
            "has_summary": "candidate_summary" in evaluation,
            "validation_timestamp": time.time(),
            "validation_method": "enhanced_search_index_lookup_dynamic",
            "dynamic_config": config_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating enhanced DYNAMIC evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error validating enhanced DYNAMIC evaluation: {str(e)}")

# Enhanced cleanup endpoint for development
@app.post("/api/cleanup-session")
async def cleanup_session():
    """Clean up current session resources with DYNAMIC cleanup."""
    try:
        if vector_manager:
            vector_manager.cleanup_session()
            logger.info("Enhanced DYNAMIC session cleanup completed")
            return {"success": True, "message": "Enhanced DYNAMIC session resources cleaned up successfully"}
        else:
            raise HTTPException(status_code=500, detail="Vector manager not initialized")
    except Exception as e:
        logger.error(f"Error during enhanced DYNAMIC session cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during enhanced DYNAMIC session cleanup: {str(e)}")

# Enhanced development endpoint for testing agents with DYNAMIC config
@app.post("/api/test-agents")
async def test_enhanced_agents(
    resume_text: str = Form(..., description="Resume text to test"),
    jd_text: str = Form(..., description="Job description text")
):
    """Test individual enhanced AI agents for development purposes with DYNAMIC configuration."""
    try:
        if not screening_chatbot:
            raise HTTPException(status_code=500, detail="Enhanced screening chatbot not initialized")
        
        # Test JD requirements extraction
        jd_requirements = screening_chatbot.extract_jd_requirements_agent(jd_text)
        
        # Test enhanced skill matching with dynamic config
        skills_analysis = screening_chatbot.skill_matching_agent(
            resume_text, 
            jd_requirements.get("technical_skills", []),
            jd_requirements
        )
        
        # Test enhanced experience extraction with dynamic thresholds
        experience_analysis = screening_chatbot.experience_extraction_agent(
            resume_text,
            jd_requirements.get("technical_skills", []),
            jd_requirements
        )
        
        # Test enhanced project evaluation with dynamic categorization
        project_analysis = screening_chatbot.project_evaluation_agent(
            resume_text,
            jd_requirements.get("technical_skills", []),
            jd_requirements
        )
        
        # Test enhanced candidate summary agent with specific details
        summary = screening_chatbot.candidate_summary_agent(
            "Test Candidate",
            skills_analysis,
            experience_analysis,
            project_analysis,
            75.0,  # Mock total score
            jd_requirements
        )
        
        # Include dynamic configuration in test results
        config_info = {
            "scoring_weights": dynamic_config.get_scoring_weights().dict(),
            "experience_thresholds": dynamic_config.get_experience_thresholds().dict(),
            "project_categories": dynamic_config.get_project_categories().dict()
        }
        
        return {
            "success": True,
            "jd_requirements": jd_requirements,
            "skills_analysis": skills_analysis.dict(),
            "experience_analysis": experience_analysis.dict(), 
            "project_analysis": project_analysis.dict(),
            "candidate_summary": summary.dict(),
            "dynamic_config": config_info,
            "test_timestamp": time.time(),
            "enhancements": "fully_dynamic_agents_with_specific_summaries_and_configurable_scoring"
        }
        
    except Exception as e:
        logger.error(f"Error testing enhanced DYNAMIC agents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error testing enhanced DYNAMIC agents: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            message=exc.detail,
            error_details=f"HTTP {exc.status_code}"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception in enhanced DYNAMIC system: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            message="Internal server error",
            error_details=str(exc)
        ).dict()
    )

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global vector_manager
    
    try:
        if vector_manager:
            vector_manager.cleanup_session()
            logger.info("Enhanced DYNAMIC accuracy screening system shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during enhanced DYNAMIC application shutdown: {str(e)}")

if __name__ == "__main__":
    # Ensure templates and static directories exist
    try:
        os.makedirs("templates", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        logger.info("Enhanced DYNAMIC directory setup complete")
    except Exception as e:
        logger.error(f"Failed to create directories: {str(e)}")
        raise
    
    logger.info("Starting Enhanced Accuracy-First AI Resume Screening API v4.2 - FULLY DYNAMIC...")
    
    # Run the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )