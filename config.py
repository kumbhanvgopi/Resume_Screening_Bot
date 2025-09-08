# config.py - Dynamic Configuration for Resume Screening System

import os
from typing import Dict, List, Any
from pydantic import BaseModel

class ScoringWeights(BaseModel):
    """Dynamic scoring weights configuration"""
    skills_max: float = 50.0
    total_experience_max: float = 10.0
    relevant_experience_max: float = 20.0
    project_exposure_max: float = 20.0
    
    def get_total_max(self) -> float:
        return self.skills_max + self.total_experience_max + self.relevant_experience_max + self.project_exposure_max

class ExperienceThresholds(BaseModel):
    """Dynamic experience threshold configuration"""
    low_experience_buffer: int = 1  # For <= 4 years required
    high_experience_buffer: int = 2  # For > 4 years required
    minimum_threshold: int = 1

class ProjectCategories(BaseModel):
    """Dynamic project categorization"""
    e2e_keywords: List[str] = [
        "implementation", "development", "end-to-end", "full project", 
        "complete", "rollout", "greenfield", "migration", "deployment"
    ]
    support_keywords: List[str] = [
        "support", "maintenance", "bug fix", "enhancement", "troubleshooting",
        "monitoring", "optimization", "upgrade"
    ]
    academic_keywords: List[str] = [
        "academic", "university", "college", "thesis", "research", 
        "coursework", "assignment", "personal project"
    ]

class DisplayLabels(BaseModel):
    """Dynamic display labels"""
    skills_section: str = "Technical Skills Analysis"
    experience_section: str = "Work Experience Analysis"
    projects_section: str = "Project Experience Analysis"
    personal_info_section: str = "Contact Information"

class DynamicConfig:
    """Main dynamic configuration class"""
    
    def __init__(self):
        self.scoring_weights = ScoringWeights()
        self.experience_thresholds = ExperienceThresholds()
        self.project_categories = ProjectCategories()
        self.display_labels = DisplayLabels()
        
        # Load from environment if available
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Scoring weights
        if os.getenv('SKILLS_MAX_SCORE'):
            self.scoring_weights.skills_max = float(os.getenv('SKILLS_MAX_SCORE'))
        if os.getenv('EXPERIENCE_MAX_SCORE'):
            self.scoring_weights.total_experience_max = float(os.getenv('EXPERIENCE_MAX_SCORE'))
        if os.getenv('RELEVANT_EXP_MAX_SCORE'):
            self.scoring_weights.relevant_experience_max = float(os.getenv('RELEVANT_EXP_MAX_SCORE'))
        if os.getenv('PROJECTS_MAX_SCORE'):
            self.scoring_weights.project_exposure_max = float(os.getenv('PROJECTS_MAX_SCORE'))
    
    def get_scoring_weights(self) -> ScoringWeights:
        """Get current scoring weights"""
        return self.scoring_weights
    
    def get_experience_thresholds(self) -> ExperienceThresholds:
        """Get current experience thresholds"""
        return self.experience_thresholds
    
    def get_project_categories(self) -> ProjectCategories:
        """Get current project categories"""
        return self.project_categories
    
    def get_display_labels(self) -> DisplayLabels:
        """Get current display labels"""
        return self.display_labels
    
    def calculate_relevant_experience_threshold(self, required_years: int) -> int:
        """Calculate relevant experience threshold dynamically"""
        if required_years <= 4:
            return max(self.experience_thresholds.minimum_threshold, 
                      required_years - self.experience_thresholds.low_experience_buffer)
        else:
            return max(self.experience_thresholds.minimum_threshold, 
                      required_years - self.experience_thresholds.high_experience_buffer)
    
    def get_score_breakdown_template(self, jd_requirements: Dict) -> Dict[str, Any]:
        """Generate dynamic score breakdown template based on JD"""
        return {
            "skills": {
                "max_points": self.scoring_weights.skills_max,
                "required_count": len(jd_requirements.get("technical_skills", [])),
                "weight_per_skill": self.scoring_weights.skills_max / max(1, len(jd_requirements.get("technical_skills", [])))
            },
            "total_experience": {
                "max_points": self.scoring_weights.total_experience_max,
                "required_years": jd_requirements.get("min_experience_years", 0)
            },
            "relevant_experience": {
                "max_points": self.scoring_weights.relevant_experience_max,
                "threshold_years": self.calculate_relevant_experience_threshold(jd_requirements.get("min_experience_years", 0))
            },
            "projects": {
                "max_points": self.scoring_weights.project_exposure_max,
                "categories": ["e2e", "support", "academic"]
            }
        }
    
    def get_dynamic_csv_columns(self, evaluation_data: Dict) -> List[str]:
        """Generate dynamic CSV columns based on actual evaluation data"""
        base_columns = [
            "Candidate_Name", "Total_Score", "Overall_Rating", "Confidence_Level", 
            "Summary", "Key_Strengths", "Key_Gaps", "Recommendation"
        ]
        
        # Add personal info columns if available
        personal_info = evaluation_data.get("personal_info", {})
        if personal_info.get("full_name"):
            base_columns.append("Full_Name")
        if personal_info.get("email_address"):
            base_columns.append("Email_Address")
        if personal_info.get("mobile_number"):
            base_columns.append("Mobile_Number")
        if personal_info.get("location"):
            base_columns.append("Location")
        
        # Add skill-specific columns dynamically
        structured_scores = evaluation_data.get("structured_scores", {})
        if "mandatory_skills" in structured_scores:
            base_columns.extend([
                "Skills_Score", "Required_Skills_Count", "Matched_Skills_Count", 
                "Missing_Skills_Count", "Matched_Skills", "Missing_Skills", "Skills_Calculation"
            ])
        
        # Add experience columns dynamically
        if "total_experience" in structured_scores:
            base_columns.extend([
                "Total_Experience_Score", "Total_Experience_Years", "Experience_Required",
                "Meets_Experience_Requirement", "Employment_Periods"
            ])
        
        if "relevant_experience" in structured_scores:
            base_columns.extend([
                "Relevant_Experience_Score", "Relevant_Experience_Years", 
                "Relevant_Experience_Threshold", "Meets_Relevant_Requirement", "Relevant_Companies"
            ])
        
        # Add project columns dynamically
        if "project_exposure" in structured_scores:
            pe = structured_scores["project_exposure"]
            base_columns.extend([
                "Project_Score", "E2E_Projects_Count", "Support_Projects_Count", 
                "Academic_Projects_Count", "Project_Scoring_Logic"
            ])
            
            # Add specific project lists if they exist
            if pe.get("e2e_projects"):
                base_columns.append("E2E_Projects")
            if pe.get("support_projects"):
                base_columns.append("Support_Projects")
            if pe.get("academic_unrelated"):
                base_columns.append("Academic_Projects")
        
        base_columns.append("Evaluation_Method")
        base_columns.append("Processing_Time")
        
        return base_columns

# Global configuration instance
dynamic_config = DynamicConfig()

def get_config() -> DynamicConfig:
    """Get the global configuration instance"""
    return dynamic_config

def update_config(**kwargs):
    """Update configuration dynamically"""
    config = get_config()
    
    # Update scoring weights
    if 'skills_max' in kwargs:
        config.scoring_weights.skills_max = kwargs['skills_max']
    if 'total_experience_max' in kwargs:
        config.scoring_weights.total_experience_max = kwargs['total_experience_max']
    if 'relevant_experience_max' in kwargs:
        config.scoring_weights.relevant_experience_max = kwargs['relevant_experience_max']
    if 'project_exposure_max' in kwargs:
        config.scoring_weights.project_exposure_max = kwargs['project_exposure_max']
    
    # Update thresholds
    if 'experience_buffer_low' in kwargs:
        config.experience_thresholds.low_experience_buffer = kwargs['experience_buffer_low']
    if 'experience_buffer_high' in kwargs:
        config.experience_thresholds.high_experience_buffer = kwargs['experience_buffer_high']
    
    return config