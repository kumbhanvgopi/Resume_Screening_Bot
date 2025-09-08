# chatbot.py - Fixed Accuracy-First AI Agent Screening System - Anti-Hallucination

import os
import json
import re
import logging
import time
import asyncio
import requests
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from pydantic import BaseModel, Field, validator
from openai import AzureOpenAI
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Try to import Levenshtein, fallback if not available
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    logging.warning("Levenshtein not available. Install with: pip install python-Levenshtein")

# Import dynamic configuration
from config import get_config, DynamicConfig

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.azuresearch import AzureSearch
    VECTORSTORES_AVAILABLE = True
except ImportError:
    VECTORSTORES_AVAILABLE = False
    logging.warning("Vector stores not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Pydantic models for structured outputs
class SkillMatch(BaseModel):
    skill_name: str
    matched: bool
    evidence_snippet: str = ""
    confidence_score: float = Field(ge=0.0, le=1.0)
    context: str = ""
    verification_status: str = "unverified"  # "verified", "unverified", "failed"

class SkillAnalysis(BaseModel):
    matched_skills: List[SkillMatch]
    missing_skills: List[SkillMatch]
    total_required_skills: int
    matched_count: int
    missing_count: int
    confidence_score: float = Field(ge=0.0, le=1.0)
    verification_stats: Dict[str, int] = Field(default_factory=dict)

class WorkPeriod(BaseModel):
    company: str
    role: str
    start_date: str
    end_date: str
    duration_months: int
    is_relevant: bool = False

class ExperienceAnalysis(BaseModel):
    work_periods: List[WorkPeriod]
    relevant_periods: List[WorkPeriod]
    total_months: int
    total_years: float
    relevant_months: int
    relevant_years: float
    meets_total_requirement: bool
    meets_relevant_requirement: bool

class ProjectAnalysis(BaseModel):
    e2e_projects: List[str]
    support_projects: List[str]
    academic_projects: List[str]
    project_score: float = Field(ge=0, le=100)  # Dynamic max score
    justification: str

class PersonalInfo(BaseModel):
    full_name: Optional[str] = None
    email_address: Optional[str] = None
    mobile_number: Optional[str] = None
    location: Optional[str] = None

class CandidateSummary(BaseModel):
    candidate_name: str
    overall_rating: str  # "Good match", "Partial match", "Poor match"
    summary_text: str
    key_strengths: List[str]
    key_gaps: List[str]
    recommendation: str

class CandidateEvaluation(BaseModel):
    candidate_name: str
    candidate_summary: CandidateSummary
    skills_analysis: SkillAnalysis
    experience_analysis: ExperienceAnalysis
    project_analysis: ProjectAnalysis
    personal_info: PersonalInfo
    total_score: float
    confidence_level: str
    evaluation_timestamp: float

class AccuracyScreeningChatbot:
    """Fixed Accuracy-First AI Agent System for Resume Screening - Anti-Hallucination"""
    
    def __init__(
        self,
        api_key: str = None,
        endpoint: str = None,
        api_version: str = None,
        deployment_name: str = None,
        deployment_mini: str = None,
        deployment_embedding: str = None,
        azure_search_endpoint: str = None,
        azure_search_key: str = None,
        azure_search_index_name: str = None
    ):
        """Initialize the fixed accuracy-focused screening system with dynamic configuration"""
        
        # Azure OpenAI configuration
        self.api_key = api_key or os.getenv("API_KEY")
        self.endpoint = endpoint or os.getenv("ENDPOINT")
        self.api_version = api_version or os.getenv("API_VERSION", "2024-12-01-preview")
        self.deployment_name = deployment_name or os.getenv("DEPLOYMENT")
        self.deployment_mini = deployment_mini or os.getenv("DEPLOYMENT_mini")
        self.deployment_embedding = deployment_embedding or os.getenv("DEPLOYMENT_EMB")
        
        # Azure Search configuration
        self.azure_search_endpoint = azure_search_endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
        self.azure_search_key = azure_search_key or os.getenv("AZURE_SEARCH_KEY")
        self.azure_search_index_name = azure_search_index_name or os.getenv("AZURE_SEARCH_INDEX_NAME")
        
        # Dynamic configuration
        self.config = get_config()
        
        # Validate required configuration
        if not all([self.api_key, self.endpoint, self.deployment_name]):
            raise ValueError("Missing required Azure OpenAI configuration. Check environment variables.")
        
        # Initialize components
        self._initialize_clients()
        
        logger.info("AccuracyScreeningChatbot initialized with ANTI-HALLUCINATION configuration")
    
    def _initialize_clients(self) -> None:
        """Initialize Azure OpenAI clients"""
        try:
            # Main Azure OpenAI client for completions
            self.azure_openai_client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
            
            # LangChain chat model for QA chains
            self.llm = AzureChatOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                model_name="o4-mini",
                temperature=0
            )
            
            # Embedding model
            if self.deployment_embedding:
                self.embedding_model = AzureOpenAIEmbeddings(
                    deployment=self.deployment_embedding,
                    model="text-embedding-3-small",
                    chunk_size=1,
                    api_key=self.api_key,
                    azure_endpoint=self.endpoint,
                    api_version=self.api_version
                )
            
            logger.info("Azure OpenAI clients initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI clients: {e}")
            raise ConnectionError(f"Azure OpenAI initialization failed: {e}")

    # ==================== ENHANCED DATE PARSING ====================
    
    def parse_date_with_fallbacks(self, date_str: str) -> datetime:
        """Fixed date parsing with safer fallback patterns"""
        if not date_str or not isinstance(date_str, str):
            return datetime.now()
        
        date_str = str(date_str).strip().lower()
        
        # Handle common end date patterns
        current_patterns = [
            'present', 'current', 'ongoing', 'till date', 'till today', 
            'to date', 'continuing', 'now', 'today', 'currently'
        ]
        
        if any(pattern in date_str for pattern in current_patterns):
            return datetime.now()
        
        # Try dateutil parser first (most reliable)
        try:
            return date_parser.parse(date_str, fuzzy=True)
        except:
            pass
        
        # Try manual patterns with better error handling
        patterns = [
            r'(\d{1,2})[/-](\d{4})',  # MM/YYYY or MM-YYYY
            r'(\d{4})[/-](\d{1,2})',  # YYYY/MM or YYYY-MM
            r'(\w+)[,\s]+(\d{4})',    # Month YYYY
            r'(\d{4})',               # Just year
        ]
        
        for pattern in patterns:
            try:
                match = re.search(pattern, date_str)
                if match:
                    groups = match.groups()
                    if len(groups) == 2:
                        if groups[1].isdigit() and len(groups[1]) == 4:  # Year is second
                            year = int(groups[1])
                            if 1900 <= year <= 2030:  # Sanity check
                                if groups[0].isdigit():
                                    month = max(1, min(12, int(groups[0])))
                                else:
                                    month = 1
                                return datetime(year, month, 1)
                        elif groups[0].isdigit() and len(groups[0]) == 4:  # Year is first
                            year = int(groups[0])
                            if 1900 <= year <= 2030:  # Sanity check
                                month = max(1, min(12, int(groups[1]))) if groups[1].isdigit() else 1
                                return datetime(year, month, 1)
                    elif len(groups) == 1 and groups[0].isdigit():
                        year = int(groups[0])
                        if 1900 <= year <= 2030:  # Sanity check
                            return datetime(year, 1, 1)
            except Exception as e:
                logger.warning(f"Date pattern matching failed for '{date_str}': {e}")
                continue
        
        # Last fallback - return current date
        logger.warning(f"Could not parse date: '{date_str}', using current date")
        return datetime.now()

    def calculate_experience_months(self, start_date_str: str, end_date_str: str) -> int:
        """Fixed calculation of work experience in months"""
        try:
            if not start_date_str or not end_date_str:
                return 0
                
            end_date = self.parse_date_with_fallbacks(end_date_str)
            start_date = self.parse_date_with_fallbacks(start_date_str)
            
            # Ensure start is before end
            if start_date > end_date:
                start_date, end_date = end_date, start_date
            
            # Calculate months difference
            months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            
            # Add 1 to include the current month, but cap at reasonable values
            months = max(1, min(months + 1, 600))  # Cap at 50 years
            
            return months
            
        except Exception as e:
            logger.warning(f"Date parsing failed for {start_date_str} to {end_date_str}: {e}")
            return 0

    # ==================== EVIDENCE VERIFICATION FUNCTIONS ====================
    
    def _generate_skill_variants(self, skill: str) -> List[str]:
        """Generate common variants of a skill for better matching"""
        skill = skill.lower().strip()
        variants = [skill]
        
        # Common programming and technical variants
        common_variants = {
            'javascript': ['js', 'javascript', 'java script', 'node.js', 'nodejs'],
            'js': ['javascript', 'java script', 'node.js', 'nodejs'],
            'python': ['py', 'python3', 'python2'],
            'typescript': ['ts', 'type script'],
            'react': ['reactjs', 'react.js', 'react js'],
            'angular': ['angularjs', 'angular.js', 'angular js'],
            'vue': ['vuejs', 'vue.js', 'vue js'],
            'postgresql': ['postgres', 'psql', 'postgresql'],
            'mysql': ['my sql', 'mysql'],
            'mongodb': ['mongo db', 'mongo', 'mongodb'],
            'amazon web services': ['aws', 'amazon aws'],
            'google cloud platform': ['gcp', 'google cloud', 'gcloud'],
            'microsoft azure': ['azure', 'azure cloud'],
            'docker': ['docker containers', 'containerization'],
            'kubernetes': ['k8s', 'k8s orchestration'],
            'git': ['github', 'git version control'],
            'html': ['html5', 'hypertext markup language'],
            'css': ['css3', 'cascading style sheets'],
            'rest api': ['rest', 'restful', 'rest apis', 'restful api'],
            'rest': ['rest api', 'restful', 'rest apis', 'restful api'],
            'api': ['apis', 'web api', 'rest api'],
            'sql': ['structured query language', 'sql queries'],
            'nosql': ['no sql', 'nosql databases'],
            'machine learning': ['ml', 'artificial intelligence', 'ai'],
            'artificial intelligence': ['ai', 'machine learning', 'ml'],
            'devops': ['dev ops', 'ci/cd', 'continuous integration'],
            'ci/cd': ['continuous integration', 'continuous deployment', 'devops'],
            'microservices': ['micro services', 'service oriented architecture'],
            'graphql': ['graph ql', 'graphql api'],
            'jenkins': ['jenkins ci', 'jenkins pipeline'],
            'terraform': ['infrastructure as code', 'iac'],
            'webpack': ['web pack', 'module bundler'],
            'npm': ['node package manager', 'package manager'],
            'yarn': ['yarn package manager', 'package manager'],
            'elasticsearch': ['elastic search', 'elk stack'],
            'redis': ['redis cache', 'caching'],
            'nginx': ['web server', 'reverse proxy'],
            'apache': ['apache web server', 'httpd'],
            'linux': ['unix', 'ubuntu', 'centos', 'rhel'],
            'windows': ['windows server', 'microsoft windows'],
            'agile': ['scrum', 'agile methodology', 'agile development'],
            'scrum': ['agile', 'agile methodology', 'agile development']
        }
        
        # Handle space/dash/underscore variations
        if ' ' in skill:
            variants.append(skill.replace(' ', ''))
            variants.append(skill.replace(' ', '-'))
            variants.append(skill.replace(' ', '_'))
        if '-' in skill:
            variants.append(skill.replace('-', ' '))
            variants.append(skill.replace('-', ''))
        if '_' in skill:
            variants.append(skill.replace('_', ' '))
            variants.append(skill.replace('_', ''))
        
        # Add specific variants if they exist
        if skill in common_variants:
            variants.extend(common_variants[skill])
        
        # Add capitalization variants for longer skills
        if len(skill) > 2:
            variants.append(skill.upper())
            variants.append(skill.capitalize())
            variants.append(skill.title())
        
        # Remove duplicates and return
        return list(set(variants))

    def _fuzzy_match_text(self, needle: str, haystack: str, threshold: float = 0.6) -> Tuple[bool, float, str]:
        """Enhanced fuzzy match text with better technical term recognition"""
        if not needle or not haystack:
            return False, 0.0, ""
        
        needle = needle.strip().lower()
        haystack = haystack.lower()
        
        # Method 1: Direct substring match (highest confidence)
        if needle in haystack:
            return True, 1.0, "exact_match"
        
        # Method 2: Handle common technical abbreviations and variations
        needle_variants = self._generate_skill_variants(needle)
        for variant in needle_variants:
            if variant in haystack:
                return True, 0.9, f"variant_match: {variant}"
        
        # Method 3: Word-level matching for technical terms
        needle_words = set(needle.split())
        haystack_words = set(haystack.split())
        
        # For technical terms, check if main words are present
        if len(needle_words) <= 3:  # Short technical terms
            word_overlap = len(needle_words.intersection(haystack_words)) / len(needle_words)
            if word_overlap >= 0.7:  # 70% word overlap
                return True, word_overlap, f"word_overlap: {word_overlap:.2f}"
        
        # Method 4: Levenshtein distance for short strings with lower threshold
        if len(needle) < 100 and HAS_LEVENSHTEIN:
            try:
                ratio = Levenshtein.ratio(needle, haystack)
                # More lenient threshold for technical terms
                adjusted_threshold = max(0.5, threshold - 0.1)
                if ratio >= adjusted_threshold:
                    return True, ratio, "levenshtein"
            except:
                pass
        
        # Method 5: Find best matching substring with lower threshold
        try:
            best_ratio = 0.0
            best_match = ""
            
            # Split haystack into sentences/chunks
            sentences = re.split(r'[.!?]\s+', haystack)
            
            for sentence in sentences:
                if len(sentence.strip()) < 10:
                    continue
                    
                ratio = SequenceMatcher(None, needle, sentence.strip()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = sentence.strip()
            
            # More lenient threshold for sentence matching
            sentence_threshold = max(0.4, threshold - 0.2)
            if best_ratio >= sentence_threshold:
                return True, best_ratio, f"sentence_match: {best_match[:100]}"
                
        except Exception as e:
            logger.warning(f"Fuzzy matching failed: {e}")
        
        return False, 0.0, "no_match"

    def _verify_skill_evidence(self, skill_name: str, evidence_snippet: str, full_resume_text: str) -> Tuple[bool, float, str]:
        """Enhanced verify that claimed skill evidence actually exists in the resume"""
        if not skill_name or not evidence_snippet or not full_resume_text:
            return False, 0.0, "missing_data"
        
        try:
            # Clean inputs
            skill_name = skill_name.strip().lower()
            evidence_snippet = evidence_snippet.strip()
            
            # Check if evidence exists in resume with more lenient matching
            evidence_found, evidence_score, evidence_method = self._fuzzy_match_text(
                evidence_snippet, full_resume_text, threshold=0.5  # More lenient for evidence
            )
            
            if not evidence_found:
                # Try shorter snippets from evidence
                words = evidence_snippet.split()
                if len(words) > 3:
                    shorter_evidence = ' '.join(words[:3])  # Try first 3 words
                    evidence_found, evidence_score, evidence_method = self._fuzzy_match_text(
                        shorter_evidence, full_resume_text, threshold=0.6
                    )
            
            if not evidence_found:
                return False, 0.0, f"evidence_not_found: {evidence_method}"
            
            # Enhanced skill verification with variants
            skill_found = False
            skill_score = 0.0
            skill_method = ""
            
            # Generate skill variants for better matching
            skill_variants = self._generate_skill_variants(skill_name)
            
            # Check each variant in the evidence snippet
            for variant in skill_variants:
                variant_found, variant_score, variant_method = self._fuzzy_match_text(
                    variant, evidence_snippet, threshold=0.4  # Lower threshold for variants
                )
                if variant_found and variant_score > skill_score:
                    skill_found = True
                    skill_score = variant_score
                    skill_method = f"variant:{variant}({variant_method})"
            
            if not skill_found:
                # Try broader context around evidence with all variants
                context_size = 250  # Larger context
                evidence_lower = evidence_snippet.lower()
                evidence_start = full_resume_text.lower().find(evidence_lower[:30])
                
                if evidence_start >= 0:
                    context_start = max(0, evidence_start - context_size)
                    context_end = min(len(full_resume_text), evidence_start + len(evidence_snippet) + context_size)
                    context = full_resume_text[context_start:context_end]
                    
                    for variant in skill_variants:
                        variant_found, variant_score, variant_method = self._fuzzy_match_text(
                            variant, context, threshold=0.3  # Even lower threshold for context
                        )
                        if variant_found and variant_score > skill_score:
                            skill_found = True
                            skill_score = variant_score
                            skill_method = f"context_variant:{variant}({variant_method})"
                
                # Last resort: check if skill appears anywhere in resume
                if not skill_found:
                    for variant in skill_variants:
                        if variant in full_resume_text.lower():
                            skill_found = True
                            skill_score = 0.7  # Moderate confidence for direct appearance
                            skill_method = f"resume_scan:{variant}"
                            break
            
            # Calculate final confidence
            base_confidence = (evidence_score + skill_score) / 2.0 if skill_found else evidence_score * 0.3
            
            # Ensure minimum confidence threshold
            if skill_found and base_confidence < 0.5:
                base_confidence = 0.5
            
            verification_details = f"evidence:{evidence_method}, skill:{skill_method}, conf:{base_confidence:.2f}"
            
            return skill_found, base_confidence, verification_details
            
        except Exception as e:
            logger.error(f"Evidence verification failed for skill '{skill_name}': {e}")
            return False, 0.0, f"verification_error: {str(e)}"

    # ==================== ENHANCED AI AGENTS ====================
    
    def extract_jd_requirements_agent(self, jd_text: str) -> Dict:
        """Enhanced JD parsing with better skill recognition for any domain"""
        
        if not jd_text or len(jd_text.strip()) < 50:
            raise ValueError("Job description is too short or empty")
        
        prompt = f"""
You are a specialized job description parser. Extract technical skills and experience requirements with precision.

Job Description:
{jd_text}

CRITICAL RULES:
1. Extract ONLY specific technical skills (programming languages, frameworks, tools, databases, technologies)
2. Preserve exact terminology and acronyms as written in the JD
3. Include both full names and abbreviations if mentioned (e.g., "JavaScript" and "JS")
4. Do NOT include: soft skills, general terms, job responsibilities, business domains
5. Be specific: "Java" not "programming", "React" not "frontend development"
6. Exclude: "experience", "knowledge", "understanding", "ability", "skills"
7. Only include concrete technical terms that can be verified in a resume
8. Extract the minimum years of experience mentioned
9. List project/domain keywords for relevance checking

Examples of VALID skills: Python, Java, React, AWS, Docker, MySQL, Kubernetes, ABAP, CDS Views, Angular
Examples of INVALID skills: programming, development, experience, leadership, communication, good understanding

Return ONLY valid JSON (no markdown, no explanations):
{{
  "technical_skills": ["list of specific technical skills only"],
  "min_experience_years": integer_value,
  "project_keywords": ["list of project/domain keywords"],
  "total_technical_skills_count": integer_count,
  "critical_skills": ["first 8 most important skills"],
  "preferred_skills": ["remaining skills if any"]
}}
"""

        try:
            response = self.azure_openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a precise technical requirement extractor. Extract exact terminology and technical skills only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2000
            )

            raw_output = response.choices[0].message.content.strip()
            
            # Clean response more thoroughly
            raw_output = re.sub(r'^```json\s*', '', raw_output)
            raw_output = re.sub(r'\s*```$', '', raw_output)
            raw_output = re.sub(r'^```\s*', '', raw_output)
            
            jd_data = json.loads(raw_output)
            
            # Enhanced validation and cleaning
            technical_skills = jd_data.get("technical_skills", [])
            
            # Filter and clean skills
            valid_skills = []
            invalid_terms = {
                'experience', 'knowledge', 'understanding', 'ability', 'skills',
                'development', 'programming', 'coding', 'software', 'application',
                'system', 'technology', 'platform', 'tool', 'framework',
                'language', 'database', 'good', 'strong', 'excellent', 'must',
                'required', 'preferred', 'should', 'will', 'can', 'need',
                'working', 'hands-on', 'practical', 'thorough'
            }
            
            for skill in technical_skills:
                if isinstance(skill, str):
                    skill_clean = skill.strip()
                    skill_lower = skill_clean.lower()
                    
                    # Skip invalid terms
                    if (len(skill_clean) < 2 or 
                        skill_lower in invalid_terms or
                        any(invalid in skill_lower for invalid in ['experience', 'knowledge', 'ability', 'understanding']) or
                        len(skill_lower.split()) > 4):  # Limit to 4 words max
                        continue
                    
                    # Skip if it's just generic terms
                    if skill_lower in ['programming', 'development', 'coding', 'software']:
                        continue
                    
                    # Keep valid technical skills
                    if skill_clean not in valid_skills:
                        valid_skills.append(skill_clean)
            
            # Remove duplicates while preserving order and handle case variations
            seen = set()
            unique_skills = []
            for skill in valid_skills:
                skill_key = skill.lower().strip()
                if skill_key not in seen:
                    unique_skills.append(skill)
                    seen.add(skill_key)
            
            # Update with cleaned skills
            jd_data["technical_skills"] = unique_skills[:25]  # Limit to 25 skills max
            jd_data["total_technical_skills_count"] = len(unique_skills)
            jd_data.setdefault("min_experience_years", 0)
            jd_data.setdefault("project_keywords", [])
            jd_data["critical_skills"] = unique_skills[:8]  # First 8 as critical
            jd_data["preferred_skills"] = unique_skills[8:20]
            
            if len(unique_skills) == 0:
                raise ValueError("No valid technical skills found in job description")
            
            logger.info(f"Enhanced JD Agent: Extracted {len(unique_skills)} validated technical skills")
            return jd_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JD parsing JSON error: {e}\nRaw output: {raw_output}")
            raise ValueError(f"Failed to parse job description requirements: Invalid JSON response")
        except Exception as e:
            logger.error(f"JD requirements extraction failed: {str(e)}")
            raise ValueError(f"Failed to parse job description requirements: {str(e)}")

    def skill_matching_agent(self, resume_text: str, required_skills: List[str], jd_requirements: Dict) -> SkillAnalysis:
        """FIXED skill matching agent with evidence verification and anti-hallucination measures"""
        
        if not resume_text or len(resume_text.strip()) < 100:
            raise ValueError("Resume text is too short or empty")
            
        if not required_skills:
            raise ValueError("No required skills provided")
        
        # Step 1: Extract skills that actually exist in the resume
        extracted_skills = self._extract_actual_resume_skills(resume_text)
        
        # Step 2: Match extracted skills against required skills
        matched_skills = []
        missing_skills = []
        verification_stats = {"verified": 0, "unverified": 0, "failed": 0}
        
        for skill in required_skills:
            # Find potential matches in extracted skills
            skill_match = self._find_skill_match(skill, extracted_skills, resume_text)
            
            if skill_match["matched"]:
                # Verify the evidence
                verified, confidence, verification_details = self._verify_skill_evidence(
                    skill, skill_match["evidence"], resume_text
                )
                
                if verified and confidence >= 0.5:  # Lowered threshold
                    matched_skills.append(SkillMatch(
                        skill_name=skill,
                        matched=True,
                        evidence_snippet=skill_match["evidence"],
                        confidence_score=confidence,
                        context=skill_match["context"],
                        verification_status="verified"
                    ))
                    verification_stats["verified"] += 1
                else:
                    # Evidence couldn't be verified - mark as missing
                    missing_skills.append(SkillMatch(
                        skill_name=skill,
                        matched=False,
                        evidence_snippet="",
                        confidence_score=0.0,
                        context=f"Evidence verification failed: {verification_details}",
                        verification_status="failed"
                    ))
                    verification_stats["failed"] += 1
            else:
                missing_skills.append(SkillMatch(
                    skill_name=skill,
                    matched=False,
                    evidence_snippet="",
                    confidence_score=0.0,
                    context="No evidence found in resume",
                    verification_status="unverified"
                ))
                verification_stats["unverified"] += 1
        
        matched_count = len(matched_skills)
        missing_count = len(missing_skills)
        avg_confidence = sum(s.confidence_score for s in matched_skills) / max(1, matched_count)
        
        analysis = SkillAnalysis(
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            total_required_skills=len(required_skills),
            matched_count=matched_count,
            missing_count=missing_count,
            confidence_score=avg_confidence,
            verification_stats=verification_stats
        )
        
        logger.info(f"Skill Agent: Matched {matched_count}/{len(required_skills)} skills (Verified: {verification_stats['verified']}, Failed: {verification_stats['failed']})")
        return analysis

    def _extract_actual_resume_skills(self, resume_text: str) -> List[Dict[str, str]]:
        """Extract skills that actually exist in the resume text"""
        
        prompt = f"""
Extract ONLY technical skills that are explicitly mentioned in this resume text. 
Do NOT infer, assume, or add skills that are not directly stated.

Resume Text:
{resume_text}

CRITICAL INSTRUCTIONS:
1. Find ONLY technical skills explicitly mentioned in the text
2. For each skill, provide the EXACT text snippet where it appears
3. Include programming languages, frameworks, tools, databases, technologies
4. Do NOT include soft skills, general terms, or implied skills
5. Provide the exact quote from the resume as evidence

Return JSON with this EXACT structure:
{{
  "extracted_skills": [
    {{
      "skill_name": "exact skill name as written",
      "evidence_snippet": "exact quote from resume containing this skill",
      "context": "surrounding text for context"
    }}
  ]
}}
"""

        try:
            response = self.azure_openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a precise skill extractor. Only extract skills explicitly mentioned with exact quotes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2000
            )

            raw_output = response.choices[0].message.content.strip()
            raw_output = re.sub(r'^```json\s*', '', raw_output)
            raw_output = re.sub(r'\s*```$', '', raw_output)
            
            result = json.loads(raw_output)
            extracted_skills = result.get("extracted_skills", [])
            
            # Validate each extracted skill
            valid_skills = []
            for skill_data in extracted_skills:
                if (isinstance(skill_data, dict) and 
                    skill_data.get("skill_name") and 
                    skill_data.get("evidence_snippet")):
                    valid_skills.append(skill_data)
            
            logger.info(f"Extracted {len(valid_skills)} skills from resume")
            return valid_skills
            
        except Exception as e:
            logger.error(f"Skill extraction failed: {e}")
            return []

    def _find_skill_match(self, required_skill: str, extracted_skills: List[Dict], resume_text: str) -> Dict:
        """Enhanced skill matching with better variant recognition and pattern matching"""
        
        required_skill_lower = required_skill.lower().strip()
        
        # Direct matching first with extracted skills
        for extracted in extracted_skills:
            extracted_skill = extracted.get("skill_name", "").lower().strip()
            
            # Exact match
            if required_skill_lower == extracted_skill:
                return {
                    "matched": True,
                    "evidence": extracted.get("evidence_snippet", ""),
                    "context": extracted.get("context", "")
                }
            
            # Enhanced equivalency check
            if self._are_skills_equivalent(required_skill_lower, extracted_skill):
                return {
                    "matched": True,
                    "evidence": extracted.get("evidence_snippet", ""),
                    "context": f"Equivalent match: {extracted_skill} ~ {required_skill_lower}"
                }
        
        # Enhanced pattern matching in full resume text
        skill_variants = self._generate_skill_variants(required_skill_lower)
        
        for variant in skill_variants:
            # Create multiple search patterns for each variant
            patterns = [
                rf'\b{re.escape(variant)}\b',  # Exact word boundary
                rf'{re.escape(variant.replace(" ", ""))}',  # No spaces
                rf'{re.escape(variant.replace("-", " "))}',  # Dash to space
                rf'{re.escape(variant.replace("_", " "))}',  # Underscore to space
            ]
            
            # Add pattern for skills with common suffixes/prefixes
            if len(variant.split()) == 1 and len(variant) > 3:
                patterns.extend([
                    rf'{re.escape(variant)}(?:\s+(?:development|developer|programming|experience|skills?))?',
                    rf'(?:experience\s+(?:in|with)\s+)?{re.escape(variant)}',
                    rf'(?:knowledge\s+(?:of|in)\s+)?{re.escape(variant)}',
                ])
            
            for pattern in patterns:
                try:
                    matches = list(re.finditer(pattern, resume_text, re.IGNORECASE))
                    if matches:
                        match = matches[0]  # Take first match
                        start = max(0, match.start() - 100)
                        end = min(len(resume_text), match.end() + 100)
                        context = resume_text[start:end].strip()
                        
                        return {
                            "matched": True,
                            "evidence": context,
                            "context": f"Pattern match: {pattern} -> {variant}"
                        }
                except re.error as e:
                    logger.warning(f"Regex pattern error: {pattern} - {e}")
                    continue
        
        # Fallback: Word-level matching for multi-word skills
        skill_words = required_skill_lower.split()
        if len(skill_words) <= 3:  # Only for short technical terms
            found_words = []
            word_positions = {}
            
            for word in skill_words:
                if len(word) >= 2:  # Skip very short words
                    word_pattern = rf'\b{re.escape(word)}\b'
                    matches = list(re.finditer(word_pattern, resume_text, re.IGNORECASE))
                    if matches:
                        found_words.append(word)
                        word_positions[word] = [m.start() for m in matches]
            
            # Need at least 70% of words to consider it a match
            if len(found_words) >= len(skill_words) * 0.7:
                # Find the best cluster where words appear together
                best_context = ""
                best_score = 0
                
                # Check each position of each found word
                for main_word, positions in word_positions.items():
                    for pos in positions:
                        context_start = max(0, pos - 120)
                        context_end = min(len(resume_text), pos + 120)
                        context = resume_text[context_start:context_end]
                        
                        # Count how many skill words appear in this context
                        score = sum(1 for word in found_words if re.search(rf'\b{re.escape(word)}\b', context, re.IGNORECASE))
                        
                        if score > best_score:
                            best_score = score
                            best_context = context
                
                # If we found most words in proximity, consider it a match
                if best_score >= max(2, len(skill_words) * 0.7):
                    return {
                        "matched": True,
                        "evidence": best_context.strip(),
                        "context": f"Multi-word match: {found_words} ({best_score}/{len(skill_words)} words)"
                    }
        
        return {
            "matched": False,
            "evidence": "",
            "context": f"No match found for '{required_skill}' or variants"
        }

    def _are_skills_equivalent(self, skill1: str, skill2: str) -> bool:
        """Enhanced skill equivalency check with better mappings"""
        
        skill1 = skill1.lower().strip()
        skill2 = skill2.lower().strip()
        
        if skill1 == skill2:
            return True
        
        # Generate variants for both skills
        skill1_variants = self._generate_skill_variants(skill1)
        skill2_variants = self._generate_skill_variants(skill2)
        
        # Check if either skill appears in the other's variants
        if skill1 in skill2_variants or skill2 in skill1_variants:
            return True
        
        # Check if any variants overlap
        if set(skill1_variants).intersection(set(skill2_variants)):
            return True
        
        # Additional fuzzy similarity for very close matches
        similarity = SequenceMatcher(None, skill1, skill2).ratio()
        if similarity >= 0.85:
            return True
        
        # Handle common abbreviations and concatenations
        # Remove common prefixes/suffixes
        skill1_clean = re.sub(r'^(sap\s*|abap\s*)', '', skill1).strip()
        skill2_clean = re.sub(r'^(sap\s*|abap\s*)', '', skill2).strip()
        
        if skill1_clean and skill2_clean and skill1_clean == skill2_clean:
            return True
        
        return False

    def experience_extraction_agent(self, resume_text: str, required_skills: List[str], jd_requirements: Dict) -> ExperienceAnalysis:
        """Enhanced experience extraction with better date handling"""
        
        project_keywords = jd_requirements.get("project_keywords", [])
        
        prompt = f"""
You are a work experience extraction specialist. Extract ALL employment periods with precise dates.

Resume Text:
{resume_text}

Required Technical Skills: {json.dumps(required_skills)}
Project Keywords: {json.dumps(project_keywords)}

CRITICAL INSTRUCTIONS:
1. Extract EVERY work experience/employment entry
2. For each job, provide EXACT company name and job title as written
3. Extract start and end dates EXACTLY as written (do not modify format)
4. Mark as relevant ONLY if the role involved ANY of the required technical skills OR project keywords
5. Be conservative - if unclear whether relevant, mark as false

Return JSON:
{{
  "work_periods": [
    {{
      "company": "exact company name as written",
      "role": "exact job title as written", 
      "start_date": "exact start date as written",
      "end_date": "exact end date as written",
      "is_relevant": true/false
    }}
  ]
}}
"""

        try:
            response = self.azure_openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a precise employment history extractor. Extract exact information without modifications."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2000
            )

            raw_output = response.choices[0].message.content.strip()
            raw_output = re.sub(r'^```json\s*', '', raw_output)
            raw_output = re.sub(r'\s*```$', '', raw_output)
            
            result = json.loads(raw_output)
            
            # Process work periods with enhanced validation
            work_periods = []
            for period_data in result.get("work_periods", []):
                if not isinstance(period_data, dict):
                    continue
                    
                company = period_data.get("company", "").strip()
                role = period_data.get("role", "").strip()
                start_date = period_data.get("start_date", "").strip()
                end_date = period_data.get("end_date", "").strip()
                
                if not all([company, role, start_date, end_date]):
                    continue
                
                duration_months = self.calculate_experience_months(start_date, end_date)
                
                # Validate duration is reasonable
                if duration_months <= 0 or duration_months > 600:  # Cap at 50 years
                    logger.warning(f"Invalid duration for {company}: {duration_months} months")
                    duration_months = max(1, min(duration_months, 600))
                
                work_periods.append(WorkPeriod(
                    company=company,
                    role=role,
                    start_date=start_date,
                    end_date=end_date,
                    duration_months=duration_months,
                    is_relevant=bool(period_data.get("is_relevant", False))
                ))
            
            # Calculate totals with overlap removal
            total_months = self.remove_overlapping_periods(work_periods)
            total_years = round(total_months / 12.0, 2)
            
            relevant_periods = [p for p in work_periods if p.is_relevant]
            relevant_months = self.remove_overlapping_periods(relevant_periods)
            relevant_years = round(relevant_months / 12.0, 2)
            
            analysis = ExperienceAnalysis(
                work_periods=work_periods,
                relevant_periods=relevant_periods,
                total_months=total_months,
                total_years=total_years,
                relevant_months=relevant_months,
                relevant_years=relevant_years,
                meets_total_requirement=False,  # Will be set later
                meets_relevant_requirement=False  # Will be set later
            )
            
            logger.info(f"Experience Agent: {total_years} total years, {relevant_years} relevant years from {len(work_periods)} periods")
            return analysis
            
        except Exception as e:
            logger.error(f"Experience extraction failed: {str(e)}")
            # Return default analysis on failure
            return ExperienceAnalysis(
                work_periods=[],
                relevant_periods=[],
                total_months=0,
                total_years=0.0,
                relevant_months=0,
                relevant_years=0.0,
                meets_total_requirement=False,
                meets_relevant_requirement=False
            )

    def remove_overlapping_periods(self, periods: List[WorkPeriod]) -> int:
        """Enhanced overlap removal with better error handling"""
        if not periods:
            return 0
        
        # Convert to date ranges
        date_ranges = []
        for period in periods:
            try:
                start_date = self.parse_date_with_fallbacks(period.start_date)
                end_date = self.parse_date_with_fallbacks(period.end_date)
                
                # Ensure valid date range
                if start_date > end_date:
                    start_date, end_date = end_date, start_date
                
                date_ranges.append((start_date, end_date))
            except Exception as e:
                logger.warning(f"Date parsing failed for {period.company}: {e}")
                continue
        
        if not date_ranges:
            return 0
        
        # Sort by start date
        date_ranges.sort()
        
        # Merge overlapping periods
        merged = [date_ranges[0]]
        for current_start, current_end in date_ranges[1:]:
            last_start, last_end = merged[-1]
            
            if current_start <= last_end:
                # Overlapping, merge them
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # No overlap, add new period
                merged.append((current_start, current_end))
        
        # Calculate total months from merged periods
        total_months = 0
        for start_date, end_date in merged:
            months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
            total_months += max(1, months)
        
        return total_months

    def project_evaluation_agent(self, resume_text: str, required_skills: List[str], jd_requirements: Dict) -> ProjectAnalysis:
        """Enhanced project evaluation with better categorization"""
        
        project_categories = self.config.get_project_categories()
        
        prompt = f"""
You are a project evaluation specialist. Categorize projects mentioned in the resume.

Resume Text:
{resume_text}

Required Technical Skills: {json.dumps(required_skills)}
E2E Keywords: {json.dumps(project_categories.e2e_keywords)}
Support Keywords: {json.dumps(project_categories.support_keywords)}
Academic Keywords: {json.dumps(project_categories.academic_keywords)}

CRITICAL INSTRUCTIONS:
1. Find ONLY projects explicitly mentioned in the resume
2. For each project, provide the EXACT project name/description as written
3. Categorize based on project description and context:
   - E2E: Complete projects using required technical skills with E2E keywords
   - Support: Maintenance, bug fixes, support work with support keywords  
   - Academic: College, university, personal, learning projects with academic keywords
4. Return project names as simple strings in arrays
5. Be conservative - if project category unclear, place in academic

Return JSON:
{{
  "e2e_projects": ["Project name 1", "Project name 2"],
  "support_projects": ["Support project 1"],
  "academic_projects": ["Academic project 1"]
}}
"""

        try:
            response = self.azure_openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a project categorization specialist. Extract exact project names and categorize conservatively."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1500
            )

            raw_output = response.choices[0].message.content.strip()
            raw_output = re.sub(r'^```json\s*', '', raw_output)
            raw_output = re.sub(r'\s*```$', '', raw_output)
            
            result = json.loads(raw_output)
            
            # Extract and validate project arrays with better error handling
            def extract_project_names(projects_data):
                if not projects_data:
                    return []
                
                project_names = []
                for item in projects_data:
                    if isinstance(item, str) and item.strip():
                        project_names.append(item.strip())
                    elif isinstance(item, dict):
                        name = item.get('name') or item.get('project_name') or str(item)
                        if name.strip():
                            project_names.append(name.strip())
                
                return project_names[:10]  # Limit to 10 projects per category
            
            e2e_projects = extract_project_names(result.get("e2e_projects", []))
            support_projects = extract_project_names(result.get("support_projects", []))
            academic_projects = extract_project_names(result.get("academic_projects", []))
            
            # Dynamic scoring logic
            project_score, justification = self.calculate_project_score(
                len(e2e_projects), 
                len(support_projects), 
                len(academic_projects), 
                jd_requirements
            )
            
            analysis = ProjectAnalysis(
                e2e_projects=e2e_projects,
                support_projects=support_projects,
                academic_projects=academic_projects,
                project_score=project_score,
                justification=justification
            )
            
            logger.info(f"Project Agent: {project_score}/{self.config.get_scoring_weights().project_exposure_max} points - {justification}")
            return analysis
            
        except Exception as e:
            logger.error(f"Project evaluation failed: {str(e)}")
            # Return default analysis on failure
            return ProjectAnalysis(
                e2e_projects=[],
                support_projects=[],
                academic_projects=[],
                project_score=0,
                justification="Project evaluation failed"
            )

    def personal_info_extraction_agent(self, resume_text: str, candidate_name: str) -> PersonalInfo:
        """Enhanced personal information extraction with validation"""
        
        prompt = f"""
Extract personal contact information from this resume with high accuracy.

{resume_text}

RULES for mobile number:
- Extract ONLY valid phone numbers
- Remove all formatting, country codes (+91, 91), spaces, dashes
- Return exactly 10 digits if valid US/Indian number
- Return null if not exactly 10 digits after cleaning
- Do not guess, pad, or modify numbers

Return JSON:
{{
  "full_name": "complete name or null",
  "email_address": "valid email or null", 
  "mobile_number": "exactly 10 digits or null",
  "location": "city, state or address or null"
}}
"""

        try:
            response = self.azure_openai_client.chat.completions.create(
                model=self.deployment_mini if self.deployment_mini else self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a precise personal information extractor. Follow formatting rules exactly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500
            )

            raw_output = response.choices[0].message.content.strip()
            raw_output = re.sub(r'^```json\s*', '', raw_output)
            raw_output = re.sub(r'\s*```$', '', raw_output)
            
            result = json.loads(raw_output)
            
            # Validate mobile number
            mobile = result.get("mobile_number")
            if mobile:
                # Clean and validate mobile number
                mobile_clean = re.sub(r'[^0-9]', '', str(mobile))
                if len(mobile_clean) == 10 and mobile_clean[0] in '6789':  # Valid Indian mobile
                    mobile = mobile_clean
                else:
                    mobile = None
            
            # Validate email
            email = result.get("email_address")
            if email and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                email = None
            
            return PersonalInfo(
                full_name=result.get("full_name") if result.get("full_name") else None,
                email_address=email,
                mobile_number=mobile,
                location=result.get("location") if result.get("location") else None
            )
            
        except Exception as e:
            logger.warning(f"Personal info extraction failed for {candidate_name}: {e}")
            return PersonalInfo()

    # ==================== DYNAMIC UTILITY FUNCTIONS ====================
    
    def calculate_skills_score(self, matched_count: int, total_required: int, jd_requirements: Dict) -> float:
        """Dynamic skill scoring calculation based on configuration"""
        if total_required == 0:
            return 0.0
        
        # Get dynamic scoring weights
        scoring_weights = self.config.get_scoring_weights()
        
        # Calculate score based on dynamic maximum
        fraction = matched_count / total_required
        score = fraction * scoring_weights.skills_max
        
        return round(score, 2)

    def calculate_experience_scores(self, total_years: float, relevant_years: float, required_years: int, jd_requirements: Dict) -> Tuple[float, float]:
        """Dynamic experience scoring based on configuration"""
        scoring_weights = self.config.get_scoring_weights()
        
        # Total experience score
        total_score = scoring_weights.total_experience_max if total_years >= required_years else 0
        
        # Relevant experience score with dynamic threshold
        threshold = self.config.calculate_relevant_experience_threshold(required_years)
        relevant_score = scoring_weights.relevant_experience_max if relevant_years >= threshold else 0
        
        return total_score, relevant_score

    def calculate_project_score(self, e2e_count: int, support_count: int, academic_count: int, jd_requirements: Dict) -> Tuple[float, str]:
        """Dynamic project scoring based on configuration"""
        scoring_weights = self.config.get_scoring_weights()
        max_project_score = scoring_weights.project_exposure_max
        
        # Dynamic scoring logic
        if e2e_count > 0:
            score = max_project_score
            justification = f"Has {e2e_count} End-to-End project(s) using relevant technologies"
        elif support_count > 0:
            score = max_project_score * 0.5  # 50% for support projects
            justification = f"Has {support_count} support project(s) only"
        else:
            score = 0
            justification = "Only academic/unrelated projects found" if academic_count > 0 else "No projects found"
        
        return round(score, 2), justification

    def candidate_summary_agent(
        self, 
        candidate_name: str,
        skills_analysis: SkillAnalysis, 
        experience_analysis: ExperienceAnalysis, 
        project_analysis: ProjectAnalysis,
        total_score: float,
        jd_requirements: Dict
    ) -> CandidateSummary:
        """Enhanced candidate summary with verification stats"""
        
        # Extract specific details for detailed summary
        matched_skills_list = [skill.skill_name for skill in skills_analysis.matched_skills if skill.matched]
        missing_skills_list = [skill.skill_name for skill in skills_analysis.missing_skills]
        
        # Get verification statistics
        verification_stats = skills_analysis.verification_stats
        
        # Get specific company names with dates
        all_companies = [f"{p.company} ({p.start_date} - {p.end_date})" for p in experience_analysis.work_periods]
        relevant_companies = [f"{p.company} ({p.start_date} - {p.end_date})" for p in experience_analysis.relevant_periods]
        
        # Get specific project names
        e2e_project_names = project_analysis.e2e_projects
        support_project_names = project_analysis.support_projects
        academic_project_names = project_analysis.academic_projects
        
        # Get dynamic scoring breakdown
        scoring_weights = self.config.get_scoring_weights()
        
        prompt = f"""
Create a comprehensive candidate summary with SPECIFIC DETAILS and verification results.

Candidate: {candidate_name}
Total Score: {total_score}/{scoring_weights.get_total_max()}

VERIFICATION RESULTS:
- Verified Skills: {verification_stats.get('verified', 0)}
- Failed Verification: {verification_stats.get('failed', 0)}
- Unverified: {verification_stats.get('unverified', 0)}

SCORING BREAKDOWN:
- Skills: {self.calculate_skills_score(skills_analysis.matched_count, skills_analysis.total_required_skills, jd_requirements)}/{scoring_weights.skills_max} (Verified: {verification_stats.get('verified', 0)})
- Total Experience: {10 if experience_analysis.meets_total_requirement else 0}/{scoring_weights.total_experience_max}
- Relevant Experience: {20 if experience_analysis.meets_relevant_requirement else 0}/{scoring_weights.relevant_experience_max}
- Projects: {project_analysis.project_score}/{scoring_weights.project_exposure_max}

VERIFIED MATCHED SKILLS ({len(matched_skills_list)}): {', '.join(matched_skills_list)}
MISSING SKILLS ({len(missing_skills_list)}): {', '.join(missing_skills_list)}

EXPERIENCE: {experience_analysis.total_years} years total, {experience_analysis.relevant_years} years relevant
COMPANIES: {'; '.join(all_companies[:3])}
PROJECTS: E2E({len(e2e_project_names)}), Support({len(support_project_names)}), Academic({len(academic_project_names)})

Create a summary that mentions VERIFIED skills and evidence-based evaluation results.

Return JSON:
{{
  "overall_rating": "Good match" or "Partial match" or "Poor match",
  "summary_text": "Detailed summary emphasizing VERIFIED skills and evidence-based matching results",
  "key_strengths": ["Include specific verified skills and strengths"],
  "key_gaps": ["Include specific missing skills and gaps"],
  "recommendation": "Evidence-based recommendation"
}}
"""

        try:
            response = self.azure_openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "Create comprehensive, evidence-based candidate summaries emphasizing verification results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )

            raw_output = response.choices[0].message.content.strip()
            raw_output = re.sub(r'^```json\s*', '', raw_output)
            raw_output = re.sub(r'\s*```$', '', raw_output)
            
            result = json.loads(raw_output)
            
            return CandidateSummary(
                candidate_name=candidate_name,
                overall_rating=result.get("overall_rating", "Unknown"),
                summary_text=result.get("summary_text", "No summary available"),
                key_strengths=result.get("key_strengths", []),
                key_gaps=result.get("key_gaps", []),
                recommendation=result.get("recommendation", "No recommendation available")
            )
            
        except Exception as e:
            logger.error(f"Summary generation failed for {candidate_name}: {e}")
            # Return evidence-based summary on failure
            return CandidateSummary(
                candidate_name=candidate_name,
                overall_rating="Review Required",
                summary_text=f"{candidate_name} scored {total_score:.1f}/{scoring_weights.get_total_max()}. Verified {verification_stats.get('verified', 0)}/{len(matched_skills_list)} claimed skills. Evidence-based evaluation shows {experience_analysis.total_years} years experience.",
                key_strengths=[f"Verified: {skill}" for skill in matched_skills_list[:3]],
                key_gaps=[f"Missing: {skill}" for skill in missing_skills_list[:3]],
                recommendation="Manual review recommended due to summary generation error"
            )

    # ==================== VALIDATION AND ORCHESTRATION ====================
    
    def validate_evaluation_consistency(self, evaluation: CandidateEvaluation, jd_requirements: Dict) -> str:
        """Enhanced validation with verification stats"""
        issues = []
        
        # Check skills calculation with dynamic scoring
        expected_skills_score = self.calculate_skills_score(
            evaluation.skills_analysis.matched_count,
            evaluation.skills_analysis.total_required_skills,
            jd_requirements
        )
        
        # Check verification stats
        verification_stats = evaluation.skills_analysis.verification_stats
        total_verified = verification_stats.get("verified", 0)
        total_failed = verification_stats.get("failed", 0)
        
        if total_failed > total_verified:
            issues.append("More skills failed verification than passed")
        
        # Check confidence levels based on verification
        if total_verified == 0 and evaluation.skills_analysis.matched_count > 0:
            issues.append("No skills could be verified despite claims")
        
        # Determine confidence level based on verification results
        if total_verified >= evaluation.skills_analysis.matched_count * 0.8 and not issues:
            confidence = "HIGH"
        elif total_verified >= evaluation.skills_analysis.matched_count * 0.6 and len(issues) <= 1:
            confidence = "MEDIUM" 
        else:
            confidence = "LOW"
        
        if issues:
            logger.warning(f"Validation issues for {evaluation.candidate_name}: {issues}")
        
        return confidence

    async def evaluate_candidate_with_agents(
        self,
        candidate_name: str,
        resume_text: str,
        jd_requirements: Dict,
        vector_manager
    ) -> CandidateEvaluation:
        """Enhanced orchestration with evidence verification"""
        
        try:
            logger.info(f"Evaluating {candidate_name} with ANTI-HALLUCINATION agents")
            
            if not resume_text or len(resume_text.strip()) < 100:
                raise ValueError(f"Resume text too short for {candidate_name}")
            
            required_skills = jd_requirements.get("technical_skills", [])
            required_experience = jd_requirements.get("min_experience_years", 0)
            
            if not required_skills:
                raise ValueError("No technical skills found in JD requirements")
            
            # Step 1: Check cache first
            jd_hash = hashlib.md5(json.dumps(jd_requirements).encode()).hexdigest()
            cached_evaluation = vector_manager.get_cached_evaluation(candidate_name, jd_hash)
            
            if cached_evaluation:
                logger.info(f"Using cached evaluation for {candidate_name}")
                return CandidateEvaluation(**cached_evaluation)
            
            # Step 2: Run AI agents with enhanced error handling
            skill_analysis_task = asyncio.create_task(
                asyncio.to_thread(self.skill_matching_agent, resume_text, required_skills, jd_requirements)
            )
            
            experience_analysis_task = asyncio.create_task(
                asyncio.to_thread(self.experience_extraction_agent, resume_text, required_skills, jd_requirements)
            )
            
            project_analysis_task = asyncio.create_task(
                asyncio.to_thread(self.project_evaluation_agent, resume_text, required_skills, jd_requirements)
            )
            
            personal_info_task = asyncio.create_task(
                asyncio.to_thread(self.personal_info_extraction_agent, resume_text, candidate_name)
            )
            
            # Wait for all agents to complete
            skill_analysis, experience_analysis, project_analysis, personal_info = await asyncio.gather(
                skill_analysis_task,
                experience_analysis_task,
                project_analysis_task,
                personal_info_task
            )
            
            # Step 3: Apply dynamic calculations
            skills_score = self.calculate_skills_score(
                skill_analysis.matched_count,
                skill_analysis.total_required_skills,
                jd_requirements
            )
            
            total_exp_score, relevant_exp_score = self.calculate_experience_scores(
                experience_analysis.total_years,
                experience_analysis.relevant_years,
                required_experience,
                jd_requirements
            )
            
            # Update experience analysis with requirement checks
            experience_analysis.meets_total_requirement = (experience_analysis.total_years >= required_experience)
            experience_analysis.meets_relevant_requirement = (relevant_exp_score > 0)
            
            # Calculate total score dynamically
            total_score = skills_score + total_exp_score + relevant_exp_score + project_analysis.project_score
            max_possible = self.config.get_scoring_weights().get_total_max()
            total_score = min(max_possible, max(0.0, total_score))
            
            # Step 4: Generate candidate summary
            summary_task = asyncio.create_task(
                asyncio.to_thread(
                    self.candidate_summary_agent, 
                    candidate_name, 
                    skill_analysis, 
                    experience_analysis, 
                    project_analysis,
                    total_score,
                    jd_requirements
                )
            )
            
            candidate_summary = await summary_task
            
            # Step 5: Create comprehensive evaluation
            evaluation = CandidateEvaluation(
                candidate_name=candidate_name,
                candidate_summary=candidate_summary,
                skills_analysis=skill_analysis,
                experience_analysis=experience_analysis,
                project_analysis=project_analysis,
                personal_info=personal_info,
                total_score=total_score,
                confidence_level="PENDING",
                evaluation_timestamp=time.time()
            )
            
            # Step 6: Enhanced validation with verification stats
            confidence_level = self.validate_evaluation_consistency(evaluation, jd_requirements)
            evaluation.confidence_level = confidence_level
            
            # Step 7: Store in cache
            evaluation_dict = evaluation.dict()
            vector_manager.store_candidate_evaluation(
                candidate_name, 
                jd_hash, 
                evaluation_dict, 
                resume_text[:1000]
            )
            
            verification_stats = skill_analysis.verification_stats
            logger.info(f"Completed evaluation for {candidate_name}: {total_score:.1f}% ({confidence_level} confidence, Verified: {verification_stats.get('verified', 0)}/{skill_analysis.matched_count})")
            return evaluation
            
        except Exception as e:
            logger.error(f"Agent evaluation failed for {candidate_name}: {str(e)}")
            # Return minimal evaluation on failure
            required_skills = jd_requirements.get("technical_skills", [])
            return CandidateEvaluation(
                candidate_name=candidate_name,
                candidate_summary=CandidateSummary(
                    candidate_name=candidate_name,
                    overall_rating="Error",
                    summary_text=f"Evaluation failed for {candidate_name}. Manual review required.",
                    key_strengths=[],
                    key_gaps=[f"Unable to evaluate against {len(required_skills)} required skills"],
                    recommendation="Manual review required due to evaluation error"
                ),
                skills_analysis=SkillAnalysis(
                    matched_skills=[],
                    missing_skills=[SkillMatch(
                        skill_name=skill,
                        matched=False,
                        evidence_snippet="",
                        confidence_score=0.0,
                        context="Evaluation failed",
                        verification_status="failed"
                    ) for skill in required_skills],
                    total_required_skills=len(required_skills),
                    matched_count=0,
                    missing_count=len(required_skills),
                    confidence_score=0.0,
                    verification_stats={"verified": 0, "unverified": 0, "failed": len(required_skills)}
                ),
                experience_analysis=ExperienceAnalysis(
                    work_periods=[],
                    relevant_periods=[],
                    total_months=0,
                    total_years=0.0,
                    relevant_months=0,
                    relevant_years=0.0,
                    meets_total_requirement=False,
                    meets_relevant_requirement=False
                ),
                project_analysis=ProjectAnalysis(
                    e2e_projects=[],
                    support_projects=[],
                    academic_projects=[],
                    project_score=0,
                    justification="Evaluation failed"
                ),
                personal_info=PersonalInfo(),
                total_score=0.0,
                confidence_level="ERROR",
                evaluation_timestamp=time.time()
            )

    # ==================== MAIN SCREENING ORCHESTRATION ====================
    
    async def process_screening_job(
        self,
        job_description: str,
        documents_with_metadata: List[Document],
        vector_manager,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Enhanced screening orchestration with anti-hallucination measures"""
        
        try:
            logger.info("Starting ANTI-HALLUCINATION screening with enhanced AI agents")
            start_time = time.time()
            
            # Step 1: Enhanced JD requirements parsing
            logger.info("Step 1: Parsing JD requirements with validation")
            
            jd_requirements = vector_manager.get_cached_jd_requirements(job_description)
            
            if jd_requirements:
                logger.info("Using cached JD requirements")
            else:
                logger.info("Parsing JD requirements with enhanced validation")
                jd_requirements = self.extract_jd_requirements_agent(job_description)
                
                # Store in search index
                jd_hash = vector_manager.store_jd_requirements(job_description, jd_requirements)
            
            technical_skills = jd_requirements.get("technical_skills", [])
            experience_req = jd_requirements.get("min_experience_years", 0)
            
            if not technical_skills:
                raise ValueError("No technical skills extracted from job description")
            
            logger.info(f"JD Analysis: {len(technical_skills)} validated skills, {experience_req} years experience")
            
            # Step 2: Enhanced candidate data extraction
            candidate_data = {}
            for doc in documents_with_metadata:
                candidate_name = doc.metadata.get('candidate_name')
                if candidate_name:
                    if candidate_name not in candidate_data:
                        candidate_data[candidate_name] = []
                    candidate_data[candidate_name].append(doc.page_content)
            
            if not candidate_data:
                raise ValueError("No candidate data found in documents")
            
            logger.info(f"Found {len(candidate_data)} unique candidates")
            
            # Step 3: Process candidates with enhanced validation
            logger.info("Step 3: Evaluating candidates with ANTI-HALLUCINATION agents")
            
            candidate_tasks = []
            for candidate_name, document_chunks in candidate_data.items():
                # Combine all chunks for this candidate
                full_resume_text = "\n\n".join(document_chunks)
                
                # Validate resume text length
                if len(full_resume_text.strip()) < 100:
                    logger.warning(f"Resume too short for {candidate_name}: {len(full_resume_text)} characters")
                    continue
                
                # Create evaluation task
                task = self.evaluate_candidate_with_agents(
                    candidate_name,
                    full_resume_text,
                    jd_requirements,
                    vector_manager
                )
                candidate_tasks.append(task)
            
            # Execute all evaluations concurrently
            logger.info(f"Processing {len(candidate_tasks)} candidates with evidence verification")
            evaluations = await asyncio.gather(*candidate_tasks, return_exceptions=True)
            
            # Step 4: Process results and compile verification statistics
            successful_evaluations = []
            failed_count = 0
            total_verification_stats = {"verified": 0, "unverified": 0, "failed": 0}
            
            for evaluation in evaluations:
                if isinstance(evaluation, Exception):
                    logger.error(f"Evaluation task failed: {str(evaluation)}")
                    failed_count += 1
                elif isinstance(evaluation, CandidateEvaluation):
                    successful_evaluations.append(evaluation)
                    
                    # Aggregate verification stats
                    eval_stats = evaluation.skills_analysis.verification_stats
                    for key in total_verification_stats:
                        total_verification_stats[key] += eval_stats.get(key, 0)
                    
                    logger.info(f"Success: {evaluation.candidate_name} ({evaluation.total_score:.1f}%, {evaluation.confidence_level}, Verified: {eval_stats.get('verified', 0)})")
                else:
                    failed_count += 1
            
            # Step 5: Sort and format results
            ranked_evaluations = sorted(successful_evaluations, key=lambda x: x.total_score, reverse=True)
            candidates_above_threshold = [e for e in successful_evaluations if e.total_score >= threshold]
            
            # Get dynamic scoring weights
            scoring_weights = self.config.get_scoring_weights()
            
            # Convert to API format with verification data
            formatted_candidates = []
            for evaluation in ranked_evaluations:
                # Create enhanced structured individual scores
                structured_scores = {
                    "mandatory_skills": {
                        "score": f"{self.calculate_skills_score(evaluation.skills_analysis.matched_count, evaluation.skills_analysis.total_required_skills, jd_requirements)}/{scoring_weights.skills_max}",
                        "required_skills": [skill.skill_name for skill in evaluation.skills_analysis.matched_skills + evaluation.skills_analysis.missing_skills],
                        "matched_skills": [skill.skill_name for skill in evaluation.skills_analysis.matched_skills if skill.matched],
                        "missing_skills": [skill.skill_name for skill in evaluation.skills_analysis.missing_skills],
                        "calculation": f"{evaluation.skills_analysis.matched_count}/{evaluation.skills_analysis.total_required_skills} × {scoring_weights.skills_max} = {self.calculate_skills_score(evaluation.skills_analysis.matched_count, evaluation.skills_analysis.total_required_skills, jd_requirements)}",
                        "verification_stats": evaluation.skills_analysis.verification_stats
                    },
                    "total_experience": {
                        "score": f"{scoring_weights.total_experience_max if evaluation.experience_analysis.meets_total_requirement else 0}/{scoring_weights.total_experience_max}",
                        "employment_periods": [f"{p.company} ({p.start_date} - {p.end_date})" for p in evaluation.experience_analysis.work_periods],
                        "total_experience": f"{evaluation.experience_analysis.total_years} years ({evaluation.experience_analysis.total_months} months)",
                        "required": f"{jd_requirements.get('min_experience_years', 0)} years",
                        "meets_requirement": evaluation.experience_analysis.meets_total_requirement
                    },
                    "relevant_experience": {
                        "score": f"{scoring_weights.relevant_experience_max if evaluation.experience_analysis.meets_relevant_requirement else 0}/{scoring_weights.relevant_experience_max}",
                        "relevant_companies": [f"{p.company} ({p.start_date} - {p.end_date})" for p in evaluation.experience_analysis.relevant_periods],
                        "total_relevant_experience": f"{evaluation.experience_analysis.relevant_years} years ({evaluation.experience_analysis.relevant_months} months)",
                        "threshold": f"{self.config.calculate_relevant_experience_threshold(jd_requirements.get('min_experience_years', 0))} years",
                        "meets_requirement": evaluation.experience_analysis.meets_relevant_requirement
                    },
                    "project_exposure": {
                        "score": f"{evaluation.project_analysis.project_score}/{scoring_weights.project_exposure_max}",
                        "e2e_projects": evaluation.project_analysis.e2e_projects,
                        "support_projects": evaluation.project_analysis.support_projects,
                        "academic_unrelated": evaluation.project_analysis.academic_projects,
                        "scoring_logic": evaluation.project_analysis.justification
                    }
                }
                
                formatted_candidates.append({
                    "name": evaluation.candidate_name,
                    "score": evaluation.total_score,
                    "reason": evaluation.candidate_summary.summary_text,
                    "overall_rating": evaluation.candidate_summary.overall_rating,
                    "key_strengths": evaluation.candidate_summary.key_strengths,
                    "key_gaps": evaluation.candidate_summary.key_gaps,
                    "recommendation": evaluation.candidate_summary.recommendation,
                    "structured_scores": structured_scores,
                    "personal_info": evaluation.personal_info.dict(),
                    "confidence_level": evaluation.confidence_level,
                    "evaluation_timestamp": evaluation.evaluation_timestamp
                })
            
            # Step 6: Generate enhanced summary with verification stats
            processing_time = time.time() - start_time
            
            if successful_evaluations:
                scores = [e.total_score for e in successful_evaluations]
                logger.info(f"Score range: {min(scores):.1f}% - {max(scores):.1f}%")
                logger.info(f"Top candidate: {ranked_evaluations[0].candidate_name} ({ranked_evaluations[0].total_score:.1f}%)")
            
            logger.info(f"ANTI-HALLUCINATION Screening complete - Verified: {total_verification_stats['verified']}, Failed: {total_verification_stats['failed']}")
            
            success_message = (
                f"ANTI-HALLUCINATION evidence-verified evaluation of {len(candidate_data)} candidates. "
                f"Evaluated against {len(technical_skills)} technical skills with evidence verification. "
                f"Skills verified: {total_verification_stats['verified']}, Failed verification: {total_verification_stats['failed']}. "
                f"{len(candidates_above_threshold)} candidates scored above {threshold}%."
            )
            
            return {
                "success": True,
                "message": success_message,
                "candidates": formatted_candidates,
                "total_candidates": len(candidate_data),
                "candidates_above_threshold": len(candidates_above_threshold),
                "processing_time": processing_time,
                "jd_requirements": jd_requirements,
                "accuracy_stats": {
                    "high_confidence": len([e for e in successful_evaluations if e.confidence_level == "HIGH"]),
                    "medium_confidence": len([e for e in successful_evaluations if e.confidence_level == "MEDIUM"]),
                    "low_confidence": len([e for e in successful_evaluations if e.confidence_level == "LOW"]),
                    "evaluation_method": "anti_hallucination_evidence_verified_ai_agents",
                    "verification_stats": total_verification_stats
                }
            }
            
        except Exception as e:
            logger.error(f"ANTI-HALLUCINATION screening failed: {str(e)}")
            return {
                "success": False,
                "message": f"ANTI-HALLUCINATION screening failed: {str(e)}",
                "candidates": [],
                "total_candidates": 0,
                "candidates_above_threshold": 0,
                "error": str(e)
            }