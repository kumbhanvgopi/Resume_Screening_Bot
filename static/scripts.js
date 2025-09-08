// scripts.js - Enhanced for FULLY DYNAMIC Backend Data

// Global variables for dynamic configuration
let currentPage = 1;
const itemsPerPage = 3;
let resumeFiles = [];
let jdFile = null;
let currentMethod = 'text';
let candidatesData = [];
let dynamicConfig = null; // Store dynamic configuration
let currentJdRequirements = null; // Store current JD requirements

// DOM elements
const greetingElement = document.getElementById('greeting');
const jobDescriptionTextarea = document.getElementById('jobDescription');
const resumeUploadArea = document.getElementById('resume-upload-area');
const resumeFileInput = document.getElementById('resume-file-input');
const resumeFileList = document.getElementById('resume-file-list');
const jdUploadArea = document.getElementById('jd-upload-area');
const jdFileInput = document.getElementById('jd-file-input');
const jdFileList = document.getElementById('jd-file-list');
const processButton = document.getElementById('process-button');
const loadingSpinner = document.querySelector('.loading-spinner');
const messageArea = document.getElementById('message-area');
const resultsSection = document.getElementById('results-section');
const candidatesContainer = document.getElementById('candidates-container');

// Dynamic utility functions
function safeDynamicGet(obj, path, defaultValue = 'N/A') {
    try {
        if (!obj || typeof obj !== 'object') return defaultValue;
        
        const keys = path.split('.');
        let result = obj;
        
        for (const key of keys) {
            if (result && typeof result === 'object' && key in result) {
                result = result[key];
            } else {
                return defaultValue;
            }
        }
        
        return result !== null && result !== undefined ? result : defaultValue;
    } catch (error) {
        console.warn(`Error accessing dynamic path ${path}:`, error);
        return defaultValue;
    }
}

function safeDynamicArray(arr, defaultValue = []) {
    try {
        return Array.isArray(arr) ? arr : defaultValue;
    } catch (error) {
        console.warn('Error processing dynamic array:', error);
        return defaultValue;
    }
}

function generateDynamicMessage(type, baseMessage, dynamicData = {}) {
    try {
        if (type === 'error' && dynamicData.errorCode) {
            return `${baseMessage} (Error Code: ${dynamicData.errorCode})`;
        } else if (type === 'success' && dynamicData.candidateCount) {
            return `${baseMessage} - Processed ${dynamicData.candidateCount} candidates successfully`;
        }
        return baseMessage;
    } catch (error) {
        console.warn('Error generating dynamic message:', error);
        return baseMessage;
    }
}

// Initialize with dynamic configuration loading
document.addEventListener('DOMContentLoaded', function() {
    try {
        console.log('DOM loaded, initializing FULLY DYNAMIC UI...');
        loadDynamicConfiguration()
            .then(() => {
                setupEventListeners();
                updateProcessButton();
                testBackendConnection();
                setupMethodToggle();
                console.log('FULLY DYNAMIC UI initialization complete');
            })
            .catch(error => {
                console.error('Error during dynamic initialization:', error);
                showMessage('error', 'Failed to load dynamic configuration. Using defaults.');
                // Continue with default setup
                setupEventListeners();
                updateProcessButton();
                testBackendConnection();
                setupMethodToggle();
            });
    } catch (error) {
        console.error('Critical error during initialization:', error);
        showMessage('error', 'Failed to initialize application. Please refresh the page.');
    }
});

async function loadDynamicConfiguration() {
    try {
        const response = await fetch('/api/config');
        if (response.ok) {
            const configData = await response.json();
            dynamicConfig = configData.config;
            console.log('Dynamic configuration loaded:', dynamicConfig);
            
            // Update UI elements based on dynamic config
            updateUIWithDynamicConfig();
        } else {
            console.warn('Failed to load dynamic configuration, using defaults');
        }
    } catch (error) {
        console.error('Error loading dynamic configuration:', error);
    }
}

function updateUIWithDynamicConfig() {
    try {
        if (!dynamicConfig) return;
        
        // Update display labels dynamically
        const displayLabels = dynamicConfig.display_labels || {};
        
        // Update section headers if they exist
        const skillsHeader = document.querySelector('h3[data-section="skills"]');
        if (skillsHeader && displayLabels.skills_section) {
            skillsHeader.textContent = displayLabels.skills_section;
        }
        
        // Update scoring weight indicators if they exist
        const scoringWeights = dynamicConfig.scoring_weights || {};
        console.log('Dynamic scoring weights applied:', scoringWeights);
        
    } catch (error) {
        console.error('Error updating UI with dynamic config:', error);
    }
}

function setupMethodToggle() {
    try {
        const methodBtns = document.querySelectorAll('.method-btn');
        const textMethod = document.getElementById('text-method');
        const fileMethod = document.getElementById('file-method');
        
        if (methodBtns && textMethod && fileMethod) {
            methodBtns.forEach(btn => {
                btn.addEventListener('click', function () {
                    methodBtns.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    if (btn.dataset.method === 'text') {
                        textMethod.style.display = '';
                        fileMethod.style.display = 'none';
                    } else {
                        textMethod.style.display = 'none';
                        fileMethod.style.display = '';
                    }
                });
            });
        }
    } catch (error) {
        console.error('Error setting up method toggle:', error);
    }
}

function testBackendConnection() {
    try {
        fetch('/api/health')
            .then(response => {
                if (response.ok) {
                    return response.json();
                }
                throw new Error(`HTTP ${response.status}`);
            })
            .then(data => {
                console.log('Backend connection successful:', data);
                if (data.architecture === 'enhanced_accuracy_focused_ai_agents_dynamic') {
                    console.log('FULLY DYNAMIC enhanced accuracy-focused AI agents system detected');
                    console.log('Dynamic features active:', data.dynamic_configuration);
                }
            })
            .catch(error => {
                console.error('Backend connection failed:', error);
                showMessage('error', generateDynamicMessage('error', 'Backend connection failed. Please check server status.', {errorCode: 'CONN_FAIL'}));
            });
    } catch (error) {
        console.error('Error testing backend connection:', error);
    }
}

function setupEventListeners() {
    try {
        // Method selector
        document.querySelectorAll('.method-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const method = this.dataset.method;
                switchMethod(method);
            });
        });

        // File upload areas
        if (resumeUploadArea && resumeFileInput) {
            setupFileUpload(resumeUploadArea, resumeFileInput, handleResumeFiles);
        }
        if (jdUploadArea && jdFileInput) {
            setupFileUpload(jdUploadArea, jdFileInput, handleJDFile);
        }

        // Process button
        if (processButton) {
            processButton.addEventListener('click', processCandidates);
        }

        // Job description textarea
        if (jobDescriptionTextarea) {
            jobDescriptionTextarea.addEventListener('input', updateProcessButton);
        }

        console.log('Dynamic event listeners setup complete');
    } catch (error) {
        console.error('Error setting up event listeners:', error);
    }
}

function switchMethod(method) {
    try {
        currentMethod = method;

        document.querySelectorAll('.method-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        const methodBtn = document.querySelector(`[data-method="${method}"]`);
        if (methodBtn) {
            methodBtn.classList.add('active');
        }

        const textMethod = document.getElementById('text-method');
        const fileMethod = document.getElementById('file-method');
        
        if (textMethod && fileMethod) {
            textMethod.style.display = method === 'text' ? 'block' : 'none';
            fileMethod.style.display = method === 'file' ? 'block' : 'none';
        }

        updateProcessButton();
    } catch (error) {
        console.error('Error switching method:', error);
    }
}

function setupFileUpload(uploadArea, fileInput, handleFunction) {
    try {
        uploadArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFunction);

        // Drag and drop
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('drag-over');
            const files = Array.from(e.dataTransfer.files);
            handleFunction({ target: { files } });
        });
    } catch (error) {
        console.error('Error setting up file upload:', error);
    }
}

function handleResumeFiles(event) {
    try {
        const files = Array.from(event.target.files);

        files.forEach(file => {
            if (isValidFile(file)) {
                resumeFiles.push(file);
            }
        });

        currentPage = 1;
        updateResumeFileList();
        updateProcessButton();
    } catch (error) {
        console.error('Error handling resume files:', error);
        showMessage('error', 'Error processing resume files');
    }
}

function handleJDFile(event) {
    try {
        const file = event.target.files[0];
        if (file && isValidFile(file)) {
            jdFile = file;
            updateJDFileList();
            updateProcessButton();
        }
    } catch (error) {
        console.error('Error handling JD file:', error);
        showMessage('error', 'Error processing job description file');
    }
}

function isValidFile(file) {
    const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!allowedTypes.includes(file.type)) {
        showMessage('error', `File "${file.name}" is not a supported format. Please use PDF or DOCX files.`);
        return false;
    }

    if (file.size > maxSize) {
        showMessage('error', `File "${file.name}" is too large. Maximum size is 10MB.`);
        return false;
    }

    return true;
}

function changePage(newPage) {
    const totalPages = Math.ceil(resumeFiles.length / itemsPerPage);
    if (newPage >= 1 && newPage <= totalPages) {
        currentPage = newPage;
        updateResumeFileList();
    }
}

function updateResumeFileList() {
    try {
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        const currentFiles = resumeFiles.slice(startIndex, endIndex);
        const totalPages = Math.ceil(resumeFiles.length / itemsPerPage);

        if (!resumeFileList) return;

        resumeFileList.innerHTML = '';

        currentFiles.forEach((file, index) => {
            const actualIndex = startIndex + index;
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <div class="file-info">
                    <i class="fas fa-file-pdf file-icon"></i>
                    <div>
                        <div class="file-name">${escapeHtml(file.name)}</div>
                        <div class="file-size">${formatFileSize(file.size)}</div>
                    </div>
                </div>
                <button class="remove-file" onclick="removeResumeFile(${actualIndex})">
                    <i class="fas fa-times"></i> Remove
                </button>
            `;
            resumeFileList.appendChild(fileItem);
        });

        if (resumeFiles.length > itemsPerPage) {
            const paginationContainer = document.createElement('div');
            paginationContainer.className = 'pagination-container';
            paginationContainer.innerHTML = `
                <div class="pagination-info">
                    Showing ${startIndex + 1}-${Math.min(endIndex, resumeFiles.length)} of ${resumeFiles.length} files
                </div>
                <div class="pagination-controls">
                    <button class="pagination-btn" onclick="changePage(${currentPage - 1})" ${currentPage === 1 ? 'disabled' : ''}>
                        <i class="fas fa-chevron-left"></i> Previous
                    </button>
                    <span style="padding: 6px 10px; font-size: 0.8rem; color: #e2e8f0;">
                        Page ${currentPage} of ${totalPages}
                    </span>
                    <button class="pagination-btn" onclick="changePage(${currentPage + 1})" ${currentPage === totalPages ? 'disabled' : ''}>
                        Next <i class="fas fa-chevron-right"></i>
                    </button>
                </div>
            `;
            resumeFileList.appendChild(paginationContainer);
        }
    } catch (error) {
        console.error('Error updating resume file list:', error);
    }
}

function updateJDFileList() {
    try {
        if (!jdFileList) return;

        jdFileList.innerHTML = '';

        if (jdFile) {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <div class="file-info">
                    <i class="fas fa-file-pdf file-icon"></i>
                    <div>
                        <div class="file-name">${escapeHtml(jdFile.name)}</div>
                        <div class="file-size">${formatFileSize(jdFile.size)}</div>
                    </div>
                </div>
                <button class="remove-file" onclick="removeJDFile()">
                    <i class="fas fa-times"></i> Remove
                </button>
            `;
            jdFileList.appendChild(fileItem);
        }
    } catch (error) {
        console.error('Error updating JD file list:', error);
    }
}

function removeResumeFile(index) {
    try {
        resumeFiles.splice(index, 1);

        const totalPages = Math.ceil(resumeFiles.length / itemsPerPage);
        if (currentPage > totalPages && totalPages > 0) {
            currentPage = totalPages;
        } else if (resumeFiles.length === 0) {
            currentPage = 1;
        }

        updateResumeFileList();
        updateProcessButton();
    } catch (error) {
        console.error('Error removing resume file:', error);
    }
}

function removeJDFile() {
    try {
        jdFile = null;
        updateJDFileList();
        updateProcessButton();
    } catch (error) {
        console.error('Error removing JD file:', error);
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function updateProcessButton() {
    try {
        if (!processButton || !jobDescriptionTextarea) return;

        const hasJobDescription = currentMethod === 'text' ?
            jobDescriptionTextarea.value.trim() !== '' :
            jdFile !== null;
        const hasResumeFiles = resumeFiles.length > 0;

        processButton.disabled = !hasJobDescription || !hasResumeFiles;
    } catch (error) {
        console.error('Error updating process button:', error);
    }
}

async function processCandidates() {
    try {
        setProcessingState(true);
        clearMessages();

        const formData = new FormData();

        resumeFiles.forEach(file => {
            formData.append('resume_files', file);
        });

        let endpoint;
        if (currentMethod === 'text') {
            formData.append('job_description', jobDescriptionTextarea.value);
            endpoint = '/api/match-candidates';
        } else {
            formData.append('job_description_file', jdFile);
            endpoint = '/api/match-candidates-with-jd-file';
        }

        console.log('Sending request to:', endpoint);

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Received DYNAMIC enhanced response:', data);

        if (data.success) {
            // Store dynamic configuration and JD requirements
            if (data.dynamic_config) {
                dynamicConfig = { ...dynamicConfig, ...data.dynamic_config };
            }
            if (data.jd_requirements) {
                currentJdRequirements = data.jd_requirements;
            }
            
            showMessage('success', generateDynamicMessage('success', data.message, {candidateCount: data.total_candidates}));
            candidatesData = data.candidates || [];
            displayResults(data);
        } else {
            showMessage('error', data.message || 'Unknown error occurred');
        }

    } catch (error) {
        console.error('Processing error:', error);
        showMessage('error', `Failed to process candidates: ${error.message}`);
    } finally {
        setProcessingState(false);
    }
}

function setProcessingState(isProcessing) {
    try {
        if (!processButton || !loadingSpinner) return;

        processButton.disabled = isProcessing;
        loadingSpinner.style.display = isProcessing ? 'block' : 'none';

        const buttonText = processButton.querySelector('span');
        if (buttonText) {
            if (isProcessing) {
                buttonText.textContent = 'Processing with DYNAMIC Enhanced AI Agents...';
            } else {
                buttonText.textContent = 'Process with Enhanced AI Agents (Fully Dynamic)';
                updateProcessButton();
            }
        }
    } catch (error) {
        console.error('Error setting processing state:', error);
    }
}

function showMessage(type, message) {
    try {
        if (!messageArea) return;

        const messageElement = document.createElement('div');
        messageElement.className = type === 'error' ? 'error-message' : 'success-message';
        messageElement.innerHTML = `<i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'check-circle'}"></i> ${escapeHtml(message)}`;

        messageArea.appendChild(messageElement);

        setTimeout(() => {
            if (messageElement.parentNode) {
                messageElement.remove();
            }
        }, 5000);
    } catch (error) {
        console.error('Error showing message:', error);
    }
}

function clearMessages() {
    try {
        if (messageArea) {
            messageArea.innerHTML = '';
        }
    } catch (error) {
        console.error('Error clearing messages:', error);
    }
}

function toggleDetails(index) {
    try {
        const detailsElement = document.getElementById(`details-${index}`);
        const button = detailsElement?.previousElementSibling;
        
        if (!detailsElement || !button) return;

        if (detailsElement.style.display === 'none' || detailsElement.style.display === '') {
            detailsElement.style.display = 'block';
            button.innerHTML = '<i class="fas fa-chevron-up"></i> Hide Details';
        } else {
            detailsElement.style.display = 'none';
            button.innerHTML = '<i class="fas fa-chevron-down"></i> Show Details';
        }
    } catch (error) {
        console.error('Error toggling details:', error);
    }
}

function displayResults(data) {
    try {
        if (!data) {
            showMessage('error', 'No data received from server');
            return;
        }

        // Update statistics with dynamic data
        const totalCandidatesEl = document.getElementById('total-candidates');
        const averageScoreEl = document.getElementById('average-score');

        if (totalCandidatesEl) {
            totalCandidatesEl.textContent = data.total_candidates || 0;
        }

        const candidates = data.candidates || [];
        const averageScore = candidates.length > 0 ?
            (candidates.reduce((sum, candidate) => sum + (candidate.score || 0), 0) / candidates.length).toFixed(1) : 0;
        
        if (averageScoreEl) {
            averageScoreEl.textContent = averageScore;
        }

        if (!candidatesContainer) return;

        candidatesContainer.innerHTML = '';

        if (candidates.length === 0) {
            candidatesContainer.innerHTML = `
                <div style="text-align: center; padding: 20px; color: #a0aec0;">
                    <i class="fas fa-search" style="font-size: 2rem; margin-bottom: 10px;"></i>
                    <p>No candidates found matching the criteria.</p>
                </div>
            `;
        } else {
            candidates.forEach((candidate, index) => {
                if (candidate) {
                    const candidateCard = createDynamicCandidateCard(candidate, index);
                    candidatesContainer.appendChild(candidateCard);
                }
            });
        }

        // Show DYNAMIC accuracy stats if available
        const accuracyStats = data.accuracy_stats;
        if (accuracyStats) {
            const maxScore = dynamicConfig && dynamicConfig.max_total_score ? dynamicConfig.max_total_score : 100;
            const statsMessage = `
                <div class="accuracy-stats" style="background: #e6fffa; padding: 10px; border-radius: 8px; margin-top: 10px;">
                    <strong>DYNAMIC Enhanced Accuracy Statistics:</strong><br>
                    High Confidence: ${accuracyStats.high_confidence || 0} | 
                    Medium Confidence: ${accuracyStats.medium_confidence || 0} | 
                    Low Confidence: ${accuracyStats.low_confidence || 0}<br>
                    <small>Evaluation Method: ${accuracyStats.evaluation_method || 'Dynamic AI Agents with Specific Summaries'}</small><br>
                    <small>Max Possible Score: ${maxScore} points (Dynamic Configuration)</small>
                </div>
            `;
            candidatesContainer.insertAdjacentHTML('afterbegin', statsMessage);
        }

        if (resultsSection) {
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    } catch (error) {
        console.error('Error displaying results:', error);
        showMessage('error', 'Error displaying results');
    }
}

function createDynamicCandidateCard(candidate, index) {
    try {
        if (!candidate) return document.createElement('div');
        
        const card = document.createElement('div');
        card.className = 'candidate-card';
        
        const score = Number(candidate.score) || 0;
        const maxScore = dynamicConfig && dynamicConfig.max_total_score ? dynamicConfig.max_total_score : 100;
        const confidenceLevel = candidate.confidence_level || 'UNKNOWN';
        const overallRating = candidate.overall_rating || 'Unknown';
        
        // Dynamic score color based on confidence and score percentage
        const scorePercentage = maxScore > 0 ? (score / maxScore) * 100 : 0;
        let scoreColor = '#e53e3e'; // Default red
        if (confidenceLevel === 'HIGH' && scorePercentage >= 60) {
            scoreColor = '#48bb78'; // Green for high confidence + good score
        } else if (confidenceLevel === 'MEDIUM' && scorePercentage >= 50) {
            scoreColor = '#ed8936'; // Orange for medium confidence + decent score
        }

        // Confidence badge color
        let confidenceColor = '#e53e3e';
        if (confidenceLevel === 'HIGH') confidenceColor = '#48bb78';
        else if (confidenceLevel === 'MEDIUM') confidenceColor = '#ed8936';

        card.innerHTML = `
            <div class="candidate-header">
                <div class="candidate-name">${escapeHtml(candidate.name || 'Unknown Candidate')}</div>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <div class="candidate-score" style="background: ${scoreColor};">
                        ${score.toFixed(1)}/${maxScore}
                    </div>
                    <div style="background: ${confidenceColor}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;">
                        ${confidenceLevel}
                    </div>
                </div>
            </div>
            <div class="candidate-summary-text" style="background: #f8fafc; padding: 12px; border-radius: 8px; margin: 10px 0;">
                <strong>${escapeHtml(candidate.name || 'Unknown')} scored ${score.toFixed(2)}/${maxScore} (${overallRating})</strong><br>
                ${escapeHtml(candidate.reason || 'No summary provided')}
            </div>
            <div class="button-group" style="margin-top: 10px;">
                <button class="toggle-details" onclick="toggleDetails(${index})">
                    <i class="fas fa-chevron-down"></i> Show Details
                </button>
            </div>
            <div class="individual-scores" id="details-${index}" style="display: none;">
                <div class="candidate-details">
                    ${formatDynamicStructuredScores(candidate.structured_scores || {}, confidenceLevel)}
                </div>
            </div>
        `;
        
        return card;
    } catch (error) {
        console.error('Error creating dynamic candidate card:', error);
        const errorCard = document.createElement('div');
        errorCard.className = 'candidate-card';
        errorCard.innerHTML = '<div class="error-message">Error displaying candidate data</div>';
        return errorCard;
    }
}

function formatDynamicStructuredScores(structuredScores, confidenceLevel) {
    try {
        if (!structuredScores || Object.keys(structuredScores).length === 0) {
            return '<div class="error-message">No detailed scoring data available.</div>';
        }

        let html = `<div style="background: #f8fafc; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
            <strong>Confidence Level: ${confidenceLevel}</strong> - This evaluation was performed using DYNAMIC enhanced specialized AI agents with evidence-based validation.
        </div>`;

        // Get dynamic scoring weights for display
        const scoringWeights = dynamicConfig && dynamicConfig.scoring_weights ? dynamicConfig.scoring_weights : {
            skills_max: 50,
            total_experience_max: 10,
            relevant_experience_max: 20,
            project_exposure_max: 20
        };

        // Process Mandatory Skills - FULLY DYNAMIC
        const mandatorySkills = structuredScores.mandatory_skills || {};
        if (Object.keys(mandatorySkills).length > 0) {
            const skillScore = safeDynamicGet(mandatorySkills, 'score', 'N/A');
            const requiredSkills = safeDynamicArray(mandatorySkills.required_skills);
            const matchedSkills = safeDynamicArray(mandatorySkills.matched_skills);
            const missingSkills = safeDynamicArray(mandatorySkills.missing_skills);
            const calculation = safeDynamicGet(mandatorySkills, 'calculation', 'Not available');
            
            // Determine if this is a low score for styling
            const isLowScore = skillScore !== 'N/A' && skillScore.includes('/') && 
                parseInt(skillScore.split('/')[0]) < (scoringWeights.skills_max * 0.6);
            
            html += `
                <h4 style="${isLowScore ? 'color: #e53e3e; font-weight: bold;' : ''}">
                    Technical Skills (Score: ${skillScore}/${scoringWeights.skills_max})
                </h4>
                <div style="background: #f7fafc; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
                    <h5>Required Skills (${requiredSkills.length})</h5>
                    <ul>
                        ${requiredSkills.map(skill => `<li>${escapeHtml(skill)}</li>`).join('')}
                    </ul>
                    
                    <h5 style="color: #48bb78; margin-top: 10px;">✅ Matched Skills (${matchedSkills.length})</h5>
                    <ul>
                        ${matchedSkills.map(skill => `<li style="color: #48bb78;">${escapeHtml(skill)}</li>`).join('')}
                    </ul>
                    
                    <h5 style="color: #e53e3e; margin-top: 10px;">❌ Missing Skills (${missingSkills.length})</h5>
                    <ul>
                        ${missingSkills.map(skill => `<li style="color: #e53e3e;">${escapeHtml(skill)}</li>`).join('')}
                    </ul>
                    
                    <div style="margin-top: 10px; font-weight: bold; color: #4a5568;">
                        Calculation: ${escapeHtml(calculation)}
                    </div>
                </div>
            `;
        }

        // Process Total Experience - FULLY DYNAMIC
        const totalExp = structuredScores.total_experience || {};
        if (Object.keys(totalExp).length > 0) {
            const expScore = safeDynamicGet(totalExp, 'score', 'N/A');
            const employmentPeriods = safeDynamicArray(totalExp.employment_periods);
            const totalExperience = safeDynamicGet(totalExp, 'total_experience', 'N/A');
            const required = safeDynamicGet(totalExp, 'required', 'N/A');
            const meetsRequirement = totalExp.meets_requirement === true;
            
            const isLowScore = expScore !== 'N/A' && expScore.includes('/') && 
                parseInt(expScore.split('/')[0]) === 0;
            
            html += `
                <h4 style="${isLowScore ? 'color: #e53e3e; font-weight: bold;' : ''}">
                    Total Experience (Score: ${expScore}/${scoringWeights.total_experience_max})
                </h4>
                <div style="background: #f7fafc; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
                    <ul>
                        <li><strong>Employment Periods:</strong> ${employmentPeriods.join(', ') || 'None listed'}</li>
                        <li><strong>Total Experience:</strong> ${escapeHtml(totalExperience)}</li>
                        <li><strong>Required:</strong> ${escapeHtml(required)}</li>
                        <li><strong>Meets Requirement:</strong> <span style="color: ${meetsRequirement ? '#48bb78' : '#e53e3e'};">
                            ${meetsRequirement ? '✅ Yes' : '❌ No'}
                        </span></li>
                    </ul>
                </div>
            `;
        }

        // Process Relevant Experience - FULLY DYNAMIC
        const relevantExp = structuredScores.relevant_experience || {};
        if (Object.keys(relevantExp).length > 0) {
            const relScore = safeDynamicGet(relevantExp, 'score', 'N/A');
            const relevantCompanies = safeDynamicArray(relevantExp.relevant_companies);
            const totalRelevantExperience = safeDynamicGet(relevantExp, 'total_relevant_experience', 'N/A');
            const threshold = safeDynamicGet(relevantExp, 'threshold', 'N/A');
            const meetsRequirement = relevantExp.meets_requirement === true;
            
            const isLowScore = relScore !== 'N/A' && relScore.includes('/') && 
                parseInt(relScore.split('/')[0]) === 0;
            
            html += `
                <h4 style="${isLowScore ? 'color: #e53e3e; font-weight: bold;' : ''}">
                    Relevant Experience (Score: ${relScore}/${scoringWeights.relevant_experience_max})
                </h4>
                <div style="background: #f7fafc; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
                    <ul>
                        <li><strong>Relevant Companies:</strong> ${relevantCompanies.join(', ') || 'None'}</li>
                        <li><strong>Total Relevant Experience:</strong> ${escapeHtml(totalRelevantExperience)}</li>
                        <li><strong>Threshold:</strong> ${escapeHtml(threshold)}</li>
                        <li><strong>Meets Requirement:</strong> <span style="color: ${meetsRequirement ? '#48bb78' : '#e53e3e'};">
                            ${meetsRequirement ? '✅ Yes' : '❌ No'}
                        </span></li>
                    </ul>
                </div>
            `;
        }

        // Process Project Exposure - FULLY DYNAMIC
        const projectExp = structuredScores.project_exposure || {};
        if (Object.keys(projectExp).length > 0) {
            const projScore = safeDynamicGet(projectExp, 'score', 'N/A');
            const e2eProjects = safeDynamicArray(projectExp.e2e_projects);
            const supportProjects = safeDynamicArray(projectExp.support_projects);
            const academicProjects = safeDynamicArray(projectExp.academic_unrelated);
            const scoringLogic = safeDynamicGet(projectExp, 'scoring_logic', 'Not available');
            
            const isLowScore = projScore !== 'N/A' && projScore.includes('/') && 
                parseInt(projScore.split('/')[0]) === 0;
            
            html += `
                <h4 style="${isLowScore ? 'color: #e53e3e; font-weight: bold;' : ''}">
                    Project Experience (Score: ${projScore}/${scoringWeights.project_exposure_max})
                </h4>
                <div style="background: #f7fafc; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
                    <h5 style="color: #48bb78;">E2E Projects:</h5>
                    <ul>
                        ${e2eProjects.length > 0 ? 
                            e2eProjects.map(project => `<li>${escapeHtml(project)}</li>`).join('') : 
                            '<li style="color: #a0aec0;">No End-to-End projects found</li>'
                        }
                    </ul>
                    
                    <h5 style="color: #ed8936; margin-top: 10px;">Support Projects:</h5>
                    <ul>
                        ${supportProjects.length > 0 ? 
                            supportProjects.map(project => `<li>${escapeHtml(project)}</li>`).join('') : 
                            '<li style="color: #a0aec0;">No support projects found</li>'
                        }
                    </ul>
                    
                    <h5 style="color: #4a5568; margin-top: 10px;">Academic/Unrelated:</h5>
                    <ul>
                        ${academicProjects.length > 0 ? 
                            academicProjects.map(project => `<li>${escapeHtml(project)}</li>`).join('') : 
                            '<li style="color: #a0aec0;">No academic/unrelated projects found</li>'
                        }
                    </ul>
                    
                    <div style="margin-top: 10px; font-weight: bold; color: #4a5568;">
                        Scoring Logic: ${escapeHtml(scoringLogic)}
                    </div>
                </div>
            `;
        }

        return html;
    } catch (error) {
        console.error('Error formatting dynamic structured scores:', error);
        return '<div class="error-message">Error formatting score details.</div>';
    }
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// CSV Download functionality with DYNAMIC support
document.addEventListener("DOMContentLoaded", function () {
    const downloadButton = document.getElementById("download-csv-btn");

    if (downloadButton) {
        downloadButton.addEventListener("click", async function () {
            try {
                showMessage('info', 'Preparing DYNAMIC enhanced CSV download...');
                
                const response = await fetch("/api/download-shortlisted");

                if (!response.ok) {
                    if (response.status === 404) {
                        showMessage('error', 'No candidates have been processed yet. Please run a screening first.');
                    } else {
                        showMessage('error', `Download failed with status: ${response.status}`);
                    }
                    return;
                }

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                
                // Dynamic filename with timestamp
                const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
                a.download = `dynamic_accuracy_screening_results_${timestamp}.csv`;
                
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                showMessage('success', 'DYNAMIC Enhanced CSV downloaded successfully!');
            } catch (err) {
                console.error("CSV download error:", err);
                showMessage('error', 'Failed to download CSV. Please try again.');
            }
        });
    }
});

// Global functions for onclick handlers
window.changePage = changePage;
window.removeResumeFile = removeResumeFile;
window.removeJDFile = removeJDFile;
window.toggleDetails = toggleDetails;

console.log('FULLY DYNAMIC Enhanced scripts.js loaded successfully for Structured Backend v4.2 - Completely Dynamic Display and Configuration');