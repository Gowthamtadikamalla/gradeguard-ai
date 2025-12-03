from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any

class PredictRequest(BaseModel):
    """Request schema for student academic standing prediction.
    
    Predicts whether a student is at-risk or in good academic standing using two-feature labeling:
    Pass requires GPA >= 2.0 AND AvgTestScore >= 73.
    All other students are labeled as Fail (at-risk).
    AttendanceRate is used as a predictive feature but not in target labeling.
    
    All features are REQUIRED for accurate predictions, as the model was trained on these features.
    Test scores are excluded to prevent data leakage (they're directly tied to GPA).
    Only actual_GPA is optional (used for validation/testing purposes only).
    """
    # Demographic features (REQUIRED)
    Age: int = Field(..., ge=14, le=18, description="Student age (14-18)")
    Grade: int = Field(..., ge=9, le=12, description="Grade level (9-12)")
    SES_Quartile: int = Field(..., ge=1, le=4, description="Socioeconomic status quartile (1-4)")
    ParentalEducation: str = Field(..., description="Parent education: <HS, HS, SomeCollege, Bachelors+")
    SchoolType: str = Field(..., description="School type: Public, Private")
    Locale: str = Field(..., description="School location: Suburban, City, Rural, Town")
    
    # Behavioral/Study characteristics (REQUIRED)
    AttendanceRate: float = Field(..., ge=0.0, le=1.0, description="Attendance rate (0.0-1.0)")
    StudyHours: float = Field(..., ge=0, le=4, description="Study hours per day (0-4)")
    # Note: Test scores (TestScore_Math, TestScore_Reading, TestScore_Science) are excluded
    # because they are directly tied to GPA, which would create data leakage
    
    # Behavioral/Personal features (REQUIRED)
    InternetAccess: int = Field(..., ge=0, le=1, description="Internet access: 1=yes, 0=no")
    Extracurricular: int = Field(..., ge=0, le=1, description="Extracurricular activities: 1=yes, 0=no")
    PartTimeJob: int = Field(..., ge=0, le=1, description="Part-time job: 1=yes, 0=no")
    ParentSupport: int = Field(..., ge=0, le=1, description="Parent support: 1=yes, 0=no")
    Romantic: int = Field(..., ge=0, le=1, description="Romantic relationship: 1=yes, 0=no")
    FreeTime: int = Field(..., ge=1, le=5, description="Free time scale (1-5)")
    GoOut: int = Field(..., ge=1, le=5, description="Go out frequency (1-5)")
    
    # Optional: For validation/testing - actual GPA if known (not used for prediction)
    actual_GPA: Optional[float] = Field(None, ge=0.0, le=4.0, description="Actual GPA (for validation only, not used in prediction)")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 17,
                "Grade": 12,
                "SES_Quartile": 3,
                "ParentalEducation": "HS",
                "SchoolType": "Public",
                "Locale": "Suburban",
                "AttendanceRate": 0.906,
                "StudyHours": 1.089,
                "InternetAccess": 1,
                "Extracurricular": 1,
                "PartTimeJob": 0,
                "ParentSupport": 0,
                "Romantic": 0,
                "FreeTime": 2,
                "GoOut": 2
            }
        }

class Recommendation(BaseModel):
    """Individual recommendation item for at-risk students."""
    model_config = ConfigDict(protected_namespaces=())
    priority: int = Field(..., ge=1, le=5, description="Priority level (1=highest impact)")
    category: str = Field(..., description="Category: GPA, TestScore, Attendance, or General")
    action: str = Field(..., description="Specific actionable step the student should take")
    reason: str = Field(..., description="Why this recommendation helps improve academic performance")
    timeline: str = Field(..., description="Expected timeline for seeing improvement")

class RecommendationsResponse(BaseModel):
    """Response containing AI-generated recommendations."""
    model_config = ConfigDict(protected_namespaces=())
    recommendations: List[Recommendation] = Field(..., description="List of personalized recommendations")
    summary: str = Field(..., description="Brief summary of the recommendations")

class SHAPFactor(BaseModel):
    """Top SHAP factors contributing to the prediction."""
    model_config = ConfigDict(protected_namespaces=())
    feature: str = Field(..., description="Feature name")
    impact: float = Field(..., description="SHAP contribution (positive increases pass probability)")
    direction: str = Field(..., description="positive or negative impact on pass probability")
    description: Optional[str] = Field(None, description="Human-readable explanation of the impact")

class PredictResponse(BaseModel):
    """Response schema for student academic standing prediction."""
    model_config = ConfigDict(protected_namespaces=())
    pass_prob: float = Field(..., ge=0.0, le=1.0, description="Probability of good academic standing (Pass: GPA >= 2.0 AND AvgTestScore >= 73 AND AttendanceRate >= 0.85)")
    pass_fail: int = Field(..., ge=0, le=1, description="Binary prediction: 1=good academic standing (Pass), 0=at-risk (Fail)")
    model_version: str = Field(..., description="Model version identifier")
    latency_ms: Optional[float] = Field(None, description="Prediction latency in milliseconds")
    recommendations: Optional[RecommendationsResponse] = Field(
        None,
        description="Personalized guidance from AI agent: improvement recommendations for at-risk students, encouragement and growth tips for passing students"
    )
    failed_conditions: Optional[Dict[str, bool]] = Field(
        None,
        description="Which conditions failed: {'gpa': bool, 'test_score': bool, 'attendance': bool}"
    )
    shap_factors: Optional[List[SHAPFactor]] = Field(
        None,
        description="Top SHAP factors explaining the prediction"
    )

