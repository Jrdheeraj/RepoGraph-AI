from pydantic import BaseModel, HttpUrl

class AnalyzeRequest(BaseModel):
    """Schema for repository analysis request."""
    github_url: HttpUrl


class AnalyzeResponse(BaseModel):
    """Schema for repository analysis response."""
    repository: str
    iq_score: float
    maintainability_risk: float
    technical_debt_score: float
    architecture_quality: float
    repograph_score: float
