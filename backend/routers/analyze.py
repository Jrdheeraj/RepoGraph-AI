import logging
from fastapi import APIRouter, HTTPException
from backend.schemas import AnalyzeRequest, AnalyzeResponse
from backend.services.analysis_service import analyze_repository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["analysis"]
)

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Analyze a GitHub repository and return health metrics and scores.
    """
    logger.info(f"Received request to analyze: {request.github_url}")
    try:
        # Pass the URL as string to the service
        url_str = str(request.github_url)
        result = analyze_repository(url_str)
        return AnalyzeResponse(**result)
    except ValueError as ve:
        logger.warning(f"Validation error during analysis: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal server error during analysis: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during analysis.")
