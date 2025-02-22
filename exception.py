from typing import Optional, Dict, Any

class AppBaseError(Exception):
    """Base exception class for the application with structured error handling"""
    def __init__(
        self,
        message: str = "Application error occurred",
        error_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.message} [Code: {self.error_code}]"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.message}, code={self.error_code})>"

class SafetyAPIError(AppBaseError):
    """Represents failures in safety-related API operations"""
    def __init__(
        self,
        message: str = "Safety API operation failed",
        status_code: Optional[int] = None,
        url: Optional[str] = None,
        response_text: Optional[str] = None,
        **kwargs
    ):
        details = {
            'service': 'safety-api',
            'status_code': status_code,
            'url': url,
            'response': response_text
        }
        super().__init__(
            message=message,
            error_code=503 if status_code is None else status_code,
            details=details,
            **kwargs
        )

class TranslationError(AppBaseError):
    """Represents failures in translation operations"""
    def __init__(
        self,
        message: str = "Translation operation failed",
        target_language: Optional[str] = None,
        source_text: Optional[str] = None,
        **kwargs
    ):
        details = {
            'service': 'translation',
            'target_language': target_language,
            'source_text_snippet': source_text[:100] if source_text else None
        }
        super().__init__(
            message=message,
            error_code=400 if target_language else 500,
            details=details,
            **kwargs
        )

class RecommendationError(AppBaseError):
    """Represents failures in recommendation operations"""
    def __init__(
        self,
        message: str = "Recommendation system failure",
        recommendation_type: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ):
        details = {
            'service': 'recommendation',
            'recommendation_type': recommendation_type,
            'user_id': user_id
        }
        super().__init__(
            message=message,
            error_code=422,  # Unprocessable Entity
            details=details,
            **kwargs
        )
