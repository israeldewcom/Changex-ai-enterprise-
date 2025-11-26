import os
import secrets
import base64
from typing import List, Dict, Any, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from cryptography.fernet import Fernet
import json

class EnterpriseSettings(BaseSettings):
    """Enhanced enterprise-grade configuration management with multi-environment support"""
    
    # Environment & Deployment
    ENVIRONMENT: str = Field(default="production", regex="^(development|staging|production)$")
    DEPLOYMENT_REGION: str = Field(default="us-east-1")
    CLUSTER_NAME: str = Field(default="changex-enterprise-cluster")
    APP_NAME: str = Field(default="ChangeX Enterprise AI")
    APP_VERSION: str = Field(default="7.0.0")
    
    # Enhanced Security Configuration
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(128))
    ALGORITHM: str = "HS512"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ENCRYPTION_KEY: str = Field(default_factory=lambda: base64.urlsafe_b64encode(Fernet.generate_key()).decode())
    JWT_ISSUER: str = "changex-enterprise-ai"
    JWT_AUDIENCE: str = "changex-enterprise-users"
    
    # Enhanced Database Configuration
    DATABASE_URL: str = Field(default="postgresql+asyncpg://user:pass@localhost:5432/changex_enterprise_ai")
    REDIS_URL: str = Field(default="redis://localhost:6379")
    DATABASE_POOL_SIZE: int = 50
    DATABASE_MAX_OVERFLOW: int = 100
    DATABASE_POOL_RECYCLE: int = 3600
    DATABASE_ECHO: bool = False
    
    # Enhanced AI Services Configuration
    OPENAI_API_KEY: str = Field(default="")
    ANTHROPIC_API_KEY: str = Field(default="")
    COHERE_API_KEY: str = Field(default="")
    HUGGINGFACE_TOKEN: str = Field(default="")
    GROQ_API_KEY: str = Field(default="")
    AWS_BEDROCK_ACCESS_KEY: str = Field(default="")
    AWS_BEDROCK_SECRET_KEY: str = Field(default="")
    
    # Enhanced Cloud Storage
    AWS_ACCESS_KEY_ID: str = Field(default="")
    AWS_SECRET_ACCESS_KEY: str = Field(default="")
    AWS_S3_BUCKET: str = Field(default="changex-ai-docs")
    GOOGLE_CLOUD_PROJECT: str = Field(default="")
    AZURE_STORAGE_CONNECTION_STRING: str = Field(default="")
    CLOUDFLARE_R2_ACCESS_KEY: str = Field(default="")
    CLOUDFLARE_R2_SECRET_KEY: str = Field(default="")
    
    # Enhanced Monitoring & Observability
    SENTRY_DSN: str = Field(default="")
    DATADOG_API_KEY: str = Field(default="")
    ELASTICSEARCH_URL: str = Field(default="http://localhost:9200")
    JAEGER_HOST: str = Field(default="localhost")
    GRAFANA_URL: str = Field(default="http://localhost:3000")
    GRAFANA_API_KEY: str = Field(default="")
    
    # Enhanced Enterprise Features
    LICENSE_KEY: str = Field(default="")
    COMPANY_NAME: str = Field(default="ChangeX Enterprise AI Corp")
    MAX_USERS: int = Field(default=10000)
    DATA_RETENTION_DAYS: int = Field(default=1095)  # 3 years
    
    # Enhanced Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=1000)
    RATE_LIMIT_BURST_SIZE: int = Field(default=100)
    RATE_LIMIT_USER_TIERS: Dict = Field(default={
        "free": 100,
        "starter": 1000,
        "professional": 10000,
        "enterprise": 100000
    })
    
    # Enhanced Performance Configuration
    MODEL_CACHE_SIZE: int = Field(default=50)
    EMBEDDING_CACHE_TTL: int = Field(default=7200)  # 2 hours
    REQUEST_TIMEOUT: int = Field(default=300)
    MAX_WORKERS: int = Field(default=100)
    
    # Enhanced Market Competitive Features
    COMPETITOR_ANALYSIS_ENABLED: bool = Field(default=True)
    REAL_TIME_MARKET_INTELLIGENCE: bool = Field(default=True)
    PREDICTIVE_ANALYTICS_ENABLED: bool = Field(default=True)
    AUTOMATED_GROWTH_OPTIMIZATION: bool = Field(default=True)
    AI_PRICING_OPTIMIZATION: bool = Field(default=True)
    CUSTOMER_CHURN_PREDICTION: bool = Field(default=True)
    
    # Enhanced Circuit Breaker Settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=10)
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(default=120)
    CIRCUIT_BREAKER_EXPECTED_EXCEPTIONS: tuple = Field(default=("HTTPException", "ConnectionError", "TimeoutError", "asyncio.TimeoutError"))
    
    # Enhanced Market Intelligence
    MARKET_DATA_API_KEY: str = Field(default="")
    FINANCIAL_DATA_API_KEY: str = Field(default="")
    COMPETITIVE_INTELLIGENCE_API_KEY: str = Field(default="")
    ALPHA_VANTAGE_API_KEY: str = Field(default="")
    FINNHUB_API_KEY: str = Field(default="")
    QUANDL_API_KEY: str = Field(default="")
    
    # Enhanced Payment Integrations
    PAYSTACK_SECRET_KEY: str = Field(default="")
    PAYSTACK_PUBLIC_KEY: str = Field(default="")
    PAYSTACK_BASE_URL: str = Field(default="https://api.paystack.co")
    STRIPE_SECRET_KEY: str = Field(default="")
    STRIPE_PUBLISHABLE_KEY: str = Field(default="")
    PAYPAL_CLIENT_ID: str = Field(default="")
    PAYPAL_CLIENT_SECRET: str = Field(default="")
    
    # Enhanced UI/UX Configuration
    SUPPORT_EMAIL: str = Field(default="support@changex-ai.com")
    COMPANY_LOGO_URL: str = Field(default="/static/images/changex-logo.png")
    FAVICON_URL: str = Field(default="/static/images/favicon.ico")
    
    # Enhanced Frontend Configuration
    THEME_PRIMARY_COLOR: str = Field(default="#2563eb")
    THEME_SECONDARY_COLOR: str = Field(default="#1e40af")
    THEME_ACCENT_COLOR: str = Field(default="#3b82f6")
    THEME_SUCCESS_COLOR: str = Field(default="#10b981")
    THEME_WARNING_COLOR: str = Field(default="#f59e0b")
    THEME_ERROR_COLOR: str = Field(default="#ef4444")
    DASHBOARD_REFRESH_INTERVAL: int = Field(default=30000)
    
    # Enhanced UI/UX Feature Flags
    DARK_MODE_ENABLED: bool = Field(default=True)
    REAL_TIME_NOTIFICATIONS: bool = Field(default=True)
    ADVANCED_ANALYTICS_DASHBOARD: bool = Field(default=True)
    CUSTOM_THEMING: bool = Field(default=True)
    MOBILE_RESPONSIVE: bool = Field(default=True)
    OFFLINE_MODE: bool = Field(default=True)
    PROGRESSIVE_WEB_APP: bool = Field(default=True)
    
    # Enhanced Base URL Configuration
    BASE_URL: str = Field(default="http://localhost:8000")
    CDN_URL: str = Field(default="https://cdn.changex-ai.com")
    API_BASE_URL: str = Field(default="https://api.changex-ai.com")
    
    # Enhanced Advanced AI Features
    COMPUTER_VISION_ENABLED: bool = Field(default=True)
    SPEECH_TO_TEXT_ENABLED: bool = Field(default=True)
    TIME_SERIES_PREDICTION_ENABLED: bool = Field(default=True)
    ANOMALY_DETECTION_ENABLED: bool = Field(default=True)
    RECOMMENDATION_ENGINE_ENABLED: bool = Field(default=True)
    GENERATIVE_AI_ENABLED: bool = Field(default=True)
    REINFORCEMENT_LEARNING_ENABLED: bool = Field(default=True)
    FEDERATED_LEARNING_ENABLED: bool = Field(default=True)
    
    # Enhanced Testing Configuration
    TESTING_ENABLED: bool = Field(default=True)
    AUTOMATED_TESTING: bool = Field(default=True)
    PERFORMANCE_TESTING: bool = Field(default=True)
    SECURITY_TESTING: bool = Field(default=True)
    LOAD_TESTING: bool = Field(default=True)
    
    # Enhanced Compliance Configuration
    GDPR_COMPLIANT: bool = Field(default=True)
    SOC2_COMPLIANT: bool = Field(default=True)
    HIPAA_COMPLIANT: bool = Field(default=False)
    PCI_DSS_COMPLIANT: bool = Field(default=True)
    ISO_27001_COMPLIANT: bool = Field(default=True)
    
    # Enhanced Security Headers
    CONTENT_SECURITY_POLICY: str = Field(default="default-src 'self'; script-src 'self' 'unsafe-inline'")
    STRICT_TRANSPORT_SECURITY: str = Field(default="max-age=31536000; includeSubDomains")
    X_CONTENT_TYPE_OPTIONS: str = Field(default="nosniff")
    X_FRAME_OPTIONS: str = Field(default="DENY")
    X_XSS_PROTECTION: str = Field(default="1; mode=block")
    
    # Enhanced Cache Configuration
    REDIS_CACHE_TTL: int = Field(default=3600)
    BROWSER_CACHE_TTL: int = Field(default=86400)
    CDN_CACHE_TTL: int = Field(default=604800)
    
    # Enhanced Email Configuration
    SMTP_SERVER: str = Field(default="smtp.gmail.com")
    SMTP_PORT: int = Field(default=587)
    SMTP_USERNAME: str = Field(default="")
    SMTP_PASSWORD: str = Field(default="")
    EMAIL_FROM: str = Field(default="noreply@changex-ai.com")
    
    # Enhanced Notification Configuration
    SLACK_BOT_TOKEN: str = Field(default="")
    SLACK_CHANNEL: str = Field(default="#alerts")
    TWILIO_ACCOUNT_SID: str = Field(default="")
    TWILIO_AUTH_TOKEN: str = Field(default="")
    TWILIO_PHONE_NUMBER: str = Field(default="")
    
    # Enhanced Analytics Configuration
    GOOGLE_ANALYTICS_ID: str = Field(default="")
    MIXPANEL_TOKEN: str = Field(default="")
    AMPLITUDE_API_KEY: str = Field(default="")
    HOTJAR_SITE_ID: str = Field(default="")
    
    # Enhanced Search Configuration
    ELASTICSEARCH_INDEX_PREFIX: str = Field(default="changex")
    SEARCH_ANALYZER: str = Field(default="standard")
    SEARCH_MAX_RESULTS: int = Field(default=1000)
    
    # Enhanced File Upload Configuration
    MAX_FILE_SIZE: int = Field(default=100 * 1024 * 1024)  # 100MB
    ALLOWED_FILE_TYPES: List[str] = Field(default=["pdf", "doc", "docx", "txt", "jpg", "png", "csv", "xlsx"])
    UPLOAD_DIR: str = Field(default="./uploads")
    
    # Enhanced Model Configuration
    DEFAULT_EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")
    DEFAULT_CLASSIFICATION_MODEL: str = Field(default="distilbert-base-uncased-finetuned-sst-2-english")
    DEFAULT_SUMMARIZATION_MODEL: str = Field(default="facebook/bart-large-cnn")
    DEFAULT_TRANSLATION_MODEL: str = Field(default="Helsinki-NLP/opus-mt-en-es")
    
    # Enhanced Business Intelligence
    BUSINESS_INTELLIGENCE_ENABLED: bool = Field(default=True)
    REAL_TIME_ANALYTICS: bool = Field(default=True)
    PREDICTIVE_MODELING: bool = Field(default=True)
    CUSTOMER_SEGMENTATION: bool = Field(default=True)
    MARKET_BASKET_ANALYSIS: bool = Field(default=True)
    
    # Enhanced Workflow Automation
    WORKFLOW_AUTOMATION_ENABLED: bool = Field(default=True)
    BUSINESS_PROCESS_MANAGEMENT: bool = Field(default=True)
    ROBOTIC_PROCESS_AUTOMATION: bool = Field(default=True)
    INTELLIGENT_DOCUMENT_PROCESSING: bool = Field(default=True)
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v:
            raise ValueError("DATABASE_URL must be set")
        return v
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        if len(v) < 64:
            raise ValueError("SECRET_KEY must be at least 64 characters long")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = 'utf-8'
        extra = 'allow'

# Global settings instance
settings = EnterpriseSettings()
