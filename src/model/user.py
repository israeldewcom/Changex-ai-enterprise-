from sqlalchemy import Column, String, Text, DateTime, JSON, Boolean, Integer, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from .base import EnhancedEnterpriseBaseModel
import uuid
from datetime import datetime

class User(EnhancedEnterpriseBaseModel):
    """Enhanced enterprise user management with advanced security"""
    __tablename__ = "users"
    
    # Authentication
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    password_changed_at = Column(DateTime, default=datetime.utcnow)
    
    # Personal Information
    full_name = Column(String(255), nullable=False)
    display_name = Column(String(255))
    avatar_url = Column(String(500))
    timezone = Column(String(50), default="UTC")
    locale = Column(String(10), default="en-US")
    
    # Professional Information
    company = Column(String(255), nullable=False, index=True)
    department = Column(String(100), index=True)
    job_title = Column(String(100))
    employee_id = Column(String(50), unique=True)
    
    # Enhanced Security
    is_verified = Column(Boolean, default=False, index=True)
    is_locked = Column(Boolean, default=False, index=True)
    verification_token = Column(String(100))
    reset_token = Column(String(100))
    lock_reason = Column(Text)
    locked_until = Column(DateTime)
    
    # MFA and Advanced Security
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(32))
    mfa_backup_codes = Column(JSONB)
    security_questions = Column(JSONB)
    last_password_reset = Column(DateTime)
    password_history = Column(JSONB)
    
    # Login Tracking
    last_login = Column(DateTime, index=True)
    last_login_ip = Column(String(45))
    last_activity = Column(DateTime, index=True)
    login_count = Column(Integer, default=0)
    failed_login_attempts = Column(Integer, default=0)
    failed_login_time = Column(DateTime)
    
    # Subscription and Billing
    subscription_plan = Column(String(50), default="starter", index=True)
    subscription_status = Column(String(20), default="active", index=True)
    subscription_expires = Column(DateTime, index=True)
    billing_address = Column(JSONB)
    payment_methods = Column(JSONB, default=list)
    
    # External Integrations
    paystack_customer_code = Column(String(100))
    stripe_customer_id = Column(String(255), unique=True, index=True)
    salesforce_contact_id = Column(String(100))
    hubspot_contact_id = Column(String(100))
    
    # Enhanced UI/UX Preferences
    theme_preference = Column(String(20), default="system")
    language_preference = Column(String(10), default="en")
    dashboard_layout = Column(JSONB, default=dict)
    notification_preferences = Column(JSONB, default=dict)
    accessibility_settings = Column(JSONB, default=dict)
    
    # Usage and Analytics
    feature_usage = Column(JSONB, default=dict)
    api_usage = Column(JSONB, default=dict)
    data_usage = Column(JSONB, default=dict)
    
    # Enhanced indexing for performance
    __table_args__ = (
        Index('ix_users_company_department', 'company', 'department'),
        Index('ix_users_role_active', 'is_active', 'subscription_status'),
        Index('ix_users_last_login', 'last_login'),
        Index('ix_users_subscription_status', 'subscription_status'),
        Index('ix_users_created_active', 'created_at', 'is_active'),
        Index('ix_users_email_verified', 'email', 'is_verified'),
    )
    
    def has_permission(self, permission_name: str) -> bool:
        """Check if user has specific permission"""
        # Implementation would check user permissions from database
        permissions = self.metadata.get('permissions', [])
        return permission_name in permissions
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role"""
        roles = self.metadata.get('roles', [])
        return role_name in roles
    
    def get_effective_permissions(self):
        """Get all effective permissions for user"""
        permissions = set(self.metadata.get('permissions', []))
        
        # Add role-based permissions
        for role in self.metadata.get('roles', []):
            role_permissions = self._get_role_permissions(role)
            permissions.update(role_permissions)
        
        return permissions
    
    def _get_role_permissions(self, role_name: str):
        """Get permissions for a specific role"""
        # This would typically query a roles table
        role_permissions = {
            'admin': ['read', 'write', 'delete', 'manage_users'],
            'user': ['read', 'write'],
            'viewer': ['read']
        }
        return role_permissions.get(role_name, [])
    
    def can_access_feature(self, feature_name: str) -> bool:
        """Check if user can access specific feature based on subscription"""
        subscription_features = {
            'free': ['basic_analysis', 'limited_documents'],
            'starter': ['basic_analysis', 'standard_documents', 'email_support'],
            'professional': ['advanced_analysis', 'unlimited_documents', 'priority_support'],
            'enterprise': ['all_features', 'dedicated_support', 'custom_integrations']
        }
        
        available_features = subscription_features.get(self.subscription_plan, [])
        return feature_name in available_features
