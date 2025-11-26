from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, Integer, Boolean, Text, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from datetime import datetime
import json

Base = declarative_base()

class EnhancedEnterpriseBaseModel(Base):
    """Enhanced base model with enterprise features and performance optimizations"""
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    version = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    metadata = Column(JSONB, default=dict)
    audit_trail = Column(JSONB, default=list)
    
    # Enhanced performance optimizations
    __table_args__ = {
        'postgresql_partition_by': 'RANGE (created_at)',
        'postgresql_with': {'fillfactor': '90'}
    }
    
    def to_dict(self, include_relationships: bool = False):
        """Convert model to dictionary with enhanced options"""
        result = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        
        # Convert UUID to string
        if 'id' in result and result['id']:
            result['id'] = str(result['id'])
        
        # Convert datetime to ISO format
        for date_field in ['created_at', 'updated_at']:
            if date_field in result and result[date_field]:
                result[date_field] = result[date_field].isoformat()
        
        if include_relationships:
            for relationship in self.__mapper__.relationships:
                rel_value = getattr(self, relationship.key)
                if rel_value is not None:
                    if isinstance(rel_value, list):
                        result[relationship.key] = [item.to_dict() for item in rel_value]
                    else:
                        result[relationship.key] = rel_value.to_dict()
        
        return result
    
    def to_json(self):
        """Convert model to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def get_partition_name(cls, date: datetime):
        """Generate partition name for time-based partitioning"""
        return f"{cls.__tablename__}_{date.strftime('%Y_%m')}"
    
    def add_audit_entry(self, action: str, user_id: str, details: dict = None):
        """Add entry to audit trail"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'user_id': user_id,
            'details': details or {}
        }
        
        if not self.audit_trail:
            self.audit_trail = []
        
        self.audit_trail.append(audit_entry)
        
        # Keep only last 100 audit entries
        if len(self.audit_trail) > 100:
            self.audit_trail = self.audit_trail[-100:]
