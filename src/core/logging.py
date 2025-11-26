import logging
import uuid
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import structlog
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import inspect
import sys

class EnhancedProductionLogger:
    """Enhanced production-grade structured logging with distributed tracing and advanced monitoring"""
    
    def __init__(self, settings):
        self.settings = settings
        self.correlation_id = str(uuid.uuid4())
        
        # Configure enhanced structlog for production
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
                structlog.processors.EventRenamer("message"),
                self._add_correlation_id,
                self._add_user_context,
                self._add_business_context,
                self._add_service_context
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger()
        
        # Initialize enhanced OpenTelemetry for distributed tracing
        self._init_enhanced_tracing()
        
        # Initialize enhanced metrics
        self._init_enhanced_metrics()
        
        # Initialize performance tracking
        self.performance_tracker = {}
        
    def _init_enhanced_tracing(self):
        """Initialize enhanced distributed tracing"""
        try:
            trace.set_tracer_provider(TracerProvider())
            
            # Jaeger for distributed tracing
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.settings.JAEGER_HOST,
                agent_port=6831,
            )
            
            # OTLP exporter for cloud providers
            otlp_exporter = OTLPSpanExporter(
                endpoint="http://localhost:4317",
                insecure=True,
            )
            
            trace.get_tracer_provider().add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
            
            trace.get_tracer_provider().add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            
            self.tracer = trace.get_tracer(__name__)
            self.logger.info("Enhanced distributed tracing initialized")
            
        except Exception as e:
            self.logger.error(f"Enhanced tracing initialization failed: {str(e)}")
    
    def _init_enhanced_metrics(self):
        """Initialize enhanced business and performance metrics"""
        try:
            # HTTP Metrics
            self.request_counter = Counter(
                'http_requests_total', 
                'Total HTTP requests', 
                ['method', 'endpoint', 'status', 'user_type', 'region']
            )
            self.request_duration = Histogram(
                'http_request_duration_seconds', 
                'HTTP request duration', 
                ['method', 'endpoint', 'status']
            )
            
            # Enhanced Business Metrics
            self.business_metrics = {
                'documents_processed': Counter(
                    'documents_processed_total', 
                    'Total documents processed',
                    ['industry', 'document_type', 'processing_type']
                ),
                'ai_predictions': Counter(
                    'ai_predictions_total', 
                    'Total AI predictions made',
                    ['model_type', 'prediction_type', 'confidence']
                ),
                'user_actions': Counter(
                    'user_actions_total',
                    'Total user actions performed',
                    ['action_type', 'user_tier', 'success']
                ),
                'revenue_events': Counter(
                    'revenue_events_total',
                    'Total revenue-generating events',
                    ['event_type', 'amount', 'currency']
                )
            }
            
            # Performance Metrics
            self.performance_metrics = {
                'response_time': Histogram(
                    'api_response_time_seconds',
                    'API response time distribution',
                    ['endpoint', 'method']
                ),
                'error_rate': Gauge(
                    'error_rate_percentage',
                    'Current error rate percentage'
                ),
                'active_users': Gauge(
                    'active_users_total',
                    'Current active users'
                )
            }
            
            self.logger.info("Enhanced metrics system initialized")
            
        except Exception as e:
            self.logger.error(f"Enhanced metrics initialization failed: {str(e)}")
    
    def _add_correlation_id(self, logger, method_name, event_dict):
        """Add correlation ID to log events"""
        if 'correlation_id' not in event_dict:
            event_dict['correlation_id'] = self.correlation_id
        return event_dict
    
    def _add_user_context(self, logger, method_name, event_dict):
        """Add user context to log events"""
        if 'user_id' not in event_dict:
            event_dict['user_id'] = 'system'
        if 'session_id' not in event_dict:
            event_dict['session_id'] = str(uuid.uuid4())
        return event_dict
    
    def _add_business_context(self, logger, method_name, event_dict):
        """Add business context to log events"""
        event_dict['business_unit'] = 'enterprise_ai'
        event_dict['environment'] = self.settings.ENVIRONMENT
        event_dict['deployment_region'] = self.settings.DEPLOYMENT_REGION
        event_dict['service_version'] = self.settings.APP_VERSION
        event_dict['company'] = self.settings.COMPANY_NAME
        return event_dict
    
    def _add_service_context(self, logger, method_name, event_dict):
        """Add service context to log events"""
        # Get calling function information
        frame = inspect.currentframe()
        try:
            for _ in range(6):  # Look through a few frames
                frame = frame.f_back
                if frame is None:
                    break
                code = frame.f_code
                if 'self' in frame.f_locals:
                    class_name = frame.f_locals['self'].__class__.__name__
                    event_dict['service'] = class_name
                    event_dict['method'] = code.co_name
                    break
        finally:
            del frame
        
        return event_dict
    
    async def log_business_event(self, event_type: str, user_id: str, details: Dict, 
                               severity: str = "INFO", business_value: float = 0.0):
        """Enhanced business event logging with value tracking"""
        try:
            with self.tracer.start_as_current_span("business_event") as span:
                span.set_attribute("event_type", event_type)
                span.set_attribute("user_id", user_id)
                span.set_attribute("severity", severity)
                span.set_attribute("business_value", business_value)
                
                log_data = {
                    "event_type": event_type,
                    "user_id": user_id,
                    "details": details,
                    "severity": severity,
                    "business_value": business_value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "trace_id": format(span.get_span_context().trace_id, '032x'),
                    "span_id": format(span.get_span_context().span_id, '016x'),
                    "environment": self.settings.ENVIRONMENT,
                    "deployment_region": self.settings.DEPLOYMENT_REGION
                }
                
                if severity == "ERROR":
                    self.logger.error(**log_data)
                elif severity == "WARNING":
                    self.logger.warning(**log_data)
                else:
                    self.logger.info(**log_data)
                
                # Track business metrics
                if event_type.startswith("REVENUE_"):
                    self.business_metrics['revenue_events'].inc(
                        event_type=event_type,
                        amount=details.get('amount', 0),
                        currency=details.get('currency', 'USD')
                    )
                    
        except Exception as e:
            self.logger.error(f"Business event logging failed: {str(e)}")
    
    async def log_performance_metric(self, metric_name: str, value: float, tags: Dict = None):
        """Log performance metrics with enhanced tracking"""
        try:
            if metric_name in self.performance_metrics:
                if tags:
                    self.performance_metrics[metric_name].labels(**tags).observe(value)
                else:
                    self.performance_metrics[metric_name].observe(value)
            
            self.logger.info(
                "performance_metric",
                metric_name=metric_name,
                value=value,
                tags=tags or {},
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Performance metric logging failed: {str(e)}")
    
    async def log_security_event(self, event_type: str, severity: str, user_id: str, details: Dict):
        """Enhanced security event logging"""
        try:
            security_context = {
                "event_type": event_type,
                "severity": severity,
                "user_id": user_id,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
                "ip_address": details.get('ip_address'),
                "user_agent": details.get('user_agent'),
                "action": details.get('action'),
                "resource": details.get('resource'),
                "outcome": details.get('outcome', 'unknown')
            }
            
            if severity in ["CRITICAL", "HIGH"]:
                self.logger.critical(**security_context)
            elif severity == "MEDIUM":
                self.logger.error(**security_context)
            else:
                self.logger.warning(**security_context)
                
        except Exception as e:
            self.logger.error(f"Security event logging failed: {str(e)}")
    
    def get_metrics(self):
        """Get all current metrics in Prometheus format"""
        try:
            return generate_latest()
        except Exception as e:
            self.logger.error(f"Metrics generation failed: {str(e)}")
            return b""
    
    def start_performance_tracking(self, operation_name: str):
        """Start tracking performance for an operation"""
        track_id = str(uuid.uuid4())
        self.performance_tracker[track_id] = {
            'operation': operation_name,
            'start_time': datetime.utcnow(),
            'start_perf_counter': asyncio.get_event_loop().time()
        }
        return track_id
    
    def end_performance_tracking(self, track_id: str, success: bool = True, metadata: Dict = None):
        """End performance tracking and log results"""
        try:
            if track_id not in self.performance_tracker:
                return
            
            track_data = self.performance_tracker[track_id]
            end_time = asyncio.get_event_loop().time()
            duration = end_time - track_data['start_perf_counter']
            
            performance_data = {
                'operation': track_data['operation'],
                'duration_seconds': duration,
                'success': success,
                'start_time': track_data['start_time'].isoformat(),
                'end_time': datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            
            self.logger.info("performance_tracking", **performance_data)
            
            # Log to metrics system
            asyncio.create_task(self.log_performance_metric(
                'response_time',
                duration,
                {'operation': track_data['operation']}
            ))
            
            # Clean up
            del self.performance_tracker[track_id]
            
        except Exception as e:
            self.logger.error(f"Performance tracking end failed: {str(e)}")
