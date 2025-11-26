import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import jwt
from jwt import PyJWTError
import bcrypt
from cryptography.fernet import Fernet
import re
import asyncio

class EnhancedSecurityEngine:
    """Enhanced security engine with advanced threat detection and enterprise security features"""
    
    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger
        self.fernet = Fernet(settings.ENCRYPTION_KEY.encode())
        self.failed_attempts = {}
        self.locked_accounts = {}
        self.rate_limits = {}
        self.suspicious_activities = []
        self.known_threats = self._load_known_threats()
        self.hmac_key = secrets.token_bytes(32)
        
    def _load_known_threats(self) -> Dict:
        """Load known threats and malicious patterns"""
        return {
            'suspicious_ips': set(),
            'malicious_user_agents': set(),
            'sql_injection_patterns': [
                r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
                r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
                r"\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",
                r"((\%27)|(\'))union"
            ],
            'xss_patterns': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"onload\s*=",
                r"onerror\s*=",
                r"onclick\s*="
            ]
        }
    
    async def initialize(self):
        """Initialize security engine"""
        try:
            # Load threat intelligence data
            await self._load_threat_intelligence()
            self.logger.logger.info("Enhanced security engine initialized")
        except Exception as e:
            self.logger.logger.error(f"Security engine initialization failed: {str(e)}")
            raise
    
    async def _load_threat_intelligence(self):
        """Load threat intelligence from external sources"""
        # This would integrate with threat intelligence feeds
        # For now, we'll use static data
        self.known_threats['suspicious_ips'].update([
            '192.168.1.100',  # Example suspicious IP
            '10.0.0.50'       # Example suspicious IP
        ])
        
        self.known_threats['malicious_user_agents'].update([
            'malicious-bot',
            'scanner-tool'
        ])
    
    async def hash_password(self, password: str) -> str:
        """Create secure password hash"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    async def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            await self.logger.log_security_event(
                event_type="PASSWORD_VERIFICATION_ERROR",
                severity="ERROR",
                user_id="unknown",
                details={"error": str(e)}
            )
            return False
    
    async def generate_jwt_token(self, payload: Dict) -> str:
        """Generate enhanced JWT token with security claims"""
        try:
            expire = datetime.utcnow() + timedelta(minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            
            to_encode = payload.copy()
            to_encode.update({
                "exp": expire,
                "iat": datetime.utcnow(),
                "iss": self.settings.JWT_ISSUER,
                "aud": self.settings.JWT_AUDIENCE,
                "jti": secrets.token_urlsafe(16)  # Unique token ID
            })
            
            # Add security claims
            to_encode.update({
                "security_level": "enhanced",
                "token_version": "2.0",
                "auth_method": "password"
            })
            
            encoded_jwt = jwt.encode(
                to_encode, 
                self.settings.SECRET_KEY, 
                algorithm=self.settings.ALGORITHM
            )
            
            await self.logger.log_security_event(
                event_type="JWT_TOKEN_GENERATED",
                severity="INFO",
                user_id=payload.get("sub", "unknown"),
                details={"token_type": "access", "expires": expire.isoformat()}
            )
            
            return encoded_jwt
            
        except Exception as e:
            await self.logger.log_security_event(
                event_type="JWT_GENERATION_ERROR",
                severity="ERROR",
                user_id=payload.get("sub", "unknown"),
                details={"error": str(e), "operation": "generate_jwt_token"}
            )
            raise

    async def verify_jwt_token(self, token: str) -> Dict:
        """Verify and decode JWT token with enhanced security checks"""
        try:
            payload = jwt.decode(
                token,
                self.settings.SECRET_KEY,
                algorithms=[self.settings.ALGORITHM],
                audience=self.settings.JWT_AUDIENCE,
                issuer=self.settings.JWT_ISSUER
            )
            
            # Additional security checks
            await self._perform_token_security_checks(payload, token)
            
            return payload
            
        except PyJWTError as e:
            await self.logger.log_security_event(
                event_type="JWT_VERIFICATION_FAILED",
                severity="WARNING",
                user_id="unknown",
                details={"error": str(e), "token": token[:50] + "..." if token else "none"}
            )
            raise
        except Exception as e:
            await self.logger.log_security_event(
                event_type="JWT_VERIFICATION_ERROR",
                severity="ERROR",
                user_id="unknown",
                details={"error": str(e), "operation": "verify_jwt_token"}
            )
            raise

    async def _perform_token_security_checks(self, payload: Dict, token: str):
        """Perform enhanced security checks on JWT token"""
        # Check token expiration
        if payload.get("exp") and datetime.utcnow() > datetime.fromtimestamp(payload["exp"]):
            raise PyJWTError("Token has expired")
        
        # Check token issuance time
        if payload.get("iat") and datetime.fromtimestamp(payload["iat"]) > datetime.utcnow():
            raise PyJWTError("Token issued in the future")
        
        # Check token ID
        if not payload.get("jti"):
            raise PyJWTError("Missing token ID")
        
        # Check security level
        if payload.get("security_level") != "enhanced":
            await self.logger.log_security_event(
                event_type="SUSPICIOUS_TOKEN",
                severity="MEDIUM",
                user_id=payload.get("sub", "unknown"),
                details={"reason": "invalid_security_level", "token_id": payload.get("jti")}
            )

    async def detect_threats(self, request_data: Dict, user_context: Dict = None) -> Dict:
        """Enhanced threat detection with behavioral analysis"""
        threats = []
        risk_score = 0
        
        # IP Address Analysis
        ip_threats = await self._analyze_ip_address(request_data.get('ip_address'))
        threats.extend(ip_threats)
        risk_score += len(ip_threats) * 10
        
        # User Agent Analysis
        ua_threats = await self._analyze_user_agent(request_data.get('user_agent'))
        threats.extend(ua_threats)
        risk_score += len(ua_threats) * 5
        
        # Behavioral Analysis
        behavior_threats = await self._analyze_behavior(request_data, user_context)
        threats.extend(behavior_threats)
        risk_score += len(behavior_threats) * 15
        
        # Rate Limiting Analysis
        rate_threats = await self._analyze_rate_limits(request_data, user_context)
        threats.extend(rate_threats)
        risk_score += len(rate_threats) * 20
        
        # Geographic Analysis
        geo_threats = await self._analyze_geography(request_data)
        threats.extend(geo_threats)
        risk_score += len(geo_threats) * 8
        
        return {
            "threats_detected": threats,
            "risk_score": min(100, risk_score),
            "risk_level": self._calculate_risk_level(risk_score),
            "recommended_action": self._get_recommended_action(risk_score),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _analyze_ip_address(self, ip_address: str) -> List[str]:
        """Analyze IP address for threats"""
        threats = []
        
        if not ip_address:
            return threats
        
        # Check against known malicious IPs
        if ip_address in self.known_threats['suspicious_ips']:
            threats.append(f"IP {ip_address} is in known threat database")
        
        # Check for suspicious IP patterns
        if self._is_suspicious_ip(ip_address):
            threats.append(f"IP {ip_address} matches suspicious pattern")
        
        # Check for VPN/Tor usage
        if await self._is_vpn_or_tor(ip_address):
            threats.append(f"IP {ip_address} appears to be VPN/Tor exit node")
        
        return threats

    async def _analyze_user_agent(self, user_agent: str) -> List[str]:
        """Analyze user agent for threats"""
        threats = []
        
        if not user_agent:
            return threats
        
        # Check for malicious user agents
        if user_agent in self.known_threats['malicious_user_agents']:
            threats.append(f"User agent {user_agent} is known malicious")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'curl', r'wget', r'scanner', r'bot', r'spider',
            r'sqlmap', r'nmap', r'nikto', r'metasploit'
        ]
        
        ua_lower = user_agent.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, ua_lower):
                threats.append(f"User agent contains suspicious pattern: {pattern}")
                break
        
        # Check for missing or generic user agents
        if user_agent in ['', 'Mozilla/5.0']:
            threats.append("Suspiciously generic or missing user agent")
        
        return threats

    async def _analyze_behavior(self, request_data: Dict, user_context: Dict) -> List[str]:
        """Analyze user behavior for anomalies"""
        threats = []
        user_id = user_context.get('user_id') if user_context else 'unknown'
        
        # Track failed login attempts
        if request_data.get('action') == 'login_failed':
            await self._track_failed_attempt(user_id, request_data)
            current_attempts = self.failed_attempts.get(user_id, 0)
            if current_attempts > 5:
                threats.append(f"Excessive failed login attempts: {current_attempts}")
        
        # Check for unusual activity timing
        if await self._is_unusual_activity_time(user_id, request_data):
            threats.append("Activity at unusual time for user")
        
        # Check for geographic anomalies
        if await self._is_geographic_anomaly(user_id, request_data):
            threats.append("Unusual geographic access pattern")
        
        return threats

    async def _analyze_rate_limits(self, request_data: Dict, user_context: Dict) -> List[str]:
        """Analyze request rate for anomalies"""
        threats = []
        user_id = user_context.get('user_id') if user_context else request_data.get('ip_address', 'unknown')
        
        # Initialize rate tracking
        current_time = time.time()
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
        
        # Clean old entries
        self.rate_limits[user_id] = [
            ts for ts in self.rate_limits[user_id] 
            if current_time - ts < 60  # Keep only last minute
        ]
        
        # Add current request
        self.rate_limits[user_id].append(current_time)
        
        # Check rate limit
        request_count = len(self.rate_limits[user_id])
        if request_count > self.settings.RATE_LIMIT_REQUESTS_PER_MINUTE:
            threats.append(f"High request rate detected: {request_count} requests/minute")
        
        return threats

    async def _analyze_geography(self, request_data: Dict) -> List[str]:
        """Analyze geographic patterns for threats"""
        threats = []
        
        # This would integrate with geoIP services in production
        ip_address = request_data.get('ip_address')
        if not ip_address:
            return threats
        
        # Simplified geographic analysis
        suspicious_countries = ['CN', 'RU', 'KP', 'IR']  # Example list
        # In production, you would lookup the country from IP
        
        return threats

    async def _track_failed_attempt(self, user_id: str, request_data: Dict):
        """Track failed authentication attempts"""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = 0
        
        self.failed_attempts[user_id] += 1
        
        # Lock account if too many failures
        if self.failed_attempts[user_id] >= 10:
            lock_duration = 3600  # 1 hour
            self.locked_accounts[user_id] = time.time() + lock_duration
            
            await self.logger.log_security_event(
                event_type="ACCOUNT_LOCKED",
                severity="HIGH",
                user_id=user_id,
                details={
                    "reason": "too_many_failed_attempts",
                    "attempts": self.failed_attempts[user_id],
                    "lock_duration": lock_duration,
                    "ip_address": request_data.get('ip_address')
                }
            )

    async def _is_unusual_activity_time(self, user_id: str, request_data: Dict) -> bool:
        """Check if activity occurs at unusual time for user"""
        # Simplified implementation
        # In production, you would analyze user's historical activity patterns
        current_hour = datetime.utcnow().hour
        return current_hour < 6 or current_hour > 22  # Unusual if outside 6AM-10PM UTC

    async def _is_geographic_anomaly(self, user_id: str, request_data: Dict) -> bool:
        """Check for geographic anomalies in user access"""
        # Simplified implementation
        # In production, you would compare with user's usual locations
        return False

    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious"""
        # Check for private IPs making external requests
        if ip_address.startswith(('10.', '172.', '192.168.')):
            return True
        
        # Check for known suspicious patterns
        suspicious_patterns = [
            r'^0\.0\.0\.0$',
            r'^127\.0\.0\.1$',
            r'^255\.255\.255\.255$'
        ]
        
        for pattern in suspicious_patterns:
            if re.match(pattern, ip_address):
                return True
        
        return False

    async def _is_vpn_or_tor(self, ip_address: str) -> bool:
        """Check if IP is VPN or Tor exit node"""
        # Simplified implementation
        # In production, integrate with VPN/Tor detection services
        return False

    def _calculate_risk_level(self, risk_score: int) -> str:
        """Calculate risk level from score"""
        if risk_score >= 70:
            return "CRITICAL"
        elif risk_score >= 50:
            return "HIGH"
        elif risk_score >= 30:
            return "MEDIUM"
        elif risk_score >= 10:
            return "LOW"
        else:
            return "MINIMAL"

    def _get_recommended_action(self, risk_score: int) -> str:
        """Get recommended action based on risk score"""
        if risk_score >= 70:
            return "BLOCK_AND_ALERT"
        elif risk_score >= 50:
            return "REQUIRE_2FA"
        elif risk_score >= 30:
            return "CAPTCHA_CHALLENGE"
        elif risk_score >= 10:
            return "MONITOR_CLOSELY"
        else:
            return "ALLOW"

    async def generate_secure_random(self, length: int = 32) -> str:
        """Generate cryptographically secure random string"""
        return secrets.token_urlsafe(length)

    async def create_hmac_signature(self, data: str) -> str:
        """Create HMAC signature for data integrity"""
        hmac_obj = hmac.new(
            self.hmac_key,
            data.encode('utf-8'),
            hashlib.sha256
        )
        return hmac_obj.hexdigest()

    async def verify_hmac_signature(self, data: str, signature: str) -> bool:
        """Verify HMAC signature for data integrity"""
        expected_signature = await self.create_hmac_signature(data)
        return hmac.compare_digest(expected_signature, signature)

    async def audit_security_event(self, event_type: str, user_id: str, details: Dict):
        """Log security event for auditing"""
        await self.logger.log_security_event(
            event_type=event_type,
            severity="INFO",
            user_id=user_id,
            details=details
        )

    async def cleanup_expired_data(self):
        """Clean up expired security data"""
        try:
            current_time = time.time()
            
            # Clean expired failed attempts
            self.failed_attempts = {
                user_id: count 
                for user_id, count in self.failed_attempts.items() 
                if current_time - self.failed_attempts.get(f"{user_id}_timestamp", 0) < 3600
            }
            
            # Clean expired locked accounts
            self.locked_accounts = {
                user_id: expiry 
                for user_id, expiry in self.locked_accounts.items() 
                if expiry > current_time
            }
            
            # Clean old rate limit data
            for user_id in list(self.rate_limits.keys()):
                self.rate_limits[user_id] = [
                    ts for ts in self.rate_limits[user_id] 
                    if current_time - ts < 300  # Keep only last 5 minutes
                ]
                if not self.rate_limits[user_id]:
                    del self.rate_limits[user_id]
            
            self.logger.logger.info("Security data cleanup completed")
            
        except Exception as e:
            self.logger.logger.error(f"Security data cleanup failed: {str(e)}")

    async def get_security_report(self) -> Dict:
        """Generate comprehensive security report"""
        return {
            "failed_attempts_count": len(self.failed_attempts),
            "locked_accounts_count": len(self.locked_accounts),
            "suspicious_activities_count": len(self.suspicious_activities),
            "current_threats": list(self.known_threats['suspicious_ips'])[:10],
            "security_metrics": {
                "total_blocks": sum(1 for event in self.suspicious_activities if "BLOCK" in event),
                "total_alerts": len(self.suspicious_activities),
                "average_risk_score": 25,  # Example metric
            },
            "timestamp": datetime.utcnow().isoformat()
        }
