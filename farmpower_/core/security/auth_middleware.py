"""
Enhanced Authentication Middleware for LogLineOS
Provides comprehensive authentication, authorization and rate limiting
Created: 2025-07-19 06:00:41 UTC
User: danvoulez
"""
import os
import json
import time
import logging
import asyncio
import hashlib
import hmac
import base64
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, Depends, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SecurityMiddleware")

class TokenData(BaseModel):
    """JWT token data"""
    sub: str
    exp: int
    iat: int
    permissions: List[str] = []
    tenant_id: Optional[str] = None
    user_id: str
    span_access: List[str] = []
    

class AuthConfig:
    """Configuration for authentication system"""
    JWT_SECRET: str = os.environ.get("JWT_SECRET", "INSECURE_DEFAULT_SECRET_CHANGE_ME")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60
    TOKEN_URL: str = "/token"
    GOD_KEY: str = os.environ.get("GOD_KEY", "user_special_key_never_share")
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_MAX_REQUESTS: Dict[str, int] = {
        "default": 100,
        "admin": 1000,
        "system": 10000
    }
    LOGGING_ENABLED: bool = True
    AUDIT_ENABLED: bool = True
    VERIFY_SIGNATURE: bool = True


class RateLimiter:
    """Rate limiting implementation with sliding window"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.request_records: Dict[str, List[float]] = {}
        self.cleanup_task = None
        
    async def start_cleanup(self):
        """Start periodic cleanup of expired records"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Periodically clean up expired rate limit records"""
        while True:
            await asyncio.sleep(self.config.RATE_LIMIT_WINDOW_SECONDS)
            await self.cleanup_expired_records()
    
    async def cleanup_expired_records(self):
        """Clean up expired rate limit records"""
        now = time.time()
        window_start = now - self.config.RATE_LIMIT_WINDOW_SECONDS
        
        for user_id, timestamps in list(self.request_records.items()):
            # Keep only timestamps within the window
            self.request_records[user_id] = [ts for ts in timestamps if ts > window_start]
            
            # Remove empty entries
            if not self.request_records[user_id]:
                del self.request_records[user_id]
    
    async def is_rate_limited(self, user_id: str, role: str = "default") -> bool:
        """
        Check if the user is rate limited
        
        Args:
            user_id: User identifier
            role: User role for determining limits
            
        Returns:
            True if rate limited, False otherwise
        """
        if not self.config.RATE_LIMIT_ENABLED:
            return False
            
        now = time.time()
        window_start = now - self.config.RATE_LIMIT_WINDOW_SECONDS
        
        # Get or initialize user's request timestamps
        timestamps = self.request_records.get(user_id, [])
        
        # Filter to current window
        current_window = [ts for ts in timestamps if ts > window_start]
        self.request_records[user_id] = current_window
        
        # Add current request
        self.request_records[user_id].append(now)
        
        # Get limit based on role
        limit = self.config.RATE_LIMIT_MAX_REQUESTS.get(role, self.config.RATE_LIMIT_MAX_REQUESTS["default"])
        
        # Check if over limit
        return len(current_window) > limit
    
    async def reset_limits(self, user_id: str):
        """Reset rate limits for a user"""
        if user_id in self.request_records:
            del self.request_records[user_id]


class AuthResult:
    """Result of authentication check"""
    
    def __init__(self, 
                valid: bool = False, 
                user_id: str = None,
                permissions: List[str] = None,
                tenant_id: str = None,
                reason: str = None,
                span_access: List[str] = None):
        self.valid = valid
        self.user_id = user_id
        self.permissions = permissions or []
        self.tenant_id = tenant_id
        self.reason = reason
        self.span_access = span_access or []
        self.timestamp = datetime.now()


class SecurityMiddleware:
    """Enhanced security middleware for LogLineOS"""
    
    def __init__(self, config: AuthConfig = None, audit_service = None):
        """Initialize security middleware"""
        self.config = config or AuthConfig()
        self.rate_limiter = RateLimiter(self.config)
        self.audit_service = audit_service
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl=self.config.TOKEN_URL)
        
        # Ensure JWT secret is secure
        if self.config.JWT_SECRET == "INSECURE_DEFAULT_SECRET_CHANGE_ME":
            logger.warning("Using default insecure JWT secret. Set JWT_SECRET environment variable!")
        
        # Start rate limiter cleanup
        asyncio.create_task(self.rate_limiter.start_cleanup())
        
        logger.info("Security middleware initialized")
    
    async def validate_token(self, token: str) -> AuthResult:
        """
        Validate JWT token
        
        Args:
            token: JWT token to validate
            
        Returns:
            AuthResult with validation details
        """
        if not token:
            return AuthResult(valid=False, reason="No token provided")
        
        # Check for god key
        if token == self.config.GOD_KEY:
            return AuthResult(
                valid=True,
                user_id="system",
                permissions=["*"],  # All permissions
                span_access=["*"],  # All spans
                reason="God key"
            )
        
        # Validate JWT token
        try:
            if self.config.VERIFY_SIGNATURE:
                payload = jwt.decode(
                    token, 
                    self.config.JWT_SECRET, 
                    algorithms=[self.config.JWT_ALGORITHM]
                )
            else:
                # For debugging only - NEVER use in production
                payload = jwt.decode(
                    token,
                    options={"verify_signature": False}
                )
                
            # Extract data
            user_id = payload.get("sub")
            permissions = payload.get("permissions", [])
            tenant_id = payload.get("tenant_id")
            span_access = payload.get("span_access", [])
            
            return AuthResult(
                valid=True,
                user_id=user_id,
                permissions=permissions,
                tenant_id=tenant_id,
                span_access=span_access
            )
            
        except jwt.ExpiredSignatureError:
            return AuthResult(valid=False, reason="Token expired")
        except jwt.InvalidTokenError:
            return AuthResult(valid=False, reason="Invalid token")
        except Exception as e:
            return AuthResult(valid=False, reason=f"Token validation error: {str(e)}")
    
    async def create_token(self, 
                         user_id: str, 
                         permissions: List[str] = None, 
                         tenant_id: str = None,
                         span_access: List[str] = None) -> str:
        """
        Create JWT token
        
        Args:
            user_id: User identifier
            permissions: List of permission strings
            tenant_id: Optional tenant identifier
            span_access: Optional list of span patterns user can access
            
        Returns:
            JWT token string
        """
        permissions = permissions or []
        span_access = span_access or []
        
        # Create token data
        now = datetime.utcnow()
        expires = now + timedelta(minutes=self.config.JWT_EXPIRATION_MINUTES)
        
        payload = {
            "sub": user_id,
            "exp": int(expires.timestamp()),
            "iat": int(now.timestamp()),
            "permissions": permissions,
            "user_id": user_id
        }
        
        if tenant_id:
            payload["tenant_id"] = tenant_id
            
        if span_access:
            payload["span_access"] = span_access
        
        # Create token
        token = jwt.encode(
            payload,
            self.config.JWT_SECRET,
            algorithm=self.config.JWT_ALGORITHM
        )
        
        return token
    
    async def verify_permission(self, 
                             permissions: List[str], 
                             required_permission: str) -> bool:
        """
        Verify if user has required permission
        
        Args:
            permissions: List of user permissions
            required_permission: Required permission string
            
        Returns:
            True if user has permission, False otherwise
        """
        # Super admin has all permissions
        if "*" in permissions:
            return True
        
        # Direct match
        if required_permission in permissions:
            return True
        
        # Check for wildcard permissions
        for permission in permissions:
            if permission.endswith("*"):
                prefix = permission[:-1]
                if required_permission.startswith(prefix):
                    return True
        
        return False
    
    async def verify_span_access(self, span_id: str, span_access: List[str]) -> bool:
        """
        Verify if user has access to a span
        
        Args:
            span_id: Span identifier
            span_access: List of span access patterns
            
        Returns:
            True if user has access, False otherwise
        """
        # Full access
        if "*" in span_access:
            return True
        
        # Direct match
        if span_id in span_access:
            return True
        
        # Pattern match
        for pattern in span_access:
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if span_id.startswith(prefix):
                    return True
        
        return False
    
    async def authenticate_request(self, request: Request) -> AuthResult:
        """
        Authenticate an HTTP request
        
        Args:
            request: FastAPI request object
            
        Returns:
            AuthResult with authentication details
        """
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        token = ""
        
        if auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
        
        # Validate token
        auth_result = await self.validate_token(token)
        
        if not auth_result.valid:
            return auth_result
        
        # Check rate limits
        role = "admin" if "admin" in auth_result.permissions else "default"
        
        if auth_result.user_id == "system":
            role = "system"
        
        is_rate_limited = await self.rate_limiter.is_rate_limited(auth_result.user_id, role)
        
        if is_rate_limited:
            return AuthResult(valid=False, reason="Rate limit exceeded", user_id=auth_result.user_id)
        
        # Audit the authentication if enabled
        if self.config.AUDIT_ENABLED and self.audit_service:
            await self.audit_service.create_entry(
                operation="authenticate",
                actor=auth_result.user_id,
                target_type="auth",
                target_id="token",
                status="success",
                details={
                    "ip": request.client.host,
                    "path": request.url.path,
                    "method": request.method,
                    "permissions": auth_result.permissions
                }
            )
        
        return auth_result
    
    async def require_permission(self, request: Request, required_permission: str) -> AuthResult:
        """
        Require a specific permission to access a resource
        
        Args:
            request: FastAPI request object
            required_permission: Required permission string
            
        Returns:
            AuthResult if successful
            
        Raises:
            HTTPException if permission check fails
        """
        auth_result = await self.authenticate_request(request)
        
        if not auth_result.valid:
            raise HTTPException(
                status_code=401,
                detail=f"Authentication failed: {auth_result.reason}"
            )
        
        # Check permission
        has_permission = await self.verify_permission(auth_result.permissions, required_permission)
        
        if not has_permission:
            # Audit the failed permission check
            if self.config.AUDIT_ENABLED and self.audit_service:
                await self.audit_service.create_entry(
                    operation="authorize",
                    actor=auth_result.user_id,
                    target_type="permission",
                    target_id=required_permission,
                    status="failure",
                    details={
                        "path": request.url.path,
                        "method": request.method,
                        "required_permission": required_permission,
                        "user_permissions": auth_result.permissions
                    }
                )
            
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {required_permission} is required"
            )
        
        return auth_result
    
    # FastAPI dependency for requiring authentication
    async def requires_auth(self, token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))):
        """FastAPI dependency for requiring authentication"""
        auth_result = await self.validate_token(token)
        
        if not auth_result.valid:
            raise HTTPException(
                status_code=401,
                detail=f"Authentication failed: {auth_result.reason}",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return auth_result
    
    # FastAPI dependency for requiring a specific permission
    def requires_permission(self, required_permission: str):
        """FastAPI dependency for requiring a specific permission"""
        
        async def permission_dependency(auth_result: AuthResult = Depends(self.requires_auth)):
            """Check if user has the required permission"""
            has_permission = await self.verify_permission(auth_result.permissions, required_permission)
            
            if not has_permission:
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {required_permission} is required"
                )
            
            return auth_result
        
        return permission_dependency


# Function to generate a secure random key
def generate_secure_key(length: int = 64) -> str:
    """Generate a secure random key"""
    return base64.b64encode(os.urandom(length)).decode('utf-8')


# Example usage with FastAPI
'''
from fastapi import FastAPI, Depends
from security.auth_middleware import SecurityMiddleware, AuthConfig

# Create app
app = FastAPI()

# Set up security middleware
config = AuthConfig(
    JWT_SECRET=os.environ.get("JWT_SECRET", generate_secure_key()),
    JWT_EXPIRATION_MINUTES=60,
    RATE_LIMIT_ENABLED=True
)
security = SecurityMiddleware(config)

# Protected endpoint
@app.get("/protected")
async def protected_route(auth_result = Depends(security.requires_auth)):
    return {"message": f"Hello, {auth_result.user_id}!"}

# Protected endpoint requiring specific permission
@app.get("/admin")
async def admin_route(auth_result = Depends(security.requires_permission("admin:access"))):
    return {"message": "Welcome, admin!"}
'''