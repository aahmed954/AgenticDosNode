"""Authentication and authorization for the Claude Code proxy."""

import hashlib
import secrets
from typing import Optional, List, Dict, Set
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Request


class APIKeyValidator:
    """Validates API keys for the proxy."""

    def __init__(self, valid_keys: List[str]):
        # Hash the keys for security
        self.valid_key_hashes: Set[str] = {
            self._hash_key(key) for key in valid_keys if key
        }
        self.key_usage: Dict[str, Dict] = {}

    def _hash_key(self, key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()

    def validate_key(self, key: str) -> bool:
        """Validate an API key."""
        if not key:
            return False

        key_hash = self._hash_key(key)
        is_valid = key_hash in self.valid_key_hashes

        # Track usage
        if is_valid:
            self._track_usage(key_hash)

        return is_valid

    def _track_usage(self, key_hash: str):
        """Track API key usage for monitoring."""
        if key_hash not in self.key_usage:
            self.key_usage[key_hash] = {
                "first_used": datetime.now(),
                "last_used": datetime.now(),
                "request_count": 0
            }

        self.key_usage[key_hash]["last_used"] = datetime.now()
        self.key_usage[key_hash]["request_count"] += 1

    def get_usage_stats(self) -> Dict[str, Dict]:
        """Get usage statistics for all keys."""
        return {
            f"key_{i}": {
                "last_used": stats["last_used"].isoformat(),
                "request_count": stats["request_count"],
                "days_active": (datetime.now() - stats["first_used"]).days
            }
            for i, stats in enumerate(self.key_usage.values())
        }


class BearerTokenValidator:
    """Validates bearer tokens."""

    def __init__(self, secret_key: str, token: Optional[str] = None):
        self.secret_key = secret_key
        self.valid_token = token
        self.token_usage = {}

    def validate_token(self, token: str) -> bool:
        """Validate a bearer token."""
        if not token or not self.valid_token:
            return False

        is_valid = secrets.compare_digest(token, self.valid_token)

        if is_valid:
            self._track_usage(token[:8])  # Track by token prefix for security

        return is_valid

    def _track_usage(self, token_prefix: str):
        """Track token usage."""
        if token_prefix not in self.token_usage:
            self.token_usage[token_prefix] = {
                "first_used": datetime.now(),
                "last_used": datetime.now(),
                "request_count": 0
            }

        self.token_usage[token_prefix]["last_used"] = datetime.now()
        self.token_usage[token_prefix]["request_count"] += 1


class JWTValidator:
    """Validates JWT tokens."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def validate_jwt(self, token: str) -> Optional[Dict]:
        """Validate and decode a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )

            # Check expiration
            if "exp" in payload:
                if datetime.fromtimestamp(payload["exp"]) < datetime.now():
                    return None

            return payload

        except jwt.InvalidTokenError:
            return None

    def create_jwt(self, payload: Dict, expires_in_hours: int = 24) -> str:
        """Create a JWT token."""
        payload["exp"] = datetime.now() + timedelta(hours=expires_in_hours)
        payload["iat"] = datetime.now()

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)


class ProxyAuth:
    """Main authentication handler for the proxy."""

    def __init__(
        self,
        auth_method: str,
        api_keys: List[str] = None,
        bearer_token: Optional[str] = None,
        jwt_secret: Optional[str] = None
    ):
        self.auth_method = auth_method.lower()
        self.api_key_validator = APIKeyValidator(api_keys or []) if api_keys else None
        self.bearer_validator = BearerTokenValidator(
            jwt_secret or "default-secret", bearer_token
        ) if bearer_token else None
        self.jwt_validator = JWTValidator(jwt_secret or "default-secret") if jwt_secret else None

        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }

    async def authenticate(self, request: Request) -> Optional[str]:
        """
        Authenticate a request and return the authenticated identity.
        Returns None if authentication fails.
        """
        if self.auth_method == "none":
            return "anonymous"

        # Extract authentication info from request
        auth_header = request.headers.get("Authorization")
        api_key_header = request.headers.get("X-API-Key")

        if self.auth_method == "api_key":
            return await self._authenticate_api_key(auth_header, api_key_header)
        elif self.auth_method == "bearer_token":
            return await self._authenticate_bearer_token(auth_header)
        elif self.auth_method == "oauth2":
            return await self._authenticate_jwt(auth_header)

        return None

    async def _authenticate_api_key(
        self,
        auth_header: Optional[str],
        api_key_header: Optional[str]
    ) -> Optional[str]:
        """Authenticate using API key."""
        api_key = None

        # Check Authorization header (Bearer format)
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]  # Remove "Bearer " prefix
        # Check X-API-Key header
        elif api_key_header:
            api_key = api_key_header

        if api_key and self.api_key_validator and self.api_key_validator.validate_key(api_key):
            return f"api_key_{hashlib.sha256(api_key.encode()).hexdigest()[:8]}"

        return None

    async def _authenticate_bearer_token(self, auth_header: Optional[str]) -> Optional[str]:
        """Authenticate using bearer token."""
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]  # Remove "Bearer " prefix

        if self.bearer_validator and self.bearer_validator.validate_token(token):
            return f"bearer_{token[:8]}"

        return None

    async def _authenticate_jwt(self, auth_header: Optional[str]) -> Optional[str]:
        """Authenticate using JWT."""
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]  # Remove "Bearer " prefix

        if self.jwt_validator:
            payload = self.jwt_validator.validate_jwt(token)
            if payload:
                return payload.get("sub", f"jwt_{token[:8]}")

        return None

    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers to add to responses."""
        return self.security_headers.copy()

    def get_auth_stats(self) -> Dict[str, any]:
        """Get authentication statistics."""
        stats = {
            "auth_method": self.auth_method,
            "total_keys": len(self.api_key_validator.valid_key_hashes) if self.api_key_validator else 0
        }

        if self.api_key_validator:
            stats["api_key_usage"] = self.api_key_validator.get_usage_stats()

        if self.bearer_validator:
            stats["bearer_token_usage"] = len(self.bearer_validator.token_usage)

        return stats


class AuthMiddleware:
    """FastAPI middleware for authentication."""

    def __init__(self, proxy_auth: ProxyAuth):
        self.proxy_auth = proxy_auth

    async def __call__(self, request: Request, call_next):
        """Process authentication for incoming requests."""
        # Skip authentication for health check and metrics endpoints
        if request.url.path in ["/health", "/metrics", "/stats"]:
            response = await call_next(request)
            # Add security headers
            for header, value in self.proxy_auth.get_security_headers().items():
                response.headers[header] = value
            return response

        # Authenticate the request
        identity = await self.proxy_auth.authenticate(request)

        if identity is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # Add identity to request state
        request.state.authenticated_identity = identity

        # Process the request
        response = await call_next(request)

        # Add security headers
        for header, value in self.proxy_auth.get_security_headers().items():
            response.headers[header] = value

        return response