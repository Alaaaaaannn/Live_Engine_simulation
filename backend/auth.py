"""
auth.py — JWT-based user authentication.

- Password hashing via bcrypt.
- Stateless access tokens signed with JWT_SECRET_KEY (HS256).
- `get_current_user` is a FastAPI dependency that resolves the bearer
  token into a User row from the database.
"""
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select

from db import User, get_session, _ENABLED as DB_ENABLED


SECRET_KEY   = os.getenv("JWT_SECRET_KEY", "dev-only-change-me-in-prod")
ALGORITHM    = "HS256"
TOKEN_TTL    = timedelta(days=7)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
router = APIRouter(prefix="/auth", tags=["Auth"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:    EmailStr
    password: str = Field(min_length=6, max_length=128)

class LoginRequest(BaseModel):
    email:    EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      str
    email:        str

class UserOut(BaseModel):
    id:    str
    email: str


# ── Hashing / JWT helpers ─────────────────────────────────────────────────────

def hash_password(pwd: str) -> str:
    return bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()

def verify_password(pwd: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pwd.encode(), hashed.encode())
    except ValueError:
        return False

def create_access_token(user_id: uuid.UUID, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub":   str(user_id),
        "email": email,
        "iat":   int(now.timestamp()),
        "exp":   int((now + TOKEN_TTL).timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ── Dependency: resolve bearer token to a User ────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    if not DB_ENABLED:
        raise HTTPException(503, "Authentication backend not configured "
                                  "(DATABASE_URL unset).")
    creds_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str: Optional[str] = payload.get("sub")
        if not user_id_str:
            raise creds_exc
        user_id = uuid.UUID(user_id_str)
    except (jwt.PyJWTError, ValueError):
        raise creds_exc

    async with get_session() as s:
        res = await s.execute(select(User).where(User.id == user_id))
        user = res.scalar_one_or_none()
        if user is None:
            raise creds_exc
        return user


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    if not DB_ENABLED:
        raise HTTPException(503, "Auth disabled — DATABASE_URL not set.")
    async with get_session() as s:
        res = await s.execute(select(User).where(User.email == req.email.lower()))
        if res.scalar_one_or_none() is not None:
            raise HTTPException(409, "Email already registered.")
        user = User(email=req.email.lower(), password_hash=hash_password(req.password))
        s.add(user)
        await s.commit()
        await s.refresh(user)
    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token, user_id=str(user.id), email=user.email)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    if not DB_ENABLED:
        raise HTTPException(503, "Auth disabled — DATABASE_URL not set.")
    async with get_session() as s:
        res = await s.execute(select(User).where(User.email == req.email.lower()))
        user = res.scalar_one_or_none()
        if user is None or not verify_password(req.password, user.password_hash):
            raise HTTPException(401, "Invalid email or password.")
    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token, user_id=str(user.id), email=user.email)


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return UserOut(id=str(user.id), email=user.email)
