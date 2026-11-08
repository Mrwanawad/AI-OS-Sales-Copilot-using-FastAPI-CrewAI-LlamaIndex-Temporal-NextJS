from collections.abc import Callable
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ai_os.modules.identity.application.service import Identity
from ai_os.modules.identity.domain.errors import (
    AccountLocked,
    InvalidCredentials,
    InvalidUser,
    UserAlreadyExists,
    UserNotFound,
)
from ai_os.modules.identity.domain.models import User
from ai_os.modules.identity.domain.passwords import WeakPassword
from ai_os.shared.auth import Authenticator, Principal, Role


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    username: str
    display_name: str
    role: Role


class MeResponse(BaseModel):
    username: str
    role: Role
    expires_at: datetime | None


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class NewUser(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    display_name: str = Field(min_length=1, max_length=120)
    role: Role
    password: str = Field(min_length=1, max_length=256)


class PasswordReset(BaseModel):
    new_password: str = Field(min_length=1, max_length=256)


class UserResponse(BaseModel):
    username: str
    display_name: str
    role: Role
    active: bool
    locked_until: datetime | None


def create_router(get_identity: Callable[[], Identity], auth: Authenticator) -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["auth"])
    manager = auth.require(Role.MANAGER)
    IdentityDep = Annotated[Identity, Depends(get_identity)]

    @router.post("/login")
    async def login(request: LoginRequest, identity: IdentityDep) -> TokenResponse:
        try:
            issued = await identity.login(request.username, request.password)
        except InvalidCredentials as error:
            raise HTTPException(status_code=401, detail=str(error)) from error
        except AccountLocked as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
        return TokenResponse(
            access_token=issued.token,
            expires_at=issued.expires_at,
            username=issued.subject,
            display_name=issued.display_name,
            role=Role(issued.role),
        )

    @router.post("/logout", status_code=204)
    async def logout(user: Annotated[Principal, Depends(auth)], identity: IdentityDep) -> None:
        await identity.logout(user.token_id, user.expires_at)

    @router.get("/me")
    async def me(user: Annotated[Principal, Depends(auth)]) -> MeResponse:
        return MeResponse(username=user.subject, role=user.role, expires_at=user.expires_at)

    @router.post("/password", status_code=204)
    async def change_password(
        request: PasswordChange,
        user: Annotated[Principal, Depends(auth)],
        identity: IdentityDep,
    ) -> None:
        try:
            await identity.change_password(
                user.subject, request.current_password, request.new_password
            )
        except InvalidCredentials as error:
            raise HTTPException(status_code=403, detail="Current password is wrong") from error
        except WeakPassword as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.get("/users")
    async def list_users(
        _user: Annotated[Principal, Depends(manager)], identity: IdentityDep
    ) -> list[UserResponse]:
        return [_user_response(u) for u in await identity.list_users()]

    @router.post("/users", status_code=201)
    async def create_user(
        request: NewUser, _user: Annotated[Principal, Depends(manager)], identity: IdentityDep
    ) -> UserResponse:
        try:
            created = await identity.create_user(
                request.username, request.display_name, request.role.value, request.password
            )
        except (InvalidUser, WeakPassword) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except UserAlreadyExists as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return _user_response(created)

    @router.post("/users/{username}/end-sessions", status_code=204)
    async def end_sessions(
        username: str, _user: Annotated[Principal, Depends(manager)], identity: IdentityDep
    ) -> None:
        await _or_404(identity.end_sessions(username))

    @router.post("/users/{username}/disable", status_code=204)
    async def disable(
        username: str, user: Annotated[Principal, Depends(manager)], identity: IdentityDep
    ) -> None:
        if username == user.subject:
            raise HTTPException(status_code=409, detail="You can't disable your own account")
        await _or_404(identity.set_active(username, False))

    @router.post("/users/{username}/enable", status_code=204)
    async def enable(
        username: str, _user: Annotated[Principal, Depends(manager)], identity: IdentityDep
    ) -> None:
        await _or_404(identity.set_active(username, True))

    @router.post("/users/{username}/password", status_code=204)
    async def reset_password(
        username: str,
        request: PasswordReset,
        _user: Annotated[Principal, Depends(manager)],
        identity: IdentityDep,
    ) -> None:
        try:
            await _or_404(identity.reset_password(username, request.new_password))
        except WeakPassword as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return router


async def _or_404(action) -> None:
    try:
        await action
    except UserNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        username=user.subject,
        display_name=user.display_name,
        role=Role(user.role),
        active=user.active,
        locked_until=user.locked_until,
    )
