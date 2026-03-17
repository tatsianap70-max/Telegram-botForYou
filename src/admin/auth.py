"""Аутентификация для веб-админки.

Реализует простую аутентификацию по логину/паролю из переменных окружения.
Данные сессии хранятся в signed cookies (защищены от подделки).
"""

import secrets

from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from typing_extensions import override

from src.config.settings import settings


class AdminAuth(AuthenticationBackend):
    """Backend аутентификации для SQLAdmin.

    Проверяет логин и пароль из настроек приложения.
    Сессия хранится в cookie с подписью (itsdangerous).
    """

    @override
    async def login(self, request: Request) -> bool:
        """Обработать форму входа.

        Args:
            request: HTTP-запрос с формой логина.

        Returns:
            True если вход успешен, False если нет.
        """
        form = await request.form()
        username = form.get("username")
        password = form.get("password")

        # Проверяем что админка настроена
        if not settings.admin.is_enabled:
            return False

        # Проверяем логин и пароль
        # Используем secrets.compare_digest для защиты от timing attacks
        username_valid = secrets.compare_digest(
            str(username or ""),
            settings.admin.username or "",
        )
        # Получаем пароль из настроек (если он есть)
        expected_password = ""
        if settings.admin.password:
            expected_password = settings.admin.password.get_secret_value()

        password_valid = secrets.compare_digest(
            str(password or ""),
            expected_password,
        )

        if username_valid and password_valid:
            # Сохраняем токен в сессию
            request.session.update({"admin_authenticated": True})
            return True

        return False

    @override
    async def logout(self, request: Request) -> bool:
        """Выйти из админки.

        Args:
            request: HTTP-запрос.

        Returns:
            Всегда True (выход всегда успешен).
        """
        request.session.clear()
        return True

    @override
    async def authenticate(self, request: Request) -> bool:
        """Проверить, авторизован ли пользователь.

        Args:
            request: HTTP-запрос.

        Returns:
            True если пользователь авторизован, False если нет.
        """
        return bool(request.session.get("admin_authenticated", False))


def get_admin_secret_key() -> str:
    """Получить секретный ключ для сессий админки.

    Секретный ключ берётся из настроек или генерируется автоматически.
    При автогенерации ключ меняется при каждом перезапуске —
    все сессии станут невалидными.

    Returns:
        Секретный ключ для подписи сессий.
    """
    if settings.admin.secret_key:
        return settings.admin.secret_key.get_secret_value()
    # Генерируем ключ — сессии не сохранятся между перезапусками
    return secrets.token_urlsafe(32)


def get_admin_auth() -> AdminAuth:
    """Создать backend аутентификации с секретным ключом.

    Returns:
        Настроенный AuthenticationBackend.
    """
    return AdminAuth(secret_key=get_admin_secret_key())
