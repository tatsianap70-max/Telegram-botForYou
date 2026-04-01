"""Проверки минимальной активации legal/consent слоя через конфиг и локали."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOCALES_DIR = PROJECT_ROOT / "locales"

RU_AGE_NOTICE = "Использование бота предназначено для пользователей 18+."
EN_AGE_NOTICE = "This bot is intended for users 18+."
RU_DISCLAIMER = (
    "Этот бот — инструмент для размышления и понимания ситуации. "
    "Он не является медицинской или профессиональной помощью."
)
EN_DISCLAIMER = (
    "This bot is a tool for reflection and understanding your situation. "
    "It is not medical or professional advice."
)


def _load_locale(locale_name: str) -> dict[str, str]:
    """Загрузить YAML-локаль как словарь ключей."""
    path = LOCALES_DIR / f"{locale_name}.yaml"
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _load_main_config() -> dict[str, object]:
    """Загрузить config.yaml как сырой YAML без runtime-переопределений."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def test_legal_is_enabled_and_has_documents_in_config() -> None:
    """Legal-слой включён и ссылки на документы настроены в config.yaml."""
    config = _load_main_config()
    legal = config["legal"]
    assert isinstance(legal, dict)

    assert legal["enabled"] is True
    assert bool(legal["privacy_policy_url"])
    assert bool(legal["terms_of_service_url"])


def test_legal_age_notice_and_disclaimer_exist_in_locales() -> None:
    """В RU/EN локалях присутствуют отдельные ключи 18+ и disclaimer."""
    ru_locale = _load_locale("ru")
    en_locale = _load_locale("en")

    assert ru_locale["legal_age_notice"] == RU_AGE_NOTICE
    assert ru_locale["legal_disclaimer"] == RU_DISCLAIMER

    assert en_locale["legal_age_notice"] == EN_AGE_NOTICE
    assert en_locale["legal_disclaimer"] == EN_DISCLAIMER


def test_legal_acceptance_request_contains_minimal_legal_texts() -> None:
    """Тексты 18+ и disclaimer встроены в legal_acceptance_request RU/EN."""
    ru_locale = _load_locale("ru")
    en_locale = _load_locale("en")

    ru_request = ru_locale["legal_acceptance_request"]
    en_request = en_locale["legal_acceptance_request"]

    assert RU_AGE_NOTICE in ru_request
    assert RU_DISCLAIMER in ru_request
    assert EN_AGE_NOTICE in en_request
    assert EN_DISCLAIMER in en_request
