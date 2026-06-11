from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from src.constants import DEFAULT_CONFIG_PATH

from .models import BeaconConfig, LoggingConfig, MssmConfig, UcsdhConfig, VietnamConfig


class _YamlConfigSettingsSource(PydanticBaseSettingsSource):
    _config_path: Path | None = None

    @classmethod
    def set_config_path(cls, path: Path):
        cls._config_path = path

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str] | None:
        if self._config_path is None:
            return None

        encoding = self.config.get("env_file_encoding")
        file_content = self._read_config_file(encoding)

        field_value = file_content.get(field_name)
        return field_value, field_name

    def _read_config_file(self, encoding: str | None) -> dict[str, Any]:
        if self._config_path is None:
            raise ValueError("Config path not set")  # noqa: TRY003

        with open(self._config_path, encoding=encoding) as f:
            return yaml.safe_load(f) or {}

    def __call__(self) -> dict[str, Any]:
        return self._read_config_file(self.config.get("env_file_encoding"))


class Config(BaseSettings):
    """
    Configuration for the application.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_nested_delimiter="__",
    )

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ucsdh: UcsdhConfig = Field(default_factory=UcsdhConfig)
    vietnam: VietnamConfig = Field(default_factory=VietnamConfig)
    beacon: BeaconConfig = Field(default_factory=BeaconConfig)
    mssm: MssmConfig = Field(default_factory=MssmConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            _YamlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


def load_config(config_path: Path) -> Config:
    """
    Loads the configuration from the given path.

    Returns:
        Config: The loaded configuration.
    """
    _YamlConfigSettingsSource.set_config_path(config_path)
    return Config()


conf = load_config(DEFAULT_CONFIG_PATH)

__all__ = ["conf"]
