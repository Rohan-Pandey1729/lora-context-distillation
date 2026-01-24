#!/usr/bin/env python3
"""Custom Singularity environment with explicit repo bind mount."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SingularityEnvironmentConfig:
    image: str
    cwd: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_SINGULARITY_EXECUTABLE", "singularity")
    """Path to the singularity executable."""
    sandbox_build_retries: int = 3
    """Number of retries for building the sandbox if an error occurs."""
    bind_host: str = ".container_work"
    """Host path to bind into the container."""
    bind_container: str = "/work"
    """Container path to use for the bound host directory."""
    no_mount: list[str] = field(default_factory=lambda: ["cwd", "bind-paths"])
    """Apptainer mount categories to disable via --no-mount."""


class SingularityEnvironment:
    def __init__(
        self, *, config_class: type = SingularityEnvironmentConfig, logger: logging.Logger | None = None, **kwargs
    ):
        """Singularity environment. See `SingularityEnvironmentConfig` for kwargs."""
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.sandbox_dir = self._build_sandbox()
        self._bind_host: Path | None = None
        self._prepare_paths()

    def _build_sandbox(self) -> Path:
        # Building the sandbox can fail (very rarely), so we retry it
        max_retries = self.config.sandbox_build_retries
        for attempt in range(max_retries):
            sandbox_dir = Path(tempfile.gettempdir()) / f"minisweagent-{uuid.uuid4().hex[:8]}"
            try:
                subprocess.run(
                    [self.config.executable, "build", "--sandbox", sandbox_dir, self.config.image],
                    check=True,
                    capture_output=True,
                )
                break
            except subprocess.CalledProcessError as e:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                self.logger.error(
                    f"Error building image {self.config.image}, stdout: {e.stdout}, stderr: {e.stderr} (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    raise
        return sandbox_dir

    def _prepare_paths(self) -> None:
        if self.config.bind_host and self.config.bind_container:
            self._bind_host = self._resolve_bind_host(self.config.bind_host)
            self._bind_host.mkdir(parents=True, exist_ok=True)
            self._ensure_container_path(self.config.bind_container)
        self._ensure_container_path(self.config.cwd)

    def _resolve_bind_host(self, path: str) -> Path:
        host = Path(path).expanduser()
        if not host.is_absolute():
            host = Path.cwd() / host
        return host.absolute()

    def _ensure_container_path(self, path: str) -> None:
        if not path or path == "/":
            return
        if not path.startswith("/"):
            raise ValueError(f"Container path must be absolute: {path}")
        target = self.sandbox_dir / path.lstrip("/")
        target.mkdir(parents=True, exist_ok=True)

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in a Singularity container and return the result as a dict."""
        cmd = [self.config.executable, "exec"]

        # Do not inherit directories and env vars from host
        cmd.extend(["--contain", "--cleanenv"])

        if self.config.no_mount:
            items = [item for item in self.config.no_mount if item]
            if items:
                cmd.extend(["--no-mount", ",".join(items)])

        if self._bind_host is not None:
            cmd.extend(["--bind", f"{self._bind_host}:{self.config.bind_container}"])

        work_dir = cwd or self.config.cwd
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.extend(["--writable", str(self.sandbox_dir), "bash", "-c", command])
        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def __del__(self):
        """Cleanup sandbox when object is destroyed."""
        self.cleanup()
