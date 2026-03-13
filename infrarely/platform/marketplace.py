"""
aos/marketplace.py — Agent Marketplace
═══════════════════════════════════════════════════════════════════════════════
A registry where developers publish capabilities and other developers
install them like npm packages.

    # CLI usage
    aos install @community/web-researcher
    aos install @community/code-reviewer
    aos install @enterprise/salesforce-sync

    # Python usage
    from infrarely.platform.marketplace import marketplace
    marketplace.install("@community/web-researcher")
    agent = infrarely.agent("my-agent", capabilities=[marketplace.get("web-researcher")])

This is the AWS Marketplace moment. When the ecosystem grows around the
platform, the platform becomes the standard.

Architecture:
  PackageMeta        — metadata for a published capability package
  PackageVersion     — immutable snapshot of a versioned release
  MarketplaceRegistry — in-memory registry (local mode) + HTTP fetch (remote)
  InstalledPackage   — local record of an installed package
  PackageManager     — install / uninstall / update / list / search
  PackageValidator   — validates package structure before publish
  Marketplace        — high-level façade combining registry + manager

Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MARKETPLACE_VERSION = "1.0"

# Valid scope prefixes
VALID_SCOPES = {"community", "enterprise", "official", "experimental"}

# Package name regex: @scope/name  (e.g. @community/web-researcher)
_PACKAGE_NAME_RE = re.compile(
    r"^@(?P<scope>[a-z][a-z0-9_-]*)\/(?P<name>[a-z][a-z0-9_-]*)$"
)

# Semver-ish: major.minor.patch  (simple form)
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class PackageStatus(Enum):
    """Status of a published package."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    UNLISTED = "unlisted"
    YANKED = "yanked"


class PackageCategory(Enum):
    """Standard capability categories."""

    RESEARCH = "research"
    WRITING = "writing"
    CODING = "coding"
    DATA = "data"
    INTEGRATION = "integration"
    ANALYSIS = "analysis"
    EDUCATION = "education"
    PRODUCTIVITY = "productivity"
    SECURITY = "security"
    UTILITY = "utility"
    OTHER = "other"


# ═══════════════════════════════════════════════════════════════════════════════
# PACKAGE METADATA
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PackageMeta:
    """
    Metadata for a capability package in the marketplace.

    A package is addressed as ``@scope/name`` — e.g. ``@community/web-researcher``.
    """

    # Required
    name: str  # e.g. "@community/web-researcher"
    version: str = "0.1.0"  # semver
    description: str = ""

    # Author
    author: str = ""
    author_email: str = ""
    license: str = "MIT"

    # Classification
    category: str = "other"
    tags: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)

    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # other packages
    python_requires: str = ">=3.10"
    aos_requires: str = ">=0.1.0"

    # Source
    homepage: str = ""
    repository: str = ""
    documentation: str = ""

    # Registry metadata
    downloads: int = 0
    rating: float = 0.0
    verified: bool = False
    status: str = "active"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def scope(self) -> str:
        """Extract scope from name (e.g. 'community' from '@community/web-researcher')."""
        m = _PACKAGE_NAME_RE.match(self.name)
        return m.group("scope") if m else ""

    @property
    def short_name(self) -> str:
        """Extract short name (e.g. 'web-researcher' from '@community/web-researcher')."""
        m = _PACKAGE_NAME_RE.match(self.name)
        return m.group("name") if m else self.name

    @property
    def qualified_name(self) -> str:
        """Full qualified name."""
        return self.name

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "author_email": self.author_email,
            "license": self.license,
            "category": self.category,
            "tags": list(self.tags),
            "capabilities": list(self.capabilities),
            "dependencies": list(self.dependencies),
            "python_requires": self.python_requires,
            "aos_requires": self.aos_requires,
            "homepage": self.homepage,
            "repository": self.repository,
            "documentation": self.documentation,
            "downloads": self.downloads,
            "rating": self.rating,
            "verified": self.verified,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PackageMeta":
        return cls(
            name=d.get("name", ""),
            version=d.get("version", "0.1.0"),
            description=d.get("description", ""),
            author=d.get("author", ""),
            author_email=d.get("author_email", ""),
            license=d.get("license", "MIT"),
            category=d.get("category", "other"),
            tags=d.get("tags", []),
            capabilities=d.get("capabilities", []),
            dependencies=d.get("dependencies", []),
            python_requires=d.get("python_requires", ">=3.10"),
            aos_requires=d.get("aos_requires", ">=0.1.0"),
            homepage=d.get("homepage", ""),
            repository=d.get("repository", ""),
            documentation=d.get("documentation", ""),
            downloads=d.get("downloads", 0),
            rating=d.get("rating", 0.0),
            verified=d.get("verified", False),
            status=d.get("status", "active"),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )

    @classmethod
    def from_json(cls, s: str) -> "PackageMeta":
        return cls.from_dict(json.loads(s))


# ═══════════════════════════════════════════════════════════════════════════════
# PACKAGE VERSION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PackageVersion:
    """An immutable versioned snapshot of a package."""

    package_name: str
    version: str
    meta: PackageMeta
    checksum: str = ""  # sha256 of content
    install_fn: Optional[Callable] = None
    capabilities_code: str = ""  # source code for the capability
    published_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_name": self.package_name,
            "version": self.version,
            "meta": self.meta.to_dict(),
            "checksum": self.checksum,
            "capabilities_code": self.capabilities_code,
            "published_at": self.published_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PackageVersion":
        return cls(
            package_name=d.get("package_name", ""),
            version=d.get("version", "0.1.0"),
            meta=PackageMeta.from_dict(d.get("meta", {})),
            checksum=d.get("checksum", ""),
            capabilities_code=d.get("capabilities_code", ""),
            published_at=d.get("published_at", time.time()),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLED PACKAGE
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class InstalledPackage:
    """Record of a locally installed marketplace package."""

    name: str
    version: str
    meta: PackageMeta
    installed_at: float = field(default_factory=time.time)
    install_path: str = ""
    active: bool = True
    checksum: str = ""
    capability_fn: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "meta": self.meta.to_dict(),
            "installed_at": self.installed_at,
            "install_path": self.install_path,
            "active": self.active,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InstalledPackage":
        return cls(
            name=d.get("name", ""),
            version=d.get("version", "0.1.0"),
            meta=PackageMeta.from_dict(d.get("meta", {})),
            installed_at=d.get("installed_at", time.time()),
            install_path=d.get("install_path", ""),
            active=d.get("active", True),
            checksum=d.get("checksum", ""),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PACKAGE VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════


class PackageValidator:
    """Validates package metadata and structure before publishing."""

    @staticmethod
    def validate_name(name: str) -> List[str]:
        """Validate a package name. Returns list of error strings."""
        errors: List[str] = []
        if not name:
            errors.append("Package name is required")
            return errors
        if not _PACKAGE_NAME_RE.match(name):
            errors.append(
                f"Invalid package name '{name}'. "
                "Must match @scope/name (e.g. @community/web-researcher)"
            )
        return errors

    @staticmethod
    def validate_version(version: str) -> List[str]:
        """Validate a semver version string."""
        errors: List[str] = []
        if not version:
            errors.append("Version is required")
            return errors
        if not _SEMVER_RE.match(version):
            errors.append(f"Invalid version '{version}'. Must be semver (e.g. 1.0.0)")
        return errors

    @staticmethod
    def validate_meta(meta: PackageMeta) -> List[str]:
        """Full validation of a PackageMeta. Returns list of error strings."""
        errors: List[str] = []
        errors.extend(PackageValidator.validate_name(meta.name))
        errors.extend(PackageValidator.validate_version(meta.version))
        if not meta.description:
            errors.append("Description is required")
        if len(meta.description) > 500:
            errors.append("Description must be <= 500 characters")
        if meta.category and meta.category not in [c.value for c in PackageCategory]:
            errors.append(f"Invalid category '{meta.category}'")
        for dep in meta.dependencies:
            dep_errors = PackageValidator.validate_name(dep)
            if dep_errors:
                errors.append(f"Invalid dependency: {dep}")
        return errors


# ═══════════════════════════════════════════════════════════════════════════════
# MARKETPLACE REGISTRY — the npm registry equivalent
# ═══════════════════════════════════════════════════════════════════════════════


class MarketplaceRegistry:
    """
    In-memory package registry — the marketplace "server" side.

    In production, this would be backed by a database + HTTP API.
    For the SDK, we keep it in-memory with optional JSON persistence.

    Thread-safe.
    """

    def __init__(self, storage_path: str = ""):
        self._lock = threading.Lock()
        self._packages: Dict[str, PackageMeta] = {}  # name → latest meta
        self._versions: Dict[str, Dict[str, PackageVersion]] = (
            {}
        )  # name → {ver → snapshot}
        self._storage_path = storage_path

        if storage_path and os.path.isfile(storage_path):
            self._load_from_disk()

    # ── Publish ───────────────────────────────────────────────────────────

    def publish(
        self,
        meta: PackageMeta,
        *,
        capabilities_code: str = "",
        install_fn: Optional[Callable] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Publish a package to the registry.

        Returns (success, errors). Validates before accepting.
        """
        errors = PackageValidator.validate_meta(meta)
        if errors:
            return False, errors

        with self._lock:
            # Check if this exact version already exists
            if meta.name in self._versions:
                if meta.version in self._versions[meta.name]:
                    return False, [
                        f"Version {meta.version} of {meta.name} already exists. "
                        "Bump the version number."
                    ]

            checksum = hashlib.sha256(
                (meta.to_json() + capabilities_code).encode()
            ).hexdigest()[:16]

            version = PackageVersion(
                package_name=meta.name,
                version=meta.version,
                meta=meta,
                checksum=checksum,
                install_fn=install_fn,
                capabilities_code=capabilities_code,
            )

            # Store
            meta.updated_at = time.time()
            self._packages[meta.name] = meta
            if meta.name not in self._versions:
                self._versions[meta.name] = {}
            self._versions[meta.name][meta.version] = version

            if self._storage_path:
                self._save_to_disk()

        return True, []

    # ── Query ─────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[PackageMeta]:
        """Get latest metadata for a package."""
        with self._lock:
            return self._packages.get(name)

    def get_version(self, name: str, version: str) -> Optional[PackageVersion]:
        """Get a specific version of a package."""
        with self._lock:
            versions = self._versions.get(name, {})
            return versions.get(version)

    def get_latest_version(self, name: str) -> Optional[PackageVersion]:
        """Get the latest published version."""
        with self._lock:
            versions = self._versions.get(name, {})
            if not versions:
                return None
            # Sort by semver (simple sort — works for well-formed semver)
            latest_key = sorted(versions.keys(), key=self._semver_key)[-1]
            return versions[latest_key]

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a package."""
        with self._lock:
            versions = self._versions.get(name, {})
            return sorted(versions.keys(), key=self._semver_key)

    def search(
        self,
        query: str = "",
        *,
        category: str = "",
        scope: str = "",
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[PackageMeta]:
        """
        Search the registry.

        Matches against name, description, tags, category.
        """
        with self._lock:
            results: List[PackageMeta] = []
            q = query.lower()

            for meta in self._packages.values():
                if meta.status != "active":
                    continue

                # Scope filter
                if scope and meta.scope != scope:
                    continue

                # Category filter
                if category and meta.category != category:
                    continue

                # Tag filter
                if tags:
                    if not any(t in meta.tags for t in tags):
                        continue

                # Query text search
                if q:
                    searchable = (
                        f"{meta.name} {meta.description} "
                        f"{' '.join(meta.tags)} {meta.category}"
                    ).lower()
                    if q not in searchable:
                        continue

                results.append(meta)

                if len(results) >= limit:
                    break

            return results

    def list_all(self) -> List[PackageMeta]:
        """List all active packages."""
        with self._lock:
            return [m for m in self._packages.values() if m.status == "active"]

    def count(self) -> int:
        """Total number of packages in the registry."""
        with self._lock:
            return len(self._packages)

    def remove(self, name: str) -> bool:
        """Remove a package from the registry entirely."""
        with self._lock:
            if name not in self._packages:
                return False
            del self._packages[name]
            self._versions.pop(name, None)
            if self._storage_path:
                self._save_to_disk()
            return True

    def deprecate(self, name: str) -> bool:
        """Mark a package as deprecated."""
        with self._lock:
            meta = self._packages.get(name)
            if not meta:
                return False
            meta.status = "deprecated"
            meta.updated_at = time.time()
            if self._storage_path:
                self._save_to_disk()
            return True

    def clear(self) -> None:
        """Remove all packages (for testing)."""
        with self._lock:
            self._packages.clear()
            self._versions.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export the entire registry as a dict."""
        with self._lock:
            return {name: meta.to_dict() for name, meta in self._packages.items()}

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_to_disk(self) -> None:
        """Persist to JSON file."""
        try:
            data = {
                "marketplace_version": MARKETPLACE_VERSION,
                "packages": {},
            }
            for name, versions in self._versions.items():
                data["packages"][name] = {
                    "meta": self._packages[name].to_dict(),
                    "versions": {v: pv.to_dict() for v, pv in versions.items()},
                }
            os.makedirs(os.path.dirname(self._storage_path) or ".", exist_ok=True)
            with open(self._storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass  # Don't crash on persistence failure

    def _load_from_disk(self) -> None:
        """Load from JSON file."""
        try:
            with open(self._storage_path) as f:
                data = json.load(f)
            for name, pkg_data in data.get("packages", {}).items():
                meta = PackageMeta.from_dict(pkg_data.get("meta", {}))
                self._packages[name] = meta
                self._versions[name] = {}
                for ver, ver_data in pkg_data.get("versions", {}).items():
                    self._versions[name][ver] = PackageVersion.from_dict(ver_data)
        except Exception:
            pass  # Don't crash on load failure

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _semver_key(version: str) -> Tuple[int, ...]:
        """Parse semver string into tuple for sorting."""
        try:
            parts = version.split(".")
            return tuple(int(p) for p in parts)
        except (ValueError, AttributeError):
            return (0, 0, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# PACKAGE MANAGER — the npm CLI equivalent
# ═══════════════════════════════════════════════════════════════════════════════


class PackageManager:
    """
    Manages local installation of marketplace packages.

    Analogous to npm or pip client commands.
    """

    def __init__(
        self,
        registry: MarketplaceRegistry,
        install_dir: str = "./.aos_packages",
    ):
        self._registry = registry
        self._install_dir = install_dir
        self._lock = threading.Lock()
        self._installed: Dict[str, InstalledPackage] = {}
        self._load_installed()

    # ── Install ───────────────────────────────────────────────────────────

    def install(
        self,
        package_name: str,
        *,
        version: str = "",
    ) -> Tuple[bool, str]:
        """
        Install a package from the registry.

        Returns (success, message).
        """
        # Validate the name
        name_errors = PackageValidator.validate_name(package_name)
        if name_errors:
            return False, "; ".join(name_errors)

        # Resolve version
        if version:
            pkg_version = self._registry.get_version(package_name, version)
            if not pkg_version:
                return (
                    False,
                    f"Version {version} of {package_name} not found in registry",
                )
        else:
            pkg_version = self._registry.get_latest_version(package_name)
            if not pkg_version:
                return False, f"Package {package_name} not found in registry"

        meta = pkg_version.meta

        # Check if already installed with same version
        with self._lock:
            existing = self._installed.get(package_name)
            if existing and existing.version == pkg_version.version:
                return True, f"{package_name}@{pkg_version.version} already installed"

        # Install dependencies first
        for dep in meta.dependencies:
            dep_ok, dep_msg = self.install(dep)
            if not dep_ok:
                return False, f"Failed to install dependency {dep}: {dep_msg}"

        # Run install function if provided
        capability_fn = None
        if pkg_version.install_fn:
            try:
                capability_fn = pkg_version.install_fn()
            except Exception as e:
                return False, f"Install function failed: {e}"

        # Create install record
        installed = InstalledPackage(
            name=package_name,
            version=pkg_version.version,
            meta=meta,
            install_path=os.path.join(self._install_dir, meta.short_name),
            checksum=pkg_version.checksum,
            capability_fn=capability_fn,
        )

        with self._lock:
            self._installed[package_name] = installed

        # Update download count in registry
        with self._registry._lock:
            reg_meta = self._registry._packages.get(package_name)
            if reg_meta:
                reg_meta.downloads += 1

        self._save_installed()

        return True, f"Installed {package_name}@{pkg_version.version}"

    # ── Uninstall ─────────────────────────────────────────────────────────

    def uninstall(self, package_name: str) -> Tuple[bool, str]:
        """Uninstall a package."""
        with self._lock:
            if package_name not in self._installed:
                return False, f"Package {package_name} is not installed"
            del self._installed[package_name]

        self._save_installed()
        return True, f"Uninstalled {package_name}"

    # ── Update ────────────────────────────────────────────────────────────

    def update(self, package_name: str) -> Tuple[bool, str]:
        """Update a package to the latest version."""
        with self._lock:
            if package_name not in self._installed:
                return False, f"Package {package_name} is not installed"
            current = self._installed[package_name]

        latest = self._registry.get_latest_version(package_name)
        if not latest:
            return False, f"Package {package_name} not found in registry"

        if latest.version == current.version:
            return True, f"{package_name}@{current.version} is already up to date"

        # Re-install with latest version
        return self.install(package_name, version=latest.version)

    # ── Query ─────────────────────────────────────────────────────────────

    def is_installed(self, package_name: str) -> bool:
        """Check if a package is installed."""
        with self._lock:
            return package_name in self._installed

    def get_installed(self, package_name: str) -> Optional[InstalledPackage]:
        """Get an installed package."""
        with self._lock:
            return self._installed.get(package_name)

    def list_installed(self) -> List[InstalledPackage]:
        """List all installed packages."""
        with self._lock:
            return list(self._installed.values())

    def installed_count(self) -> int:
        """Count of installed packages."""
        with self._lock:
            return len(self._installed)

    def outdated(self) -> List[Tuple[str, str, str]]:
        """
        List packages that have newer versions available.

        Returns list of (name, current_version, latest_version).
        """
        result: List[Tuple[str, str, str]] = []
        with self._lock:
            for name, pkg in self._installed.items():
                latest = self._registry.get_latest_version(name)
                if latest and latest.version != pkg.version:
                    result.append((name, pkg.version, latest.version))
        return result

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_installed(self) -> None:
        """Persist installed packages to disk."""
        try:
            os.makedirs(self._install_dir, exist_ok=True)
            manifest_path = os.path.join(self._install_dir, "manifest.json")
            data = {
                "marketplace_version": MARKETPLACE_VERSION,
                "installed": {
                    name: pkg.to_dict() for name, pkg in self._installed.items()
                },
            }
            with open(manifest_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass

    def _load_installed(self) -> None:
        """Load installed packages from disk."""
        manifest_path = os.path.join(self._install_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            return
        try:
            with open(manifest_path) as f:
                data = json.load(f)
            for name, pkg_data in data.get("installed", {}).items():
                self._installed[name] = InstalledPackage.from_dict(pkg_data)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# MARKETPLACE — high-level façade
# ═══════════════════════════════════════════════════════════════════════════════


class Marketplace:
    """
    High-level marketplace API — the single entry point.

    Usage::

        from infrarely.platform.marketplace import marketplace

        # Publish
        marketplace.publish(
            name="@community/web-researcher",
            version="1.0.0",
            description="Web research capability",
            category="research",
            tags=["web", "research", "scraping"],
            capabilities=["web_research"],
            install_fn=lambda: web_research_fn,
        )

        # Install
        ok, msg = marketplace.install("@community/web-researcher")

        # Use
        pkg = marketplace.get("@community/web-researcher")
        agent = infrarely.agent("research-bot", capabilities=[pkg.capability_fn])

        # Search
        results = marketplace.search("research")
        results = marketplace.search(category="coding")

        # CLI style
        marketplace.install("@community/code-reviewer")
        marketplace.install("@enterprise/salesforce-sync")
    """

    def __init__(
        self,
        registry: Optional[MarketplaceRegistry] = None,
        manager: Optional[PackageManager] = None,
        install_dir: str = "./.aos_packages",
    ):
        self._registry = registry or MarketplaceRegistry()
        self._manager = manager or PackageManager(self._registry, install_dir)
        self._event_listeners: Dict[str, List[Callable]] = {}

    # ── Publish ───────────────────────────────────────────────────────────

    def publish(
        self,
        name: str,
        *,
        version: str = "0.1.0",
        description: str = "",
        author: str = "",
        author_email: str = "",
        license: str = "MIT",
        category: str = "other",
        tags: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        homepage: str = "",
        repository: str = "",
        install_fn: Optional[Callable] = None,
        capabilities_code: str = "",
        **kwargs: Any,
    ) -> Tuple[bool, List[str]]:
        """
        Publish a capability package to the marketplace.

        Returns (success, errors).
        """
        meta = PackageMeta(
            name=name,
            version=version,
            description=description,
            author=author,
            author_email=author_email,
            license=license,
            category=category,
            tags=tags or [],
            capabilities=capabilities or [],
            dependencies=dependencies or [],
            homepage=homepage,
            repository=repository,
        )

        ok, errors = self._registry.publish(
            meta,
            capabilities_code=capabilities_code,
            install_fn=install_fn,
        )

        if ok:
            self._emit(
                "publish",
                {
                    "name": name,
                    "version": version,
                    "author": author,
                },
            )

        return ok, errors

    # ── Install / Uninstall ───────────────────────────────────────────────

    def install(
        self,
        package_name: str,
        *,
        version: str = "",
    ) -> Tuple[bool, str]:
        """Install a package. Returns (success, message)."""
        ok, msg = self._manager.install(package_name, version=version)
        if ok and "already installed" not in msg:
            self._emit(
                "install", {"name": package_name, "version": version or "latest"}
            )
        return ok, msg

    def uninstall(self, package_name: str) -> Tuple[bool, str]:
        """Uninstall a package."""
        ok, msg = self._manager.uninstall(package_name)
        if ok:
            self._emit("uninstall", {"name": package_name})
        return ok, msg

    def update(self, package_name: str) -> Tuple[bool, str]:
        """Update a package to latest version."""
        ok, msg = self._manager.update(package_name)
        if ok:
            self._emit("update", {"name": package_name})
        return ok, msg

    # ── Query ─────────────────────────────────────────────────────────────

    def get(self, package_name: str) -> Optional[InstalledPackage]:
        """Get an installed package."""
        return self._manager.get_installed(package_name)

    def info(self, package_name: str) -> Optional[PackageMeta]:
        """Get registry info about a package (installed or not)."""
        return self._registry.get(package_name)

    def search(
        self,
        query: str = "",
        *,
        category: str = "",
        scope: str = "",
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[PackageMeta]:
        """Search the marketplace."""
        return self._registry.search(
            query,
            category=category,
            scope=scope,
            tags=tags,
            limit=limit,
        )

    def list_installed(self) -> List[InstalledPackage]:
        """List all installed packages."""
        return self._manager.list_installed()

    def installed_count(self) -> int:
        """Count of installed packages."""
        return self._manager.installed_count()

    def is_installed(self, package_name: str) -> bool:
        """Check if a package is installed."""
        return self._manager.is_installed(package_name)

    def list_available(self) -> List[PackageMeta]:
        """List all available packages in the registry."""
        return self._registry.list_all()

    def available_count(self) -> int:
        """Count of packages in the registry."""
        return self._registry.count()

    def outdated(self) -> List[Tuple[str, str, str]]:
        """List packages that have newer versions available."""
        return self._manager.outdated()

    # ── Registry access ───────────────────────────────────────────────────

    @property
    def registry(self) -> MarketplaceRegistry:
        """Direct access to the underlying registry."""
        return self._registry

    @property
    def manager(self) -> PackageManager:
        """Direct access to the underlying package manager."""
        return self._manager

    # ── Events ────────────────────────────────────────────────────────────

    def on(self, event: str, callback: Callable) -> None:
        """Register an event listener (publish, install, uninstall, update)."""
        if event not in self._event_listeners:
            self._event_listeners[event] = []
        self._event_listeners[event].append(callback)

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Emit an event to listeners."""
        for cb in self._event_listeners.get(event, []):
            try:
                cb(data)
            except Exception:
                pass

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Registry + installation summary."""
        return {
            "marketplace_version": MARKETPLACE_VERSION,
            "available_packages": self._registry.count(),
            "installed_packages": self._manager.installed_count(),
            "outdated_packages": len(self._manager.outdated()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

_marketplace_instance: Optional[Marketplace] = None
_marketplace_lock = threading.Lock()


def get_marketplace() -> Marketplace:
    """Return the global Marketplace singleton."""
    global _marketplace_instance
    with _marketplace_lock:
        if _marketplace_instance is None:
            _marketplace_instance = Marketplace()
        return _marketplace_instance


def _reset_marketplace() -> None:
    """Reset singleton (for testing)."""
    global _marketplace_instance
    with _marketplace_lock:
        if _marketplace_instance is not None:
            _marketplace_instance._registry.clear()
            _marketplace_instance._manager._installed.clear()
        _marketplace_instance = None
    # Clean default install dir to prevent stale manifest leaking between runs
    default_manifest = os.path.join("./.aos_packages", "manifest.json")
    if os.path.isfile(default_manifest):
        try:
            os.remove(default_manifest)
        except OSError:
            pass
