"""Copy data/metadata into a value workspace; original videos remain read-only links."""

from pathlib import Path
import json
import shutil
import tempfile

from omegaconf import OmegaConf
from submodules.contracts import resolve_path, sha256


def source_identity(root):
    root = Path(root).resolve()
    result = {
        str(p.relative_to(root)): sha256(p)
        for folder in ("data", "meta")
        for p in sorted((root / folder).rglob("*"))
        if p.is_file()
    }
    for p in sorted((root / "videos").rglob("*.mp4")):
        stat = p.stat()
        result[str(p.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
    return result


def prepare_workspace(config_path, source_dirs, work_root, *, analyze=False):
    config = OmegaConf.to_container(OmegaConf.load(resolve_path(config_path)), resolve=True)
    work = Path(work_root).resolve()
    sources = [Path(p).resolve() for p in source_dirs]
    if not sources or len({p.name for p in sources}) != len(sources):
        raise ValueError("Provide source dataset directories with distinct names")
    known = {entry["name"]: entry for entry in config["datasets"]}
    specs = []
    for source in sources:
        if work == source or work in source.parents or source in work.parents:
            raise ValueError("Workspace and source dataset paths must not contain one another")
        if not (source / "meta/info.json").is_file() or not (source / "data").is_dir():
            raise FileNotFoundError(f"Not a complete source dataset: {source}")
        spec = dict(known.get(source.name, dict(outcome_override=None, joint_indices=None, expected_joint_dim=None)))
        spec.update(name=source.name, source_path=str(source))
        specs.append(spec)
    config["datasets"] = specs
    config["data_dir"] = str(work / "raw")
    config["output_dir"] = str(work / "splits")
    if analyze:
        # Read-only audit may inspect arbitrary source paths without writing effective config.
        from submodules.data_workflow import audit_dataset

        for spec, source in zip(specs, sources):
            audit_dataset({**config, "data_dir": str(source.parent)}, spec)
        return None
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "workspace_manifest.json"
    identity = {str(p): source_identity(p) for p in sources}
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old is not None:
        if old["sources"] != identity:
            raise ValueError("Source data changed; select a new workspace instead of reusing stale splits")
    else:
        target = work / "raw"
        if target.exists():
            raise FileExistsError(f"Untracked workspace data already exists: {target}")
        temporary = Path(tempfile.mkdtemp(prefix=".raw-", dir=work))
        try:
            for source in sources:
                output = temporary / source.name
                shutil.copytree(source / "data", output / "data")
                shutil.copytree(source / "meta", output / "meta")
                if not (source / "videos").is_dir():
                    raise FileNotFoundError(source / "videos")
                (output / "videos").symlink_to(source / "videos", target_is_directory=True)
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        manifest_path.write_text(json.dumps(dict(status="complete", sources=identity), indent=2))
    effective = work / "adaptation.yaml"
    content = OmegaConf.to_yaml(OmegaConf.create(config))
    if effective.exists() and effective.read_text() != content:
        raise ValueError("Workspace preparation config changed; use a new work directory")
    effective.write_text(content)
    return effective
