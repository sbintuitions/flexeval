"""Generate reference pages for preset configs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mkdocs_gen_files

config_root = Path(__file__).parent.parent / "flexeval" / "preset_configs"


config_path_list: list[str] = []
for config_file in sorted(config_root.rglob("*.jsonnet")):
    config_path = config_file.relative_to(config_root)
    config_path_list.append(str(config_path))


def _path_list_to_nested_dict(path_list: list[str]) -> dict[str, Any]:
    nested_dict: dict[str, Any] = {}
    for path in path_list:
        parts = path.split("/")
        current = nested_dict
        for part in parts[:-2]:
            if part not in current:
                current[part] = {}
            current = current[part]
        key, value = parts[-2:]
        if key not in current:
            current[key] = []
        current[key].append(value)
    return nested_dict


all_pages = _path_list_to_nested_dict(config_path_list)


def _nested_dict_to_markdown(d: dict[str, Any], level: int = 0, current_path: str = ".") -> str:
    def _get_level_prefix(level: int) -> str:
        if level == 0:
            return "## "
        if level == 1:
            return "### "
        return "    " * (level - 2) + "* "

    markdown = ""
    for key, value in d.items():
        markdown += _get_level_prefix(level) + str(key) + "\n"
        if isinstance(value, dict):
            markdown += _nested_dict_to_markdown(value, level + 1, current_path + "/" + key)
        elif isinstance(value, list):
            for file_path in value:
                item = file_path.replace(".jsonnet", "")
                item_path = "/".join([current_path, key, f"index.md#{item}"])
                markdown += _get_level_prefix(max(level + 1, 2)) + f"[{item}]({item_path})" + "\n"
    return markdown


with mkdocs_gen_files.open(Path("preset_configs") / "index.md", "w") as fd:
    fd.write("# Preset Configs\n")
    fd.write(
        "You can check the config using the following command:\n"
        "```bash\n"
        "flexeval_presets <config_name>\n"
        "```\n",
    )
    fd.write(_nested_dict_to_markdown(all_pages))


def _jsonnet_to_markdown(jsonnet_path: Path) -> str:
    with open(jsonnet_path) as f:
        config = f.read()

    # extract multiline comments at the beginning of the file "/* ... */"
    comment = re.search(r"/\*.*?\*/", config, re.DOTALL)
    if comment:
        comment = comment.group(0)
        config = config.replace(comment, "").strip()
        comment = comment.replace("/*", "").replace("*/", "").strip()
    else:
        comment = ""

    markdown = f"## {jsonnet_path.stem}\n"
    markdown += f"{comment}\n"
    markdown += f"```\n{config}\n```"
    return markdown


def _recursive_write_pages(pages_dict: dict[str, dict | list[str]], current_path: str = "./") -> None:
    for key, value in pages_dict.items():
        if isinstance(value, list):
            with mkdocs_gen_files.open(Path("preset_configs") / f"{current_path}{key}/index.md", "a") as fd:
                for config_file in value:
                    config_path = config_root / Path(f"{current_path}{key}/index.md").parent / config_file
                    markdown = _jsonnet_to_markdown(config_path)
                    fd.write(markdown)
                    fd.write("\n")
        else:
            with mkdocs_gen_files.open(Path("preset_configs") / f"{current_path}{key}/index.md", "w") as fd:
                fd.write(f"# {key}\n")
                fd.write(_nested_dict_to_markdown(value))

            _recursive_write_pages(value, current_path=current_path + f"{key}/")


_recursive_write_pages(all_pages)
