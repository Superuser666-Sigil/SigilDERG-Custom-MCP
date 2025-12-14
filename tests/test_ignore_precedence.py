import sigil_mcp.ignore_utils as ignore_utils
from sigil_mcp.config import Config


def test_repositories_config_includes_ignore_patterns(tmp_path):
    cfg_data = {
        "repositories": {
            "projA": {
                "path": str(tmp_path / "projA"),
                "respect_gitignore": True,
                "ignore_patterns": ["target/", "!keep/"],
            },
            "projB": str(tmp_path / "projB")
        }
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("""{""" + "\n" + "\n".join([]) + """}""")
    # write a combined JSON with repositories only
    import json
    cfg_file.write_text(json.dumps(cfg_data))

    cfg = Config(cfg_file)
    repos = cfg.repositories_config
    assert "projA" in repos
    assert repos["projA"]["path"] == str(tmp_path / "projA")
    assert repos["projA"]["respect_gitignore"] is True
    assert isinstance(repos["projA"]["ignore_patterns"], list)
    assert "target/" in repos["projA"]["ignore_patterns"]

    assert "projB" in repos
    assert repos["projB"]["ignore_patterns"] == []


def test_per_repo_allow_overrides_global_ignore(tmp_path):
    repo = tmp_path / "repo"
    (repo / "node_modules" / "a.js").mkdir(parents=True, exist_ok=True)
    p = (repo / "node_modules" / "a.js")

    # Global ignores node_modules/, repo explicitly allows it via !node_modules/
    assert not ignore_utils.should_ignore(
        p,
        repo,
        config_ignore_patterns=["node_modules/"],
        repo_ignore_patterns=["!node_modules/"],
    )
    from sigil_mcp.config import Config
    from sigil_mcp.ignore_utils import should_ignore


    def test_repositories_config_includes_ignore_patterns(tmp_path):
        cfg_data = {
            "repositories": {
                "projA": {
                    "path": str(tmp_path / "projA"),
                    "respect_gitignore": True,
                    "ignore_patterns": ["target/", "!keep/"],
                },
                "projB": str(tmp_path / "projB")
            }
        }
        cfg_file = tmp_path / "config.json"
        # write a combined JSON with repositories only
        import json
        cfg_file.write_text(json.dumps(cfg_data))

        cfg = Config(cfg_file)
        repos = cfg.repositories_config
        assert "projA" in repos
        assert repos["projA"]["path"] == str(tmp_path / "projA")
        assert repos["projA"]["respect_gitignore"] is True
        assert isinstance(repos["projA"]["ignore_patterns"], list)
        assert "target/" in repos["projA"]["ignore_patterns"]

        assert "projB" in repos
        assert repos["projB"]["ignore_patterns"] == []


    def test_per_repo_allow_overrides_global_ignore(tmp_path):
        repo = tmp_path / "repo"
        (repo / "node_modules" / "a.js").mkdir(parents=True, exist_ok=True)
        p = (repo / "node_modules" / "a.js")

        # Global ignores node_modules/, repo explicitly allows it via !node_modules/
        do_ignore = should_ignore(
            p,
            repo,
            config_ignore_patterns=["node_modules/"],
            repo_ignore_patterns=["!node_modules/"],
        )
        assert do_ignore is False


    def test_per_repo_ignore_overrides_global_allow(tmp_path):
        repo = tmp_path / "repo2"
        (repo / "target" / "x").mkdir(parents=True, exist_ok=True)
        p = (repo / "target" / "x")

        # Global allows target via !target/, but repo explicitly sets target/ to ignore
        do_ignore = ignore_utils.should_ignore(
            p,
            repo,
            config_ignore_patterns=["!target/"],
            repo_ignore_patterns=["target/"],
        )
        assert do_ignore is True
