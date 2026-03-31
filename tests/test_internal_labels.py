import app as app_module


def test_flip_regression_domain_key_simple() -> None:
    assert app_module._flip_regression_domain_key("reuters.com") == "com.reuters"


def test_flip_regression_domain_key_multilevel_suffix() -> None:
    # Uses PSL-aware logic when tldextract is available.
    assert app_module._flip_regression_domain_key("theregister.co.uk") == "co.uk.theregister"


def test_load_binary_label_lookup_skips_empty_values(tmp_path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "domain,bin,reg\n"
        "Example.com,1,\n"
        "foo.org,0,\n"
        "bar.net,,\n",
        encoding="utf-8",
    )

    labels = app_module._load_binary_label_lookup(csv_path)

    assert labels == {
        "example.com": True,
        "foo.org": False,
    }


def test_load_binary_label_lookup_skips_invalid_domains(tmp_path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "domain,bin,reg\n"
        "1.179.170.7,0,\n"
        "valid.org,1,\n",
        encoding="utf-8",
    )

    labels = app_module._load_binary_label_lookup(csv_path)

    assert labels == {
        "valid.org": True,
    }


def test_load_regression_label_lookup_reads_dqr_schema(tmp_path) -> None:
    csv_path = tmp_path / "domain_pc1.csv"
    csv_path.write_text(
        "domain,pc1\n"
        "Reuters.com,1\n"
        "apnews.com,0.98\n",
        encoding="utf-8",
    )

    labels = app_module._load_regression_label_lookup(csv_path)

    assert labels == {
        "reuters.com": 1.0,
        "apnews.com": 0.98,
    }


def test_build_prediction_result_uses_predictions_only(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "binary_label_lookup", {"example.com": False})
    monkeypatch.setattr(app_module, "regression_label_lookup", {"example.com": 0.91})

    result = app_module._build_prediction_result(
        "example.com",
        0.12,
        True,
    )

    assert result == {
        "domain": "example.com",
        "credibility_level": 0.12,
        "credible": True,
    }


def test_build_ground_truth_result_uses_gt_labels(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "binary_label_lookup", {})
    monkeypatch.setattr(app_module, "regression_label_lookup", {"example.com": 0.77})

    result = app_module._build_ground_truth_result("example.com")

    assert result == {
        "domain": "example.com",
        "credibility_level": 0.77,
    }


def test_build_ground_truth_result_uses_both_gt_scores(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "binary_label_lookup", {"example.com": True})
    monkeypatch.setattr(app_module, "regression_label_lookup", {"example.com": 0.66})

    result = app_module._build_ground_truth_result("example.com")

    assert list(result.keys()) == [
        "domain",
        "credibility_level",
        "credible",
    ]