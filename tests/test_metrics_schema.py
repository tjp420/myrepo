import json
from pathlib import Path

from jsonschema import Draft7Validator, FormatChecker


def test_metrics_example_matches_schema():
    schema = json.loads(Path("docs/json_schema.json").read_text(encoding="utf-8"))
    data = json.loads(
        Path("docs/json_metrics_example.json").read_text(encoding="utf-8")
    )
    validator = Draft7Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    assert not errors, "Schema validation errors: " + "; ".join(
        [f"{e.message} at {list(e.path)}" for e in errors]
    )
