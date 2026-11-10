"""Plain-assert verification for the pipeline check."""

from pipeline import run_pipeline


def test_pipeline_produces_eager_and_compiled_rows():
    rows = run_pipeline(chains=["simple"], write=False)

    assert len(rows) >= 4, len(rows)
    assert any(row["config"] == "eager" and row["status"] == "ok" for row in rows)
    assert any(row["config"] != "eager" for row in rows)
    assert all("cell_class" in row for row in rows)


if __name__ == "__main__":
    test_pipeline_produces_eager_and_compiled_rows()
    print("test_pipeline_produces_eager_and_compiled_rows PASS")
