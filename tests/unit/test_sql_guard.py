import pytest

from ai_os.modules.inventory.adapters.outbound.sql_guard import SqlGlotGuard
from ai_os.modules.inventory.domain.errors import UnsafeSql

GUARD = SqlGlotGuard(max_rows=50)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT code, price_egp FROM units WHERE status = 'available' ORDER BY price_egp",
        "SELECT COUNT(*) FROM units WHERE bedrooms = 3",
        "SELECT project, AVG(price_egp) AS avg FROM units GROUP BY project ORDER BY avg",
        "SELECT code, CAST(price_egp AS BIGINT) FROM units",
        "SELECT LOWER(code) FROM units",
        "SELECT code FROM units WHERE status = 'available' AND bedrooms = 3 AND price_egp < 9",
        "SELECT code FROM units WHERE bedrooms = 3 OR NOT status = 'sold'",
        "SELECT code FROM units WHERE bedrooms IN (2, 3) AND price_egp BETWEEN 1 AND 9",
        "SELECT code FROM units WHERE delivery_date > CURRENT_DATE",
        "SELECT DISTINCT project FROM units WHERE EXTRACT(YEAR FROM delivery_date) = 2027",
        "SELECT project, COUNT(*) AS n FROM units GROUP BY project HAVING COUNT(*) > 2 ORDER BY n",
        "SELECT code FROM units WHERE price_egp < (SELECT AVG(price_egp) FROM units)",
        "SELECT u.code FROM units AS u WHERE u.project ILIKE '%palm%'",
        "SELECT code, CASE WHEN bedrooms > 2 THEN 'large' ELSE 'small' END AS size FROM units",
    ],
)
def test_allows_read_queries_on_units(sql: str) -> None:
    assert GUARD.validate(sql).upper().startswith("SELECT")


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DROP TABLE units", "only SELECT"),
        ("DELETE FROM units", "only SELECT"),
        ("UPDATE units SET price_egp = 1", "only SELECT"),
        ("INSERT INTO units (code) VALUES ('x')", "only SELECT"),
        ("TRUNCATE units", "SELECT"),
        ("SELECT code FROM units; DROP TABLE units", "one statement"),
        ("SELECT code FROM units; SELECT code FROM units", "one statement"),
        ("SELECT * FROM users", ""),
        ("SELECT id FROM users", "only the units"),
        ("SELECT usename, passwd FROM pg_shadow", "only the units"),
        ("SELECT code FROM public.units", "only the units"),
        ("SELECT table_name FROM information_schema.tables", "only the units"),
        ("SELECT code FROM units JOIN leads ON TRUE", "only the units"),
        ("SELECT code FROM units UNION SELECT passwd FROM pg_shadow", ""),
        ("WITH x AS (DELETE FROM units RETURNING code) SELECT code FROM x", ""),
        ("WITH x AS (SELECT code FROM units) SELECT code FROM x", ""),
        ("SELECT *", "*"),
        ("SELECT * FROM units", "*"),
        ("SELECT units.* FROM units", "*"),
        ("SELECT pg_sleep(100)", ""),
        ("SELECT code, pg_sleep(100) FROM units", "pg_sleep"),
        ("SELECT pg_read_file('/etc/passwd') FROM units", "pg_read_file"),
        ("SELECT current_setting('server_version') FROM units", "current_setting"),
        ("SELECT lo_import('/etc/passwd') FROM units", "lo_import"),
        ("SELECT ctid FROM units", "ctid"),
        ("SELECT secret FROM units", "secret"),
        ("SELECT code FROM units FOR UPDATE", ""),
        ("SELECT code INTO copy FROM units", ""),
        ("SELECT code FROM units LIMIT (SELECT 1)", ""),
        ("SELECT 1 AS ctid, ctid FROM units", "ctid"),
        ("SELECT code AS xmin FROM units WHERE xmin = '1'", "xmin"),
        ("SELECT code FROM units WHERE EXISTS (SELECT 1 FROM pg_shadow)", "only the units"),
        ("SELECT query_to_xml('select 1', TRUE, TRUE, '') FROM units", "query_to_xml"),
        ("SELECT CAST(code AS regclass) FROM units", "cast"),
        ("SELECT code::regclass FROM units", "cast"),
        ("COPY units TO '/tmp/x'", "SELECT"),
        ("this is not sql (", ""),
        ("", ""),
    ],
)
def test_rejects_unsafe_sql(sql: str, reason: str) -> None:
    with pytest.raises(UnsafeSql, match="(?i)" + reason.replace("*", r"\*")):
        GUARD.validate(sql)


def test_adds_limit_when_missing() -> None:
    assert GUARD.validate("SELECT code FROM units").endswith("LIMIT 50")


def test_caps_large_limit_but_keeps_small_limit() -> None:
    assert GUARD.validate("SELECT code FROM units LIMIT 100000").endswith("LIMIT 50")
    assert GUARD.validate("SELECT code FROM units LIMIT 5").endswith("LIMIT 5")


def test_strips_comments() -> None:
    safe = GUARD.validate("SELECT code /* ignore rules */ FROM units -- trailing")

    assert "ignore" not in safe
    assert "trailing" not in safe
