import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from ai_os.modules.inventory.domain.errors import UnsafeSql

_DIALECT = "postgres"
_TABLE = "units"
_COLUMNS = frozenset(
    {
        "code",
        "project",
        "unit_type",
        "bedrooms",
        "area_sqm",
        "price_egp",
        "status",
        "delivery_date",
    }
)
_MAX_ROWS = 50
# Anything else is rejected, which blocks pg_sleep, pg_read_file, and similar.
_ALLOWED_FUNCTIONS = (
    exp.Count,
    exp.Min,
    exp.Max,
    exp.Avg,
    exp.Sum,
    exp.Round,
    exp.Lower,
    exp.Upper,
    exp.Coalesce,
    exp.Cast,
    exp.If,
    exp.Case,
    exp.Extract,
    exp.CurrentDate,
    # sqlglot models boolean connectors as functions.
    exp.And,
    exp.Or,
)
# Blocks catalog casts such as ::regclass.
_ALLOWED_CASTS = (
    exp.DataType.Type.TEXT,
    exp.DataType.Type.VARCHAR,
    exp.DataType.Type.INT,
    exp.DataType.Type.BIGINT,
    exp.DataType.Type.DECIMAL,
    exp.DataType.Type.FLOAT,
    exp.DataType.Type.DOUBLE,
    exp.DataType.Type.DATE,
)
_FORBIDDEN_NODES = (exp.With, exp.Union, exp.Except, exp.Intersect, exp.Lock, exp.Into, exp.Lateral)


class SqlGlotGuard:
    def __init__(self, max_rows: int = _MAX_ROWS) -> None:
        self._max_rows = max_rows

    def validate(self, sql: str) -> str:
        select = self._parse_single_select(sql)
        self._check_nodes(select)
        self._check_tables(select)
        self._check_columns(select)
        self._check_functions(select)
        self._cap_limit(select)
        # Regenerating drops comments, so only what was validated reaches the database.
        return select.sql(dialect=_DIALECT, comments=False)

    def _parse_single_select(self, sql: str) -> exp.Select:
        try:
            statements = [s for s in sqlglot.parse(sql, dialect=_DIALECT) if s is not None]
        except SqlglotError as error:
            raise UnsafeSql("SQL could not be parsed") from error
        if len(statements) != 1:
            raise UnsafeSql("exactly one statement is allowed")
        statement = statements[0]
        if not isinstance(statement, exp.Select):
            raise UnsafeSql("only SELECT is allowed")
        return statement

    def _check_nodes(self, select: exp.Select) -> None:
        for node in select.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                raise UnsafeSql(f"{type(node).__name__.upper()} is not allowed")
            if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count):
                raise UnsafeSql("name columns explicitly instead of *")
        if select.args.get("locks"):
            raise UnsafeSql("locking clauses are not allowed")
        if select.args.get("into"):
            raise UnsafeSql("SELECT INTO is not allowed")

    def _check_tables(self, select: exp.Select) -> None:
        tables = list(select.find_all(exp.Table))
        if not tables:
            raise UnsafeSql(f"query must read from {_TABLE}")
        for table in tables:
            if table.name != _TABLE or table.db or table.catalog:
                raise UnsafeSql(f"only the {_TABLE} table is allowed")

    def _check_columns(self, select: exp.Select) -> None:
        aliases = {node.alias for node in select.find_all(exp.Alias)}
        for column in select.find_all(exp.Column):
            if column.name in _COLUMNS:
                continue
            # Only ORDER BY, GROUP BY and HAVING may refer to a select-list alias. Elsewhere an
            # alias named ctid would otherwise whitelist the real system column.
            in_alias_scope = column.find_ancestor(exp.Order, exp.Group, exp.Having) is not None
            if column.name in aliases and in_alias_scope and not column.table:
                continue
            raise UnsafeSql(f"column {column.name} is not allowed")

    def _check_functions(self, select: exp.Select) -> None:
        for function in select.find_all(exp.Func):
            if not isinstance(function, _ALLOWED_FUNCTIONS):
                raise UnsafeSql(f"function {function.sql(dialect=_DIALECT)} is not allowed")
            if isinstance(function, exp.Cast) and not function.to.is_type(*_ALLOWED_CASTS):
                raise UnsafeSql(f"cast to {function.to.sql(dialect=_DIALECT)} is not allowed")

    def _cap_limit(self, select: exp.Select) -> None:
        limit = select.args.get("limit")
        if limit is None:
            select.limit(self._max_rows, copy=False)
            return
        value = limit.expression
        if not (isinstance(value, exp.Literal) and value.is_int):
            raise UnsafeSql("LIMIT must be a whole number")
        if int(value.this) > self._max_rows:
            select.limit(self._max_rows, copy=False)
