import asyncio
from pathlib import Path

import pandas as pd
import pandera.errors
import pandera.pandas as pa

from ai_os.modules.knowledge.domain.errors import InvalidPriceSheet
from ai_os.modules.knowledge.domain.models import ParsedSection, SourceDocument

_ALIASES = {
    "unit_code": {
        "unit_code",
        "unit code",
        "unit",
        "unit no",
        "unit number",
        "كود الوحدة",
        "الوحدة",
    },
    "price": {"price", "price egp", "total price", "السعر", "سعر الوحدة"},
    "area_sqm": {"area_sqm", "area", "bua", "المساحة"},
    "bedrooms": {"bedrooms", "beds", "الغرف", "عدد الغرف"},
    "delivery": {"delivery", "delivery date", "التسليم", "موعد التسليم"},
}
_REQUIRED = {"unit_code", "price"}

_SCHEMA = pa.DataFrameSchema(
    {
        "unit_code": pa.Column(str, pa.Check.str_length(min_value=1), unique=True),
        "price": pa.Column(float, pa.Check.gt(0), coerce=True),
    }
)


class PriceSheetParser:
    """Each row becomes one atomic section, so a price never separates from its unit.

    The citation page holds the spreadsheet row number.
    """

    async def parse(self, path: Path, document: SourceDocument) -> list[ParsedSection]:
        return await asyncio.to_thread(self._parse, path)

    def _parse(self, path: Path) -> list[ParsedSection]:
        sheets = pd.read_excel(path, sheet_name=None, dtype=object)
        sections: list[ParsedSection] = []
        problems: list[str] = []
        found_price_sheet = False
        for sheet_name, raw in sheets.items():
            frame = raw.rename(columns=_canonical_column).dropna(how="all")
            present = _REQUIRED & set(frame.columns)
            if not present:
                continue  # Notes or legend tab.
            found_price_sheet = True
            missing = _REQUIRED - present
            if missing:
                problems.append(f"{sheet_name}: missing columns {sorted(missing)}")
                continue
            frame["unit_code"] = frame["unit_code"].map(_clean_code)
            try:
                _SCHEMA.validate(frame, lazy=True)
            except pandera.errors.SchemaErrors as error:
                problems.extend(_describe(sheet_name, error))
                continue
            sections.extend(_row_section(sheet_name, index, row) for index, row in frame.iterrows())
        if not found_price_sheet:
            problems.append("no sheet has unit and price columns")
        if problems:
            raise InvalidPriceSheet(problems)
        return sections


def _canonical_column(column: object) -> str:
    name = str(column).strip().casefold()
    for canonical, aliases in _ALIASES.items():
        if name in aliases:
            return canonical
    return name


def _clean_code(value: object) -> object:
    return value if pd.isna(value) else str(value).strip()


def _describe(sheet_name: str, error: pandera.errors.SchemaErrors) -> list[str]:
    cases = error.failure_cases
    return [
        f"{sheet_name}: row {_row_number(row['index'])} {row['column']} failed {row['check']}"
        f" (value {row['failure_case']!r})"
        for _, row in cases.iterrows()
    ]


def _row_number(index: object) -> object:
    return index if pd.isna(index) else int(index) + 2


def _row_section(sheet_name: str, index: int, row: pd.Series) -> ParsedSection:
    fields = [
        f"Sheet: {sheet_name}",
        f"Unit: {row['unit_code']}",
        f"Price: {float(row['price']):,.0f} EGP",
    ]
    for label, column in (
        ("Area", "area_sqm"),
        ("Bedrooms", "bedrooms"),
        ("Delivery", "delivery"),
    ):
        if column in row and pd.notna(row[column]):
            fields.append(f"{label}: {row[column]}")
    return ParsedSection(
        text=" | ".join(fields),
        # +2: pandas index is 0-based and the header occupies row 1.
        page=int(index) + 2,
        unit_codes=(str(row["unit_code"]),),
        atomic=True,
    )
