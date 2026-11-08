from pathlib import Path

import pandas as pd
import pytest

from ai_os.modules.knowledge.adapters.outbound.price_sheet_parser import PriceSheetParser
from ai_os.modules.knowledge.domain.errors import InvalidPriceSheet
from ai_os.modules.knowledge.domain.models import DocumentKind, SourceDocument

DOCUMENT = SourceDocument("prices", "Palm Hills", DocumentKind.PRICE_SHEET, 1)


def write_sheets(path: Path, **sheets: pd.DataFrame) -> Path:
    with pd.ExcelWriter(path) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return path


async def test_each_row_becomes_atomic_section_with_unit_and_row_number(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"Unit Code": ["A-101", "A-102"], "Price": [5000000, 6500000], "Bedrooms": [3, None]}
    )
    path = write_sheets(tmp_path / "prices.xlsx", Phase1=frame)

    sections = await PriceSheetParser().parse(path, DOCUMENT)

    assert sections[0].text == "Sheet: Phase1 | Unit: A-101 | Price: 5,000,000 EGP | Bedrooms: 3"
    assert sections[0].unit_codes == ("A-101",)
    assert sections[0].atomic
    assert sections[0].page == 2
    assert "Bedrooms" not in sections[1].text


async def test_accepts_arabic_headers(tmp_path: Path) -> None:
    frame = pd.DataFrame({"كود الوحدة": ["ب-٢٠"], "السعر": [4200000]})
    path = write_sheets(tmp_path / "prices.xlsx", Sheet1=frame)

    sections = await PriceSheetParser().parse(path, DOCUMENT)

    assert sections[0].unit_codes == ("ب-٢٠",)


async def test_bad_rows_reject_whole_file_and_name_each_problem(tmp_path: Path) -> None:
    frame = pd.DataFrame({"Unit": ["A-1", "A-1", "A-3"], "Price": [100, 200, -5]})
    path = write_sheets(tmp_path / "prices.xlsx", Phase1=frame)

    with pytest.raises(InvalidPriceSheet) as error:
        await PriceSheetParser().parse(path, DOCUMENT)

    text = str(error.value)
    assert "Phase1" in text
    assert "unit_code" in text
    assert "price" in text
    assert "row 4" in text


async def test_notes_tab_is_skipped_but_partial_tab_fails(tmp_path: Path) -> None:
    prices = pd.DataFrame({"Unit": ["A-1"], "Price": [100]})
    notes = pd.DataFrame({"Note": ["Prices valid until March"]})
    ok = write_sheets(tmp_path / "ok.xlsx", Prices=prices, Notes=notes)
    assert len(await PriceSheetParser().parse(ok, DOCUMENT)) == 1

    partial = pd.DataFrame({"Unit": ["B-1"]})
    bad = write_sheets(tmp_path / "bad.xlsx", Prices=prices, Phase2=partial)
    with pytest.raises(InvalidPriceSheet, match="Phase2: missing columns"):
        await PriceSheetParser().parse(bad, DOCUMENT)


async def test_workbook_without_price_columns_fails(tmp_path: Path) -> None:
    path = write_sheets(tmp_path / "x.xlsx", Notes=pd.DataFrame({"Note": ["hello"]}))

    with pytest.raises(InvalidPriceSheet, match="no sheet"):
        await PriceSheetParser().parse(path, DOCUMENT)
