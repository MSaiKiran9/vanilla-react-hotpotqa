"""Excel report writer."""

from __future__ import annotations

import pandas as pd

from config import OUTPUT_EXCEL


def save_evaluation(results: list[dict]) -> None:
    """Save all model summaries to one workbook sheet."""

    df = pd.DataFrame(results)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Model Comparison", index=False)

        worksheet = writer.sheets["Model Comparison"]
        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value or "")) for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = (
                max_length + 2
            )
