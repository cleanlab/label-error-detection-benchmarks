import pandas as pd
from pathlib import Path


def get_label_errors(
    annotation_path: str = "annotations/andrew-ng-dcai-comp-2021-manual-review-for-label-errors.xlsx",
):
    """Get list of image file names with label error (based on manual review)"""

    df_annotation = pd.read_excel(annotation_path)

    label_errors = (
        df_annotation[df_annotation.label_error]
        .file_path.map(lambda x: Path(x).name)
        .tolist()
    )

    return label_errors
