from __future__ import annotations
import pandas as pd
from typing import List

# This is a wrapper class to allow preprocessing data in different ways.
# To add a method to this class, just make sure it returns itself at the end.
# See remove_columns for an example


class Preprocess:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.df: pd.DataFrame

        self.load_file()

    def load_file(self) -> None:
        self.df = pd.read_csv(self.filepath)

    # Remove columns from the dataset
    # By default this just removes the CLIENTNUM column.
    def remove_columns(self, columns: List[str] = ["CLIENTNUM"]) -> Preprocess:
        # Validate columns type
        if type(columns) != list:
            raise TypeError(
                f"Columns must be a list, but it is of type {type(columns)} instead.")

        # Check that all columns exist in the DataFrame
        if not set(columns).issubset(self.df.columns):
            no_exist = [c for c in columns if c not in self.df.columns]
            raise LookupError(
                f"One or more columns were not found in the DataFrame: {no_exist}")

        self.df.drop(columns, axis=1, inplace=True)
        return self

    # Remove highly correlated features from the dataset
    def remove_high_correlation(self, threshold):
        # get sorted correlation pairs
        c = self.df.corr().abs()
        s = c.unstack()
        so = s.sort_values(kind="quicksort")

        # remove from dataset if above threshold
        last_label1 = ""
        last_label2 = ""
        cols_to_remove = []
        for label, corr_value in zip(so.axes[0], so):
            if (label[0] == last_label2 and label[1] == last_label1):  # if pair is equal
                continue
            last_label1 = label[0]
            last_label2 = label[1]
            if (corr_value != 1 and corr_value > threshold):
                if (not label[0] in cols_to_remove):
                    cols_to_remove.insert(len(cols_to_remove), label[0])
                row = (label[0], label[1], str(corr_value))
                print("{: >30} {: >30} {: >30}".format(*row))  # print label
        self.remove_columns(cols_to_remove)
        return self

    # Remove features with low correlation towards the output
    def remove_low_impact():
        return self


if __name__ == "__main__":
    # print(Preprocess("data/full.csv").remove_columns().df)
    print(Preprocess("data/full.csv").remove_high_correlation(.7).df.columns)
