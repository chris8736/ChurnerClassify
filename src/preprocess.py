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
            raise TypeError(f"Columns must be a list, but it is of type {type(columns)} instead.")

        # Check that all columns exist in the DataFrame
        if not set(columns).issubset(self.df.columns):
            no_exist = [c for c in columns if c not in self.df.columns]
            raise LookupError(f"One or more columns were not found in the DataFrame: {no_exist}")
        
        self.df.drop(columns, axis = 1, inplace = True)
        return self

if __name__ == "__main__":
    print(Preprocess("data/full.csv").remove_columns().df)