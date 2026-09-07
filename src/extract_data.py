# %%
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import fastf1
import pandas as pd

fastf1.set_log_level(level=100)

PATH_RAW = os.environ["PATH_RAW"]
CURRENT_YEAR = datetime.now(UTC).year


class ExtractData:
    def __init__(
        self,
        years: list[int] | None = None,
        reload_data: bool = False,
        identifiers: list[str] | None = None,
        base_data: str = "results",
    ) -> None:
        self.years = list(years) if years is not None else [CURRENT_YEAR]
        if reload_data:
            self.years = [i for i in range(1980, CURRENT_YEAR)]

        self.identifiers = list(identifiers) if identifiers is not None else ["R", "S"]
        self.reload_data = reload_data
        self.path_save_data = f"{PATH_RAW}/{base_data}"

        os.makedirs(self.path_save_data, exist_ok=True)

    def get_data(self, year: int, gp: int, identifier: str) -> pd.DataFrame:
        try:
            session = fastf1.get_session(year, gp, identifier)
        except ValueError:
            empty = pd.DataFrame()
            return empty

        session.load()
        df = self.prepare_data(session)

        return df

    def prepare_data(self, session) -> pd.DataFrame:
        df = session.results
        df["Year"] = session.date.year
        df["Date"] = session.date
        df["Mode"] = session.name
        df["RoundNumber"] = session.event["RoundNumber"]
        df["OfficialEventName"] = session.event["OfficialEventName"]
        df["EventName"] = session.event["EventName"]
        df["Country"] = session.event["Country"]
        df["Location"] = session.event["Location"]

        df["Date"] = df["Date"].astype("datetime64[us]")
        columns_names = ["Q1", "Q2", "Q3", "Time"]
        for i in columns_names:
            df[i] = df[i].dt.total_seconds()
        return df

    def process_data(self, year: int, gp: int, identifier: str) -> bool:
        filename = f"{self.path_save_data}/{year}_{(gp):02}_{identifier}.parquet"
        if Path(filename).is_file() and not self.reload_data:
            return False

        df = self.get_data(year, gp, identifier)
        if df.empty:
            return False

        self.save_data_parquet(df, filename)
        return True

    def save_data_parquet(self, df: pd.DataFrame, filename: str) -> None:
        df.to_parquet(filename, index=False)

    def process_years(self) -> bool:
        new_data = False
        for year in self.years:
            new_data = self.process_identifiers(year) or new_data
            time.sleep(10)
        return new_data

    def process_identifiers(self, year: int) -> bool:
        new_data = False
        for identifier in self.identifiers:
            for gp in range(1, 30):
                new_data = self.process_data(year, gp, identifier) or new_data
                # if (not new_data and
                #         identifier == 'R'):
                #     return new_data
                time.sleep(1)
        return new_data


def main():
    raw = ExtractData()
    new_data = raw.process_years()
    print(new_data)


# %%
if __name__ == "__main__":
    main()
# %%
