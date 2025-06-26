from pathlib import Path
from settings import config
import pandas as pd

DATA_DIR = config("DATA_DIR")

class Preprocessor:
    def __init__(self,dir):
        self.df = self._load_data(Path(dir))

    def _load_data(self,path):
        self.df = pd.read_parquet(path)

        return self.df 
    
    def detrend(self):
        # How do we check if its trending?
        # Rates - difference, level - log diff
        
        return self

    def deseaonalize(self):
        return self
    
    def get(self):

        return self.df


if __name__=="__main__":
    macro_map_dir = DATA_DIR/"macro_map.parquet"
    macro_latest_series = DATA_DIR/"macro_latest_series.parquet"

    macro_map = Preprocessor(macro_map_dir).get()
    macro_latest_series = Preprocessor(macro_latest_series).get()

    print(macro_map)
    print(macro_latest_series)
    print(macro_latest_series['WEI'])