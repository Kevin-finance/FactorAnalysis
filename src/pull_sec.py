import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from operator import itemgetter
import backoff
import pandas as pd
import tiktoken

from sec_api import ExtractorApi, QueryApi
from tqdm import tqdm

from settings import config


ENC = tiktoken.encoding_for_model("gpt-4o")
SEC_API_KEY = config("SEC_API")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")
OPENAI_SECRET_KEY = config("OPENAI_SECRET")
DATA_DIR = config("DATA_DIR")


query_api = QueryApi(api_key=SEC_API_KEY)
extractorApi = ExtractorApi(api_key=SEC_API_KEY)

# --- Keyword Filtering Patterns ---
PHASE_PATTERN = r"""
    phase                # 'phase'
    \s*                  # optional space
    (?:[123]|I{1,3})      # numeric or roman I-II-III
    (?:[ab])?             # optional a/b
    (?:/(?:[123]|I{1,3})(?:[ab])?)?  # optional slash combo
"""

KEYWORD_PATTERNS = [
    PHASE_PATTERN,
    r"primary endpoint",
    r"topline",
    r"\bNDA\b",
    r"\bBLA\b",
    r"sNDA",
    r"complete response letter",
    r"\bCRL\b",
    r"refusal to file",
    r"\bRTF\b",
    r"\bapproved\b",
    r"acquir",
    r"merger",
    r"ruled against",
    r"ruled in favor",
    r"license",
    r"patent",
    r"awarding .* damages",
    r"infringement",
    r"litigation",
    r"endpoint",
    r"court",
    r"law",
]

# Compiled regex patterns
COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.VERBOSE) for pattern in KEYWORD_PATTERNS
]


class SEC:
    def __init__(self, df):
        self.batch_size = 5  # API limit: 10/sec, 2 Call for Url (7‑1,8‑1) → batch_size = 10 // 2 = 5 URLs/sec
        self.df = df
        self.filing_dict = {}

    def binary_matrix(self, df):
        """
        This method returns binary matrix of all tickers for timeframe.
        If it was a universe of VHT at the time it returns 1 otherwise 0
        """

        # 1) Manipulation on VHT returns table
        df["rdate"] = pd.to_datetime(df["rdate"])
        # Following code converts e.g 03-31 to 03-01, because such holdings are holdings as of first day of 3,6,9,12
        df["rdate"] = df["rdate"].dt.to_period("M").dt.to_timestamp()
        df.dropna(
            subset=["ticker"], inplace=True
        )  # SEC matches up with tickers thus drop rows with no ticker
        df["exist"] = 1

        # 2) Checking whether it was universe at a certain time interval
        matrix = pd.pivot_table(
            df, index="rdate", columns="ticker", values="exist", fill_value=0
        )
        return matrix

    def extract_intervals(self):
        """
        This method returns start and end date for each tickers by using binary matrix.
        Expected output is as follows:
        # ticker start_date end_date num_intervals
        # AAPL    2015-01-01 2015-05-30 2
        # AAPL    2016-01-01 2016-05-30 2

        """
        matrix = self.binary_matrix(self.df)
        results = []

        for ticker in matrix.columns:
            series = matrix[ticker].fillna(0).astype(int)

            # Check with previous values to detect any change
            prev = series.shift(1, fill_value=0)
            next_ = series.shift(-1, fill_value=0)

            starts = (series == 1) & (prev == 0)
            ends = (series == 1) & (next_ == 0)

            start_dates = matrix.index[starts]
            end_dates = matrix.index[ends]

            # match up with ticker
            for s, e in zip(start_dates, end_dates):
                results.append({"ticker": ticker, "start_date": s, "end_date": e})
        df = pd.DataFrame(results)

        num_interval_df = df.groupby("ticker").agg(
            num_intervals=("end_date", lambda x: len(x))
        )

        merge_df = df.merge(num_interval_df, on="ticker", how="left")

        return merge_df

    def query_api(self):
        """
        This method pulls out accessionNo, linkedToHTML, filedAt info for tickers in 8-K 7-1,8-1 section.

        """
        merge_df = self.extract_intervals()
        unique_tickers = merge_df["ticker"].unique()
        offset = 0
        size = 50
        # 3) Passing queries for each unique tickers to SEC API
        for ticker in tqdm(unique_tickers, desc="Processing each tickers"):
            part_df = merge_df[merge_df["ticker"] == ticker]  # only single ticker
            date_ranges = list(
                part_df[["start_date", "end_date"]].itertuples(index=False, name=None)
            )  # This comes out in [(start_date1,end_date1),(start_date2,end_date2)...]
            should_clauses = [
                {
                    "range": {
                        "filedAt": {
                            "gte": s.strftime("%Y-%m-%d"),
                            "lte": e.strftime("%Y-%m-%d"),
                        }
                    }
                }
                for s, e in date_ranges
            ]

            while True:
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "query_string": {
                                        "query": f'ticker:{ticker} AND formType:"8-K" AND (items : "7.01" OR items : "8.01")'  # Only Scrap 7-1, 8-1
                                    }
                                }
                            ],
                            "filter": {
                                "bool": {
                                    "should": should_clauses,
                                    "minimum_should_match": 1,
                                }
                            },
                        }
                    },
                    "from": offset,
                    "size": size,
                    "sort": [{"filedAt": {"order": "desc"}}],
                }

                # Requests corresponding start,end date filings for each ticker
                response = query_api.get_filings(query)
                filings = response.get("filings", [])

                # For each filing in between start_date, end_date save filings info
                for filing in filings:
                    ticker = filing["ticker"]
                    link = filing.get("linkToFilingDetails")
                    filed_at = filing.get("filedAt")
                    accession_no = filing.get("accessionNo", None)

                    if ticker not in self.filing_dict:
                        self.filing_dict[ticker] = {
                            "linkToFilingDetails": [],
                            "filedAt": [],
                            "accessionNo": [],
                        }

                    self.filing_dict[ticker]["linkToFilingDetails"].append(link)
                    self.filing_dict[ticker]["filedAt"].append(filed_at)
                    self.filing_dict[ticker]["accessionNo"].append(accession_no)

                if len(filings) < size:
                    break  # No more to retrieve
                offset += size

        return self

    def filter_relevant_keywords(text):
        """
        This method filters text that doesn't meet our criteria or predefined filter words
        Such method is useful to reduce the number of text being fed into GPT, reducing cost and latency.
        """
        if not text or len(text.strip()) < 10:
            return False

        for pattern in COMPILED_PATTERNS:
            if pattern.search(text):
                return True

        return False

    @backoff.on_exception(
        backoff.expo,
        max_tries=5,
        on_backoff=lambda details: print(
            f"[BACKOFF] Retrying {details['target'].__name__} - attempt {details['tries']}"
        ),
    )
    def get_section_with_retry(url, section):
        """
        Wrapper function for extractorApi.get_section
        """
        return extractorApi.get_section(url, section)

    def fetch_sections(self, urls: list[str]) -> list[dict]:
        print(f"[DEBUG] Input URLs count: {len(urls)}")
        print(f"[DEBUG] Unique URLs count: {len(set(urls))}")

        # For each url make a dictionary
        temp = {url: {"7-1": "", "8-1": ""} for url in urls}
        batch_size = self.batch_size

        for i in range(0, len(urls), batch_size):
            batch = urls[i : i + batch_size]
            with ThreadPoolExecutor(max_workers=batch_size * 2) as executor:
                futures = {
                    executor.submit(self.get_section_with_retry, url, sec): (url, sec)
                    for url in batch
                    for sec in ("7-1", "8-1")
                }
                for fut in as_completed(futures):
                    url, sec = futures[fut]
                    temp[url][sec] = fut.result()

            # Sleep between batch
            time.sleep(1)

        return self.filter_relevant_text(self, urls, temp)

    def filter_relevant_text(self, urls, temp):
        results = []
        filtered_count = 0
        total_count = len(urls)
        for url in urls:
            combined_text = temp[url]["7-1"] + temp[url]["8-1"]

            # Checking if the combined text has relevant keywords
            # If it does append otherwise none
            if self.has_relevant_keywords(combined_text):
                results.append({"url": url, "text": combined_text})
            else:
                results.append({"url": url, "text": None})
                filtered_count += 1

        filter_rate = (filtered_count / total_count) * 100 if total_count > 0 else 0

        print(
            f"[FILTERING] Total: {total_count}, Filtered: {filtered_count}, Passed: {total_count - filtered_count}"
        )
        print(
            f"[FILTERING] Filter rate: {filter_rate:.1f}% (reduced GPT calls by {filter_rate:.1f}%)"
        )

        return results

    def extractor_api(self):
        # Once pulling out filings info is done, now we access each section to pull out the texts

        for idx, ticker in enumerate(tqdm(self.filing_dict, desc="Event Processing")):
            # Use Multithreading for 7-1 , 8-1
            filing_urls = self.filing_dict[ticker]["linkToFilingDetails"]
            print(f"[DEBUG] Ticker: {ticker}")
            print(f"[DEBUG] Total URLs for this ticker: {len(filing_urls)}")
            print(f"[DEBUG] Unique URLs for this ticker: {len(set(filing_urls))}")

            if len(filing_urls) != len(set(filing_urls)):
                print(f"[DEBUG] DUPLICATE URLs found in ticker {ticker}!")
            sec_items = self.fetch_sections(filing_urls)
            self.filing_dict[ticker]["text"] = [d["text"] for d in sec_items]

            print(self.filing_dict[ticker]["text"])

        return self

    def get(self):
        return self.filing_dict


def main(self):
    df = pd.read_parquet(DATA_DIR / "vht_holdings.parquet")
    sec = SEC(df)
    sec.query_api()
    sec.extractor_api()
    with open(DATA_DIR / "sec_text.pkl", "wb") as f:
        pickle.dump(sec.get(), f)


if __name__ == "__main__":
    main()
