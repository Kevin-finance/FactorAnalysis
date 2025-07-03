import pandas as pd
from openai import OpenAI
from sec_api import ExtractorApi, QueryApi
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from settings import config
import time
import pickle

SEC_API_KEY = config("SEC_API")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")
OPENAI_SECRET_KEY = config("OPENAI_SECRET")
DATA_DIR = config("DATA_DIR")
CHECK_POINT = DATA_DIR / "event_log.pkl"

client = OpenAI(api_key=OPENAI_SECRET_KEY, max_retries=5)


query_api = QueryApi(api_key=SEC_API_KEY)
extractorApi = ExtractorApi(api_key=SEC_API_KEY)


def get_section_with_retry(url, section, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return extractorApi.get_section(url, section, "text")
        except Exception as e:
            if "429" in str(e):
                print(
                    f"[SEC API] 429 Too Many Requests on {section} for {url} (attempt {attempt + 1})"
                )
                time.sleep(delay * (attempt + 1))
            else:
                print(f"[SEC API] Other error on {section}: {e}")
                break
    return ""


def classify_filing(url):
    """
    This method feeds SEC 8-K section 7.1 and 8.1 to the OPENAI api and returns corresponding events
    It is useful in the sense that it parallelizes GPT calls, SEC api calls which are a very expensive operation.
    """
    extracted_section_7 = get_section_with_retry(url, "7-1")
    extracted_section_8 = get_section_with_retry(url, "8-1")
    text = extracted_section_7 + extracted_section_8

    if (len(text) < 10) or not (
        ("fda" in text.lower()) or ("phase" in text.lower())
    ):  # Doesn't pass texts that are less than length 10
        print(text)
        return None

    try:
        instructions = """Suppose you are an investment manager.  
Review the text and return:
1. **1, 2, or 3**  
    If it clearly states that a Phase 1/2/3 trial **successfully met its primary endpoint** \n
    (e.g. “met its primary endpoint,” “positive topline results,” “demonstrated significant improvement”), \n
    return **1**, **2**, or **3**, using only the number immediately following the word “Phase.”

2. **4**  
    If it announces a **new** NDA or BLA being **filed**, **submitted**, or **resubmitted**, return **4**.

3. **5**  
    If it describes receipt of a **Complete Response Letter** (CRL) from the FDA, return **5**.

4. **6**  
    If it announces a **new FDA approval event**—i.e., it contains an FDA-approval term **plus** a new-approval verb or date \n
    (e.g. “received FDA approval on [date],” “was approved by the FDA on [date],” “gained accelerated approval”), return **6**.

5. **-1**  
    Otherwise (including background mentions of existing approvals, withdrawals, “announcing data,” “first doses,” negative/mixed trial results, historical recaps, etc.), return **-1**.

    Review your reponses.
    """
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=instructions,
            input=text,
        )
        return int(response.output_text.strip())

    except Exception as e:
        return None


def extract_intervals(binary_matrix):
    results = []

    for ticker in binary_matrix.columns:
        series = binary_matrix[ticker].fillna(0).astype(int)

        # Check with previous values to detect any change
        prev = series.shift(1, fill_value=0)
        next_ = series.shift(-1, fill_value=0)

        starts = (series == 1) & (prev == 0)
        ends = (series == 1) & (next_ == 0)

        start_dates = binary_matrix.index[starts]
        end_dates = binary_matrix.index[ends]

        # match up with ticker
        for s, e in zip(start_dates, end_dates):
            results.append({"ticker": ticker, "start_date": s, "end_date": e})
    df = pd.DataFrame(results)

    return df


def pull_filings_link(df):
    # df is a panel data with holdings of VHT ETF
    # This searches for 8-K filings of a given company a html link

    df["rdate"] = pd.to_datetime(df["rdate"])
    df["rdate"] = df["rdate"].dt.to_period("M").dt.to_timestamp()
    df.dropna(subset=["ticker"], inplace=True)
    df.loc[:, "exist"] = 1

    # This binary matrix returns 1 if its holding at that time 0 otherwise
    binary_matrix = pd.pivot_table(
        df, index="rdate", columns="ticker", values="exist", fill_value=0
    )

    date_parse_df = extract_intervals(binary_matrix)
    num_interval_df = date_parse_df.groupby("ticker").agg(
        num_intervals=("end_date", lambda x: len(x))
    )

    merge_df = date_parse_df.merge(num_interval_df, on="ticker", how="left")

    filing_dict = {}

    for ticker in tqdm(merge_df["ticker"].unique(), desc="Processing each tickers"):
        part_df = merge_df[merge_df["ticker"].isin([ticker])]  # only single ticker
        date_ranges = list(
            part_df[["start_date", "end_date"]].itertuples(index=False, name=None)
        )
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
        offset = 0
        size = 50

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

            response = query_api.get_filings(
                query
            )  # value: total number of filings matching the query
            filings = response.get("filings", [])

            for filing in filings:
                ticker = filing["ticker"]
                link = filing.get("linkToFilingDetails")
                filed_at = filing.get("filedAt")
                accession_no = filing.get("accessionNo", None)

                if ticker not in filing_dict:
                    filing_dict[ticker] = {
                        "linkToFilingDetails": [],
                        "filedAt": [],
                        "accessionNo": [],
                    }

                filing_dict[ticker]["linkToFilingDetails"].append(link)
                filing_dict[ticker]["filedAt"].append(filed_at)
                filing_dict[ticker]["accessionNo"].append(accession_no)

            if len(filings) < size:
                break  # No more to retrieve
            offset += size

    # Distinguishing events
    for idx, ticker in enumerate(tqdm(filing_dict, desc="Event Processing")):
        # print(ticker)
        if "event" not in filing_dict[ticker]:
            filing_dict[ticker]["event"] = []

        # Change
        filing_urls = filing_dict[ticker]["linkToFilingDetails"]
        with ThreadPoolExecutor(max_workers=20) as executor:
            events = list(executor.map(classify_filing, filing_urls))
        filing_dict[ticker]["event"] = events

        if idx % 10 == 0:
            with open(CHECK_POINT, "wb") as f:
                pickle.dump(filing_dict, f)
            # print(f" Checkpoint saved at ticker {idx}")
            # print(f"Saving to {CHECK_POINT.resolve()}")
            # print(f"📦 filing_dict size at idx {idx}: {len(filing_dict)}")
            # print(f"📦 {ticker} has {len(filing_dict[ticker]['linkToFilingDetails'])} filings")
        print(filing_dict[ticker]["event"])
    return filing_dict


if __name__ == "__main__":
    # Note : it takes around 1hr to produce .pkl file
    df = pd.read_parquet(DATA_DIR / "vht_holdings.parquet")
    filings = pull_filings_link(df)

    with open(DATA_DIR / "filings_dict.pkl", "wb") as f:
        pickle.dump(filings, f)

    # with open(DATA_DIR / "filing_dict.pkl","rb") as f:
    #     pickle.load(f)
