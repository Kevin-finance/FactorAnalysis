import pandas as pd
from openai import OpenAI
from sec_api import ExtractorApi, QueryApi
from tqdm import tqdm

from settings import config

# SEC_API_KEY = config("SEC_API")
# extractorApi = ExtractorApi(api_key= SEC_API_KEY)
# filing_url_8k = "https://www.sec.gov/Archives/edgar/data/66600/000149315222016468/form8-k.htm"
# # Use CUSIP/CIK/Ticker Mapping API

# #data/#CIK/
# # extract section 1.01 "Entry into Material Definitive Agreement" as cleaned text
# extracted_section_8k = extractorApi.get_section(filing_url_8k, "1-1", "text")
# print(extracted_section_8k)


SEC_API_KEY = config("SEC_API")  # or use raw string
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")
OPENAI_SECRET_KEY = config("OPENAI_SECRET")

client = OpenAI(api_key=OPENAI_SECRET_KEY, max_retries = 5 )

# Purpose : For every holdings at the time loop through 8-K and return 1 if have such filings otherwise 0


query_api = QueryApi(api_key=SEC_API_KEY)
extractorApi = ExtractorApi(api_key=SEC_API_KEY)


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

    def extract_intervals(binary_matrix):
        results = []

        for ticker in binary_matrix.columns:
            series = binary_matrix[ticker].fillna(0).astype(int)

            # 이전 값과 비교해 변화 감지
            prev = series.shift(1, fill_value=0)
            next_ = series.shift(-1, fill_value=0)

            starts = (series == 1) & (prev == 0)
            ends = (series == 1) & (next_ == 0)

            start_dates = binary_matrix.index[starts]
            end_dates = binary_matrix.index[ends]

            # 매칭해서 ticker별로 결과 저장
            for s, e in zip(start_dates, end_dates):
                results.append({"ticker": ticker, "start_date": s, "end_date": e})

        return pd.DataFrame(results)

    date_parse_df = extract_intervals(binary_matrix)
    num_interval_df = date_parse_df.groupby("ticker").agg(
        num_intervals=("end_date", lambda x: len(x))
    )

    merge_df = date_parse_df.merge(num_interval_df, on="ticker", how="left")

    filing_dict = {}

    for ticker in tqdm(merge_df["ticker"].unique(), desc="Processing each tickers"):
        part_df = merge_df[merge_df["ticker"].isin([ticker])]  # only single ticker
        date_ranges = list(
            part_df[["start_date", "end_date"]].itertuples(index=False, name=None) # 
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
                                    "query": f'ticker:{ticker} AND formType:"8-K"' # Only Scrap 7-1,8-1 , think we should do multithreading as well 
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
    for ticker in tqdm(filing_dict, desc="Event Processing"):
        print(ticker)
        if "event" not in filing_dict[ticker]:
            filing_dict[ticker]["event"] = []
        print(ticker)
        print(filing_dict[ticker]["linkToFilingDetails"])
        for num in tqdm(
            range(len(filing_dict[ticker]["linkToFilingDetails"])),
            desc="GPT Processing",
        ):
            url = filing_dict[ticker]["linkToFilingDetails"][num]
            extracted_section_7 = extractorApi.get_section(
                url, "7-1", "text"
            )  # If theres no subsection it returns blank
            extracted_section_8 = extractorApi.get_section(url, "8-1", "text")
            # extracted_section_9 = extractorApi.get_section(url, "9-1", "text")
            texts = extracted_section_7 + extracted_section_8 
            print(type(texts))
            print("all ")
            print(texts)
            if texts == "":
                pass
            else:
                print("Not None")
                print(texts)
                response = client.responses.create(
                    model="gpt-4o-mini",
                    instructions="Suppose you are a investment manager. According to the text that is fed, if there FDA approval, \
                    termination of clinical trial phase 1, 2 or 3 related texts then return: 0 (FDA approval), 1 (Phase 1), 2 (Phase 2), 3 (Phase 3)",
                    input=texts,
                )
                try:
                    event_number = int(response.output_text.strip())
                except:
                    event_number = None

                filing_dict[ticker]["event"].append(event_number)
                print("event")
                print(filing_dict[ticker]["event"])
    return filing_dict


if __name__ == "__main__":
    DATA_DIR = config("DATA_DIR")
    # df = pd.read_parquet(DATA_DIR / "vht_holdings.parquet")
    # temp = pull_filings_link(df)
    # temp.to_parquet(DATA_DIR/"filing_dict.parquet")
    # # print(temp)
    
