import pandas as pd
from openai import OpenAI
from sec_api import ExtractorApi, QueryApi
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from settings import config
import time
import pickle
from operator import itemgetter
import math

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
        return None
    # reference : https://platform.openai.com/docs/api-reference/responses/object
    try:
        instructions = """"Suppose you are an investment manager.  
1. 
If it clearly states that a Phase 1, Phase 1a or Phase 1b trial successfully met its primary endpoint
(e.g. “met its primary endpoint,” “positive topline results,” “demonstrated significant improvement”), return 1.
Example1. On September 18, 2023, Aclaris Therapeutics, Inc. (the “Company”) issued a press release announcing positive preliminary results from its Phase 1 multiple ascending dose trial of ATI-2138, an investigational oral covalent ITK/JAK3 inhibitor (the “ATI-2138 Phase 1 Results”).
Example2. On April 9, 2024, Alkermes plc (the “Company”) announced positive topline results from the narcolepsy type 2 and idiopathic hypersomnia cohorts of its phase 1b study in which it evaluated ALKS 2680, the Company’s novel, investigational, oral orexin 2 receptor agonist


2.
Same for Phase 2/2a/2b → return **2**
If Phase 1/1a/1b and Phase2/2a/2b is mixed then classify as 2
Example1. On January 28, 2020, Akcea Therapeutics, Inc. (the “Company”) announced positive topline results from the Phase 2 study of AKCEA-ANGPTL3-LRx in patients with hypertriglyceridemia, type 2 diabetes and non-alcoholic fatty liver disease.
Example2. On April 12, 2021, Sage Therapeutics, Inc. issued a press release titled “Sage Therapeutics and Biogen Announce SAGE-324 Phase 2 Placebo-Controlled KINETIC Study in Essential Tremor Met Primary Endpoint.”

3.
Same for Phase 3 → return **3**
If Phase Phase2/2a/2b and Phase3 is mixed then classify as 3
Example1. On January 26, 2021, Agios Pharmaceuticals, Inc. issued a press release announcing that its global phase 3 ACTIVATE-T trial of mitapivat in adults with pyruvate kinase deficiency who are regularly transfused met its primary endpoint.
Example2. On September 26, 2016, Array issued a press release announcing the top-line results from Part 1 of the ongoing Phase 3 clinical trial of binimetinib and encorafenib in patients with advanced BRAF-mutant melanoma, known as the COLUMBUS trial. The study met its primary endpoint of improving progression-free survival.
Example3. On December 1, 2022, Anavex Life Sciences Corp., a Nevada corporation (the “Company”) issued a press release (the “Press Release”) providing top line data from its Phase 2b/3 double-blind, placebo-controlled trial of ANAVEX®2-73 in Alzheimer’s disease. The trial met its primary and key secondary endpoints with statistically significant results.
Example4. On January 13, 2020, Biohaven Pharmaceutical Holding Company Ltd. will be making an investor presentation (the “Presentation”), which includes updates for communication received from the United States Food and Drug Administration in December 2019, summarizing a Late Cycle Communication regarding the rimegepant new drug application review, and positive topline results in the pivotal Phase 2/3 study of vazegepant.
Example5. On April 27, 2020, Axsome Therapeutics, Inc. (the “Company”) issued a press release announcing that AXS-05 met the prespecified primary endpoint and significantly improved agitation in patients with Alzheimer’s disease in the Company’s ADVANCE-1 Phase 2/3 trial.

4.
If it announces a new NDA or BLA being filed, submitted, or resubmitted, return 4.
Example1. In the press release dated October 31, 2016, ARIAD Pharmaceuticals, Inc. (“ARIAD” or the “Company”) announced that the U.S. Food and Drug Administration (FDA) has accepted for review the New Drug Application (NDA) for ARIAD’s investigational oral anaplastic lymphoma kinase (ALK) inhibitor, brigatinib, in patients with metastatic ALK-positive (ALK+) non-small cell lung cancer (NSCLC) who have progressed on crizotinib
Example2. On January 19, 2016, Incyte Corporation (“Incyte”) and Eli Lilly and Company (“Lilly”) announced that Lilly had submitted a new drug application (“NDA”) to the U.S. Food and Drug Administration (“FDA”) for the approval of oral once-daily baricitinib for the treatment of moderately-to-severely active rheumatoid arthritis.


5.
**Only** if the text contains the exact phrase **“Complete Response Letter”** or the abbreviation **“CRL”** (case-insensitive) in the context of **receipt** (e.g. “received a CRL,” “received a Complete Response Letter”), return **5**..
Example1. On August 11, 2021, FibroGen, Inc. (“FibroGen”) issued a press release announcing that the U.S. Food and Drug Administration (the “FDA”) has issued a complete response letter regarding the New Drug Application (“NDA”) for roxadustat for the treatment of anemia of chronic kidney disease. The letter indicates the FDA will not approve the roxadustat NDA in its present form and has requested additional clinical study of roxadustat to be conducted, prior to resubmission.
Example2. On April 14, 2017, Eli Lilly and Company and Incyte Corporation announced that the U.S. Food and Drug Administration (“FDA”) has issued a complete response letter for the New Drug Application of the investigational medicine baricitinib, a once-daily oral medication for the treatment of moderate-to-severe rheumatoid arthritis.


6.
If it announces a new FDA approval event—i.e., contains an approval term plus a new-approval verb or date
Example1. On February 13, 2018, ABIOMED, Inc. issued a press release reporting that is has received an expanded U.S. Food and Drug Administration (FDA) Pre-Market Approval (PMA) for its Impella 2.5®, Impella CP®, Impella 5.0® and Impella LD® heart pumps to provide treatment for heart failure associated with cardiomyopathy leading to cardiogenic shock.
Example2. On January 31, 2020, Aimmune Therapeutics, Inc. (“Aimmune” or the “Company”) issued a press release announcing that the U.S. Food and Drug Administration had approved PALFORZIA™ (Peanut (Arachis hypogaea) Allergen Powder-dnfp). PALFORZIA is the first approved treatment for patients with peanut allergy. PALFORZIA is an oral immunotherapy indicated for the mitigation of allergic reactions, including anaphylaxis, that may occur with accidental exposure to peanut.


0.
Otherwise (background mentions, negative/mixed results, historical recaps, “announcing data,” “first doses,” etc.), return 0.

Return 0~6 only. Just the number. No dash or words.
Review your output. 
    """
 
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": text}
            ],
            temperature=0, # to ensure fully deterministic behavior (argmax)
            logprobs=True,
            top_logprobs=7, # Show only top 7 tokens' log prob
            max_tokens = 1, # maximum token size = 1
        )
        logprobs_dict = {
        top.token.strip(): math.exp(top.logprob)
        for top in response.choices[0].logprobs.content[0].top_logprobs}
        print(logprobs_dict)

        return logprobs_dict

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

def extract_best_token(logprob_dict):
    if not logprob_dict:
        return None
    return max(logprob_dict.items(), key=itemgetter(1))[0]  # returns the key with highest log prob

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
        if "event" not in filing_dict[ticker]:
            filing_dict[ticker]["event"] = []
            filing_dict[ticker]["logprob"] = []

        filing_urls = filing_dict[ticker]["linkToFilingDetails"]
        with ThreadPoolExecutor(max_workers=20) as executor:
            # classify_filing either returns none or a dictionary of log prob when certain event is involved.
            logprobs_list = list(executor.map(classify_filing, filing_urls))  

        # Pull the token with highest probability for each filings
        best_tokens = [
            int(token) if token is not None else None
            for token in [extract_best_token(lp) for lp in logprobs_list]]

        filing_dict[ticker]["event"] = best_tokens
        filing_dict[ticker]["logprob"] = logprobs_list
        # print(filing_dict[ticker]["event"])
        # print(filing_dict[ticker]["logprob"])

        if idx % 10 == 0:
            with open(CHECK_POINT, "wb") as f:
                pickle.dump(filing_dict, f)
    return filing_dict



if __name__ == "__main__":
    # # Note : it takes around 1hr to produce .pkl file
    # df = pd.read_parquet(DATA_DIR / "vht_holdings.parquet")
    # filings = pull_filings_link(df)

    # with open(DATA_DIR / "filings_dict.pkl", "wb") as f:
    #     pickle.dump(filings, f)

    with open(DATA_DIR / "filings_dict.pkl","rb") as f:
        temp = pickle.load(f)
    
