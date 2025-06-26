from sec_api import ExtractorApi
from settings import config

# SEC_API_KEY = config("SEC_API")
# extractorApi = ExtractorApi(api_key= SEC_API_KEY)
# filing_url_8k = "https://www.sec.gov/Archives/edgar/data/66600/000149315222016468/form8-k.htm"
# # Use CUSIP/CIK/Ticker Mapping API

# #data/#CIK/
# # extract section 1.01 "Entry into Material Definitive Agreement" as cleaned text
# extracted_section_8k = extractorApi.get_section(filing_url_8k, "1-1", "text")
# print(extracted_section_8k)

import requests
SEC_API_KEY = config("SEC_API")  # or use raw string
from sec_api import QueryApi
query_api = QueryApi(api_key=SEC_API_KEY)

query = {
    "query": {
        "query_string": {
            "query": "ticker:TSLA AND formType:\"8-K\" AND filedAt:[2022-01-01 TO 2022-12-31]"
        }
    },
    "from": "0",
    "size": "100",
    "sort": [{"filedAt": {"order": "desc"}}]
}

response = query_api.get_filings(query)

# Print first few URLs
link = []
for filing in response['filings'][:5]:
    print(filing['linkToHtml'])
    link.append(filing['linkToHtml'])
    print(filing['filedAt'])

from urllib.parse import urlparse

def get_base_filing_url(index_url):
    parsed = urlparse(index_url)
    parts = parsed.path.split("/")
    base_path = "/".join(parts[:6])  # Keep only: /Archives/edgar/data/CIK/accession_number
    return f"https://www.sec.gov{base_path}"
    # print(filing.keys())
# print(filings)
link = [get_base_filing_url(url)+"/form8-k.htm" for url in link]
print(link)
extractorApi = ExtractorApi(api_key= SEC_API_KEY)
# filing_url_8k = "https://www.sec.gov/Archives/edgar/data/66600/000149315222016468/form8-k.htm"
# # Use CUSIP/CIK/Ticker Mapping API

# #data/#CIK/
# # extract section 1.01 "Entry into Material Definitive Agreement" as cleaned text
for filing_url in link:
    extracted_section_8k = extractorApi.get_section(filing_url, "8-1", "text")
    # If theres no subsection it returns blank
    print(extracted_section_8k)


# Can we even get the composition? WRDS locked at the moment. As an alternative we can just use snapshot of current holdings.