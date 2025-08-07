import asyncio
import json
import math
import pickle
from decimal import Decimal
import utils
from settings import config
import backoff
import openai
import requests
import tiktoken
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import random
from preprocessing import Preprocessor
from settings import config
from collections import Counter
import copy
from collections import defaultdict
import copy
from typing import List


ENC = tiktoken.encoding_for_model("gpt-4o")
DATA_DIR = config("DATA_DIR")
PROMPT_DIR = config("PROMPT_DIR")
OPENAI_SECRET_KEY = config("OPENAI_SECRET")

rpm_limiter = AsyncLimiter(300, 60)  # 300 calls for 60seconds. Used to control RPM


class OpenAIClassfier:
    def __init__(self, sample_seed = 42 ):

        self.sec = pickle.load(open(DATA_DIR/"sec_texts.pkl", "rb"))
        self.sem = asyncio.Semaphore(50) # Number of asynchronous calls
        self.rpm_limiter = AsyncLimiter(300, 60)
 

        self.client = OpenAI(api_key=OPENAI_SECRET_KEY, max_retries=5)
        self.async_client = AsyncOpenAI(api_key=OPENAI_SECRET_KEY, max_retries=5)
        self.sample_seed = sample_seed
        self.n_per_label = 40
        self.sample_filing = None # filing_dict with event labels
        self.ticker_dict = None
        

    @staticmethod
    def truncate_to_n_tokens(text: str, n: int = 1500) -> str:
        """
        This is a static method where it cuts the text to the first n tokens
        Reduces cost and latency
        """

        # 1) Tokenize given text
        tokens = ENC.encode(text)
        # 2) Cut the first n token
        if len(tokens) > n:
            tokens = tokens[:n]
        # 3) Decode it and convert it back to literal
        return ENC.decode(tokens)


    async def first_pass_classification(self, model, system_prompt, text_dict = None):
        """
        This method does classification on the texts.
        The dictionary should be saved as such format: {'A':{'linkToFilingDetails':[...],'filedAt':[...]},'B':...}
        If text_dict parameter is not given, for e.g. first cheap pass then it uses saved sec_texts file.
        text_dict can be given as a sample as well but in the same format. 
        
        Returns {"MSFT": {"linkToFilingDetails": [ … ],"text": [ … ],"event":[ 1, 1, 4, … ],"logprob":[ {…}, {…}, {…}, … ],},"AAPL":{"linkToFilingDetails":[...]}
        """
        filing_dict = copy.deepcopy(self.sec) if text_dict is None else copy.deepcopy(text_dict)
        
        for ticker in tqdm(filing_dict,desc = "Filing texts sorting processing"):
            items = [
                {"link": link, "text": text}
                for link, text in zip(
                    filing_dict[ticker]["linkToFilingDetails"],
                    filing_dict[ticker]["text"]
                )
            ]
            # asynchrnous calls within events , serially among tickers
            classified = await self.run_classification(items, model,system_prompt) # [{'link':...''text':...},{}]
           
            filing_dict[ticker]["event"] = [int(r["best_token"]) for r in classified] 
            filing_dict[ticker]["logprob"] = [r["logprob"] for r in classified]
            
        self.sample_filing = filing_dict # used for stratified sampling(?)
       
        return filing_dict

    async def second_pass_judge(self,target_event: List, model: str, grader_template: str, classification_prompt: str, pickle_path: str = None) -> dict:
        """
        1) Load sample_filing from pickle if needed 
        2) Find all entries where event == target_event, track their indices
        3) Run grader LLM on those items
        4) Write back judge_score into exactly those positions
        Returns the updated sample_filing.
        """
        print("수정")
        # 1) Ensure we have sample_filing
        if getattr(self, "sample_filing", None) is None:
            if not pickle_path:
                raise ValueError("No in-memory sample_filing and no pickle_path provided.")
            
            with open(pickle_path, "rb") as f:
                self.sample_filing = pickle.load(f)

        # 2) Deep copy to work on
        filing = copy.deepcopy(self.sample_filing)

        # 3) Collect indices + items to grade
        
        subset = defaultdict(lambda: {"indices": [], "items": []})
        for ticker, info in filing.items():
            # initialize judge_score array if missing
            if "judge_score" not in info:
                info["judge_score"] = [None] * len(info.get("event", []))

            for idx, (ev, link, text) in enumerate(
                zip(info.get("event", []),
                    info.get("linkToFilingDetails", []),
                    info.get("text", []))
            ):
                if ev in target_event:
                    print("target event!!")
                    print({
                        "link": link,
                        "text": text,
                        "previous_pred": str(ev)
                    })
                    subset[ticker]["indices"].append(idx) 
                    subset[ticker]["items"].append({
                        "link": link,
                        "text": text,
                        "previous_pred": str(ev)
                    })

        # 4) Grade each ticker’s subset and merge back (subset only contains target event)
        for ticker, data in tqdm(subset.items(),desc="Grader Processing"):
            if not data["items"]:
                continue
            
            graded = await self.run_grading(
                items=data["items"],
                model=model,
                grader_template=grader_template,
                classification_prompt=classification_prompt,
            )
            print("graded")
            print(graded)

            # write back
            for pos, res in zip(data["indices"], graded):
                filing[ticker]["judge_score"][pos] = res["judge_score"]

        # 5) Save and return
        self.sample_filing = filing

        return filing

    def compute_distribution(self, sample_filing: dict) -> dict[int, int]:
        """
        CHEAP PASS CLASSIFICATION STEP2
        Returns number of events for each events
        """
        all_events = []
        for v in self.sample_filing.values():
            all_events.extend(v["event"])
        dist = Counter(all_events) # {0: 120, 1: 45, 2: 230, ...}
        
        return dict(dist)

    def stratified_sample(self,threshold):
        """
        This event returns stratified sample according to the first pass classification
        
        """
        # 1) For reproducibility
        random.seed(self.sample_seed)

        # 2) bucketizing by event label
        buckets = defaultdict(list) # {'0':[{'ticker':...,},{'ticker':...}]}
        for ticker, info in self.sample_filing.items():
            filedAts      = info.get("filedAt",      [None] * len(info["text"]))
            judge_scores  = info.get("judge_score",  [None] * len(info["text"]))
            for (link, text, lbl, logprob, filedAt, js) in zip(
                info["linkToFilingDetails"],
                info["text"],
                info["event"],
                info["logprob"],
                filedAts,
                judge_scores
              
            ):
                if max(logprob.values()) < threshold:
                    continue
                else:
                    buckets[lbl].append({
                    "ticker":       ticker,
                    "filedAt":      filedAt,
                    "link":         link,
                    "event":        lbl,
                    "logprob":      logprob,
                    "text":         text,
                    "judge_score":  js
                    
                })

        # 3) stratified sampling
        sampled = [] 
        for items in buckets.values():
            k = min(self.n_per_label, len(items))
            sampled.extend(random.sample(items, k))
        random.shuffle(sampled)

        # 4) group back into ticker→filing_dict style
        ticker_dict: dict[str, dict] = {}
        for s in sampled:
            t = s["ticker"]
            if t not in ticker_dict:
                ticker_dict[t] = {
                    "linkToFilingDetails": [],
                    "text":                 [],
                    "event":                [],
                    "logprob":              [],
                    "filedAt":              [],
                    "judge_score":          []
                }
            d = ticker_dict[t]
            d["linkToFilingDetails"].append(s["link"])
            d["text"].append(s["text"])
            d["event"].append(s["event"])
            d["logprob"].append(s["logprob"])
            d["filedAt"].append(s["filedAt"])
            d["judge_score"].append(s["judge_score"])
        self.ticker_dict = ticker_dict
           
        return ticker_dict

    def save_ground_truth(self, filtered_events, output_path):
        """
        This generates excel file so that we can fill in the groundtruth for events
        """

        with pd.ExcelWriter(output_path) as writer:
            for event_name, filings in filtered_events.items():
                df = pd.DataFrame(filings)
                if df.empty:
                    print(f"[WARNING] '{event_name}' is empty. Skipping...")
                    continue
                df.sort_values(by='logprob',inplace=True)
                df['groundtruth'] = None
                df.to_excel(writer,sheet_name=event_name,index=False)
    
    @backoff.on_exception(
        backoff.expo,
        openai.RateLimitError,
        max_tries=5,
        on_backoff=lambda details: print(
            f"[BACKOFF] Retrying {details['target'].__name__} - attempt {details['tries']}"
        ),
    )
    async def classify_text_async(self, item: dict, sem: asyncio.Semaphore, model, system_prompt):
        """
        If this method is called then it feeds text with predefined instruction

        """
        # If the text is None then don't feed it to GPT, it reduces cost and latency
        # Even though it's filtered ones, there could be text with None
        if item["text"] is None:
            return {"url": item["link"], "best_token": "0", "logprob": {"0": 1.0}}

        # Truncate the text to 1500 tokens to boost the speed and take care of context window
        truncated = self.truncate_to_n_tokens(item["text"])

    

        async with self.sem,self.rpm_limiter:  # Start async context manager, if one process is done, it lets the other queued process
            resp = await self.async_client.chat.completions.create(  # This requests response and then suspends temporarily, requests the others
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": truncated},
                ],
                temperature=0,
                logprobs=True,
                top_logprobs=8,
                max_tokens=2,
                seed=0,
            )
            tops = resp.choices[0].logprobs.content[0].top_logprobs
            logprob = {top.token.strip(): math.exp(top.logprob) for top in tops}

            if not logprob:
                return {"url": item["link"], "best_token": "0", "logprob": {"0": 1.0}}

            best = max(logprob, key=logprob.get)
            item["best_token"] = best
            # print("best")
            # print(best)
            return {
                "url": item["link"],
                "text": item["text"],
                "best_token": best,
                "logprob": logprob,
            }
    
    async def run_classification(self, items: list[dict], model, system_prompt) -> list[dict]:
        """
        items: [{"link":..., "text":...}, ...]
        return: [{"url":..., "text":..., "best_token":..., "logprob":...}, ...]
        """
        # 1) Filter out texts that are None
        filtered_items = [item for item in items if item["text"] is not None]
        none_items = [item for item in items if item["text"] is None]

        # 2) Async classification
        # Each items run in parallel because this runs within for idx, ticker in enumerate
        tasks = [self.classify_text_async(item, self.sem,model = model, system_prompt = system_prompt) for item in filtered_items]

        # This doesn't guarantee the sequence
        classified_results = await asyncio.gather(*tasks)

        # 3) Taking care of None
        for item in none_items:
            classified_results.append(
                {
                    "url": item["link"],
                    "text": item.get("text", None),
                    "best_token": "0",
                    "logprob": {"0": 1.0},
                }
            )

        # 4) Reorder
        url_to_result = {r["url"]: r for r in classified_results}  # {'link' : {}}

        ordered_results = [url_to_result[item["link"]] for item in items]

        return ordered_results

    @backoff.on_exception(
        backoff.expo,
        openai.RateLimitError,
        max_tries=5,
        on_backoff=lambda details: print(
            f"[BACKOFF] Retrying {details['target'].__name__} – attempt {details['tries']}"
        ),
    )
    async def grade_text_async(self, item: dict, sem: asyncio.Semaphore, model: str, grader_template: str,classification_prompt: str,) -> dict:
        """
        Runs the grader LLM on a single item.
        Expects:
          item = {
            "link": "...",
            "text": "...",
            "previous_pred": "4"    # string of the 1st-pass label
          }
        grader_template is your {GRADER_PROMPT} block, containing:
          {{ CLASSIFICATION_PROMPT }}  and  {{ item.text }}  and  {{ item.previous_pred }}
        classification_prompt is the original instructions to be inlined.
        Returns:
          {
            "url": item["link"],
            "judge_score": <whatever your LLM returns as score>
          }
        """
        if item["text"] is None:

            return {"url": item["link"], "judge_score": None}
        
        # Fill in the template
        print("item_ticket")
        print(item)
        prompt = (
            grader_template
            .replace("{{CLASSIFICATION_PROMPT}}", classification_prompt)
            .replace("{{item.ticket_text}}", item["text"])
            .replace("{{item.previous_pred}}", item["previous_pred"])
        )
        print("ticket_corrected!!")

        async with self.sem, self.rpm_limiter:
            resp = await self.async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                ],
                # temperature=0,
                seed = 0
                # assume grader returns a JSON or scalar you can parse
            )

        # Here you parse resp into a judge_score. For example:
        content = resp.choices[0].message.content.strip() 
        # print('content!!')
        print(content)
        # e.g. content might be "CORRECT" or "INCORRECT" or a probability
        # You can customize how you extract numeric score or label
        return {"url": item["link"], "judge_score": int(content)}


    async def run_grading(self, items: list[dict], model: str, grader_template: str, classification_prompt: str,) -> list[dict]:
        """
        items: [
          {"link":..., "text":..., "previous_pred": "..."},
          ...
        ]
        returns: [
          {"url":..., "judge_score": ...},
          ...
        ]
        in the same order as items.
        """
        # 1) filter out None-texts if you want (optional)
        filtered = [it for it in items if it["text"] is not None]
        none_items = [it for it in items if it["text"] is None]

        # 2) launch grade_text_async in parallel
        tasks = [
            self.grade_text_async(
                it, self.sem, model, grader_template, classification_prompt
            )
            for it in filtered
        ]
        graded = await asyncio.gather(*tasks)

        # 3) append back any None-text items
        for it in none_items:
            graded.append({"url": it["link"], "judge_score": None})

        # 4) reorder to match original items[]
        url_map = {r["url"]: r for r in graded}

        return [url_map[it["link"]] for it in items]
    
        
    def excel_to_eval_jsonl(self,excel_path: str, output_path: str, sheet_name=None):
        """
        Reads an Excel file and writes a JSONL for evals.
        If sheet_name is provided, only that sheet is processed; otherwise all sheets.
        Expects each sheet to have columns 'text' and 'groundtruth'.
        """
        # Read one or all sheets
        sheets = pd.read_excel(excel_path, sheet_name=sheet_name) 
        # Normalize to dict of DataFrames
        if isinstance(sheets, pd.DataFrame):
            sheets = {sheet_name or 'Sheet1': sheets}

        total = 0
        sample_id = 0 # Later in the day, evals doesn't preserve the order thus, attach sample id to later use it as a unique id
        with open(output_path, 'w', encoding='utf-8') as f:
            for name, df in sheets.items():
                # Skip sheets without necessary columns
                if not {'text', 'groundtruth'}.issubset(df.columns):
                    print(f"[WARNING] Sheet '{name}' missing 'text' or 'groundtruth', skipping.")
                    continue

                for _, row in df.iterrows():
                    text = str(row['text']).strip()
                    label = str(row['groundtruth']).strip()
                    entry = {
                        "item":{
                        "ticket_text": text,
                        "correct_label": label,
                        "sample_id" : sample_id,
                    }
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total += 1
                    sample_id += 1

        print(f"[INFO] Wrote {total} entries to {output_path}")

   


if __name__ == "__main__":
    
    print("hi")
    
    
    
    
   