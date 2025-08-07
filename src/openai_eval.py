from openai import OpenAI,AsyncOpenAI
import json
from settings import config 

data_source_config = {
    "type": "custom",
    "item_schema": {
        "type": "object",
        "properties": {
            "ticket_text": {"type": "string"},  # a string of text with the contents of support ticket
            "correct_label": {"type": "string"},  # a "ground truth" output that the model should match (human provided)
            "sample_id":{"type":"integer"}
        },
        "required": ["ticket_text", "correct_label","sample_id"],
    },
    "include_sample_schema": True,
}

testing_criteria = [
  {
    "type": "string_check",
    "name": "Exact match",
    "input":  "{{ sample.output_text }}",
    "operation": "eq",
    "reference": "{{ item.correct_label }}"
  }
]

data_source_config_judge = {
    "type": "custom",
    "item_schema": {
        "type": "object",
        "properties": {
            "ticket_text": {"type": "string"},
            "previous_pred":  {"type": "string"},
            "correct_label" :{"type":"string"},
             "sample_id":{"type":"integer"}
        },
        "required": ["ticket_text", "previous_pred","correct_label","sample_id"]
    },
    "include_sample_schema": True,
}



DATA_DIR = config("DATA_DIR")
PROMPT_DIR = config("PROMPT_DIR")
OPENAI_SECRET_KEY = config("OPENAI_SECRET")

class OpenAIEval:
    def __init__(self, eval_name="SEC_TEXT_EVALS"):
        self.client = OpenAI(api_key=OPENAI_SECRET_KEY, max_retries=5)
        self.async_client = AsyncOpenAI(api_key=OPENAI_SECRET_KEY, max_retries=5)
        self.eval_name = eval_name
        self.eval_id = None
        self.file_id = None

    def create_eval_obj(self, data_source_config = data_source_config, testing_criteria = testing_criteria):
        """
        Creates an eval object and sets self.eval_id.
        """
        print("▶️ Evals payload:\n", json.dumps({
            "name": self.eval_name,
            "data_source_config": data_source_config,
            "testing_criteria": testing_criteria
        }, indent=2))

        eval_obj = self.client.evals.create(
            name=self.eval_name,
            data_source_config=data_source_config,
            testing_criteria=testing_criteria
        )
        print("Eval created:", eval_obj)
        self.eval_id = eval_obj.id

        return self

    def upload_tickets(self, jsonl_path: str):
        """
        Uploads the JSONL file to the OpenAI files API for evals.
        """
        with open(jsonl_path, "rb") as f:
            file = self.client.files.create(file=f, purpose="evals")

        print("File uploaded:", file)
        self.file_id = file.id
        return self

    def run_eval(self, model: str, prompt_name : str, input_messages ):
        """
        Runs the eval with a single prompt-model combination.
        First pass, we have the groundtruth so we pass on the same prompt
        """
        if not self.eval_id or not self.file_id:
            raise ValueError("Must create eval and upload tickets first.")

        run = self.client.evals.runs.create(
            self.eval_id,
            name=f"Run_{model}-{prompt_name}",
            data_source={
                "type": "completions",
                "model": model,
                "input_messages": {
                    "type": "template",
                    "template":input_messages
                    # "template": [
                    #     {"role": "developer", "content": prompt},
                    #     {"role": "user", "content": "{{ item.ticket_text }}"},
                    # ],
                },
                "source": {"type": "file_id", "id": self.file_id},
            },
        )
        print("Eval run started:", run)
        
        return run
    
    def search_eval(self,eval_name,run_name):

        eval_resp = self.client.evals.list(limit = 100)
        evals = eval_resp.data
        eval_id = next((e.id for e in evals if e.name == eval_name), None)

        if eval_id is None:
            raise ValueError(f"No eval found with name '{eval_name}'")
        
        runs_resp = self.client.evals.runs.list(eval_id=eval_id, limit=100)
        runs = runs_resp.data  # list of RunListResponse
        run_id = next((r.id for r in runs if r.name == run_name), None)

        if run_id is None:
            raise ValueError(f"No run named '{run_name}' found under eval '{eval_name}'")
        
        # This runs in a separate object to first object creation thus is fine.
        self.eval_id = eval_id 
        self.run_id = run_id 

        return self
    
    def update_jsonl(self,output_path):
        
        items = list(self.client.evals.runs.output_items.list(eval_id=self.eval_id, run_id=self.run_id,))

        # 2) Sort by sample_id so you restore your original order
        items = sorted(items, key=lambda page: page.datasource_item["sample_id"])
        with open(output_path, "w", encoding="utf-8") as fout:
            for page in items:
                rec = {
                    'item':{
                    "ticket_text":   page.datasource_item["ticket_text"],
                    "correct_label": page.datasource_item["correct_label"],
                    "sample_id":     page.datasource_item["sample_id"],
                    "previous_pred":    page.sample.output[0].content # output generated by the prompt (if second_pass, previous_pred is my current pred)
                }
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return items
    


if __name__ == "__main__":
    
    eval_obj = OpenAIEval()
    JSONL_PATH = DATA_DIR / "eval_jsonl_pred.jsonl"

    base_fp = PROMPT_DIR / "GRADER_PROMPT.txt"
    template_fp = PROMPT_DIR / "GRADER_TEMPLATE_PROMPT.txt"

    base_prompt = base_fp.read_text(encoding='utf-8')
    template    = template_fp.read_text(encoding="utf-8")
    
    base_classification_fp = PROMPT_DIR / "CLASSIFICATION_PROMPT.txt"
    template_classification_fp = PROMPT_DIR / "CLASSIFICATION_TEMPLATE_PROMPT.txt"
    pos_shot_fp = PROMPT_DIR / "CLASSIFICATION_TEMPLATE_WITH_POS_EXAMPLES.txt"
    all_shot_fp = PROMPT_DIR / "CLASSIFICATION_TEMPLATE_WITH_ALL_EXAMPLES.txt"

    base_prompt = base_classification_fp.read_text(encoding='utf-8')
    template    = template_classification_fp.read_text(encoding="utf-8")
    pos_shot_template = pos_shot_fp.read_text(encoding="utf-8")
    all_shot_template = all_shot_fp.read_text(encoding="utf-8")

    CLASSIFICATION_BASE_PROMPT = template.replace("{CLASSIFICATION_PROMPT}", base_prompt)
    CLASSIFICATION_POS_SHOT_PROMPT = pos_shot_template.replace("{CLASSIFICATION_TEMPLATE_PROMPT}", CLASSIFICATION_BASE_PROMPT)
    CLASSIFICATION_ALL_SHOT_PROMPT = all_shot_template.replace("{CLASSIFICATION_TEMPLATE_POS_EXAMPLES_PROMPT}", CLASSIFICATION_POS_SHOT_PROMPT)

    GRADER_BASE_PROMPT = template
    GRADER_TEMPLATE_PROMPT = template.replace("{GRADER_PROMPT}", GRADER_BASE_PROMPT).replace("{CLASSIFICATION_PROMPT}",CLASSIFICATION_ALL_SHOT_PROMPT)

    grader_messages = [
  {"role":"developer", "content": GRADER_TEMPLATE_PROMPT},
  {"role":"user",      "content": "{{ item.ticket_text }}"},
  {"role":"user",      "content": "{{ item.previous_pred }}"}
]
    prompts = [('basic', GRADER_TEMPLATE_PROMPT)]
    test_models = ['o4-mini','gpt-4.1-mini']
    evals_manager = OpenAIEval()
    evals_manager.create_eval_obj(data_source_config_judge, testing_criteria) \
        .upload_tickets(JSONL_PATH)
    for model in test_models:
        for prompt_name,prompt_text in prompts:
            print(f"[RUNNING] {model}-{prompt_name} | hash={hash(prompt_text)}")
            try: 
                run = evals_manager.run_eval(model=model, prompt_name=prompt_name, input_messages=grader_messages)
                print("✅ Started", run.name, run.report_url)
            except Exception as e:
                print(f"❌ Failed {model}-{prompt_name}:", e)
     



    ### TICKET TEXT, PRED , GT ## FOR confusion matrix afterward but you need to know eval_id, and run_id // best 조합 eval_id, run_id 
    # temp = client.evals.runs.output_items.list(run_id= run_id, eval_id= eval_id)
    # for page in temp:
    #     print(page)
    #     print("results")
    #     print(page.datasource_item['ticket_text'])
    #     print(page.datasource_item['correct_label'])
    #     print(page.sample.output[0].content) # predicted label..
    #     break

    ### TICKET TEXT, PRED , GT ## END 




        







    ### RUNNING EVAL ## 


    # JSONL_PATH = DATA_DIR / "eval_jsonl.jsonl"

    # base_fp = PROMPT_DIR / "CLASSIFICATION_PROMPT.txt"
    # template_fp = PROMPT_DIR / "CLASSIFICATION_TEMPLATE_PROMPT.txt"
    # pos_shot_fp = PROMPT_DIR / "CLASSIFICATION_TEMPLATE_WITH_POS_EXAMPLES.txt"
    # all_shot_fp = PROMPT_DIR / "CLASSIFICATION_TEMPLATE_WITH_ALL_EXAMPLES.txt"

    # base_prompt = base_fp.read_text(encoding='utf-8')
    # template    = template_fp.read_text(encoding="utf-8")
    # pos_shot_template = pos_shot_fp.read_text(encoding="utf-8")
    # all_shot_template = all_shot_fp.read_text(encoding="utf-8")

    # CLASSIFICATION_BASE_PROMPT = template.replace("{CLASSIFICATION_PROMPT}", base_prompt)
    # CLASSIFICATION_POS_SHOT_PROMPT = pos_shot_template.replace("{CLASSIFICATION_TEMPLATE_PROMPT}", CLASSIFICATION_BASE_PROMPT)
    # CLASSIFICATION_ALL_SHOT_PROMPT = all_shot_template.replace("{CLASSIFICATION_TEMPLATE_POS_EXAMPLES_PROMPT}", CLASSIFICATION_POS_SHOT_PROMPT)

    # prompts = [('basic', CLASSIFICATION_BASE_PROMPT), ('pos_shot', CLASSIFICATION_POS_SHOT_PROMPT),('all_shot', CLASSIFICATION_ALL_SHOT_PROMPT)]
    # test_models = ['gpt-4.1-nano', 'gpt-4.1-mini','gpt-4o']
    # evals_manager = OpenAIEval(data_dir=DATA_DIR)
    # evals_manager.create_eval_obj(data_source_config, testing_criteria) \
    #     .upload_tickets(JSONL_PATH)
    # for model in test_models:
    #     for prompt_name,prompt_text in prompts:
    #         print(f"[RUNNING] {model}-{prompt_name} | hash={hash(prompt_text)}")
    #         try: 
    #             run = evals_manager.run_eval(model=model, prompt_name=prompt_name, prompt= prompt_text)
    #             print("✅ Started", run.name, run.report_url)
    #         except Exception as e:
    #             print(f"❌ Failed {model}-{prompt_name}:", e)
    
   ### RUNNING EVAL ENDS 
