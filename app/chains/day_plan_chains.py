import json
from langchain_core.prompts import PromptTemplate
#from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_llm():
    # Lightweight instruction model
    model_name = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=gen)

def build_chain():
    schemas = [
        ResponseSchema(name="itinerary", description="List of rows with day, activity, approx_cost_egp"),
        ResponseSchema(name="notes", description="Brief notes about timing, transport, and local tips")
    ]
    parser = StructuredOutputParser.from_response_schemas(schemas)
    fi = parser.get_format_instructions()

    prompt = PromptTemplate(
        input_variables=["destination", "days", "daily_caps", "places_json", "format_instructions"],
        template=(
            "You are a concise travel planner for {destination}. "
            "Create a realistic day-by-day itinerary for {days} days using the provided places. "
            "Respect the daily budget caps per day (EGP): {daily_caps}. "
            "Return only structured data per instructions.\n\n"
            "{format_instructions}\n\n"
            "Places:\n{places_json}\n\n"
            "Rules:\n"
            "- Prioritize variety (history, culture, food, markets) and best_time.\n"
            "- Keep each day's total activity costs within or near the daily cap.\n"
            "- Use approximate costs from input; if missing, estimate modestly.\n"
            "- Distribute time across morning/afternoon/evening; avoid too many far trips in one day."
        )
    )
    llm = load_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain, parser
