import sys
import os
import argparse
from tqdm import tqdm

import api_factory
import prompt_factory
import domain_utils

from dotenv import load_dotenv


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Query LLM models for domain credibility evaluation.")
    parser.add_argument("url_list_path", help="Path to CSV or parquet file with domain list")
    parser.add_argument("provider", help="API provider ('openai' or 'vllm')")
    parser.add_argument("model", help="Model name to query")
    parser.add_argument("output_root", help="Directory for output files")
    parser.add_argument(
        "--web-search",
        action="store_true",
        default=False,
        help="Enable web search for OpenAI API (only applies to OpenAI provider).",
    )
    parser.add_argument(
        "--domain-source",
        choices=["file", "credibench", "creditext"],
        default="file",
        help=(
            "Where to load domains from. "
            "Use 'file' to read url_list_path (default), "
            "'credibench' for the CrediBench dataset, or 'creditext' for CrediText DQR."
        ),
    )
    
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") if args.provider == "openai" else None
    
    # Create API client with web search option for OpenAI
    api_client = api_factory.create_api_client(
        args.provider,
        api_key,
        model_name=args.model,
        enable_web_search=args.web_search
    )
    
    if args.domain_source == "credibench":
        domains_to_query = domain_utils.get_domains_from_credibench(args.output_root)
    elif args.domain_source == "creditext":
        domains_to_query = domain_utils.get_domains_from_creditext(args.output_root)
    else:
        domains_to_query = domain_utils.get_domain_to_query(args.url_list_path, args.output_root)

    print(f"{len(domains_to_query)} domains to query")
    if args.web_search and args.provider == "openai":
        print("Web search is ENABLED")
    
    for domain in tqdm(domains_to_query):
        system_prompt = prompt_factory.SYS_BASE
        user_instruction = f"{prompt_factory.USER_INSTRUCTION.format(domain=domain)} {prompt_factory.USER_FORMAT}"

        result = api_client.query_model(args.model, system_prompt, user_instruction)
        with open(os.path.join(args.output_root, f"{domain}.txt"), "w") as f:
            f.write(result)
