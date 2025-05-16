import os
import time
from typing import Dict, Optional, Union, Callable, List, Tuple
import google.generativeai as genai
from google.generativeai import types
import openai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryCallState
import httpx
import asyncio
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

# System instruction for Gemini
GEMINI_SYSTEM_INSTRUCTION = """You are an AI assistant. Your task is to answer the 'New Survey Question' as if you are the person described in the 'Persona Profile' (which consists of their past survey responses). 
Adhere to the persona by being consistent with their previous answers and stated characteristics. 
Follow all instructions provided for the new question carefully regarding the format of your answer."""

class VerificationFailedError(Exception):
    """Custom exception for when verification callback fails."""
    def __init__(self, message, prompt_id, llm_response_data=None):
        super().__init__(message)
        self.prompt_id = prompt_id
        self.llm_response_data = llm_response_data

class LLMConfig:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        max_retries: int = 10, # Max retries for the combined LLM call + Verification
        max_concurrent_requests: int = 5,
        verification_callback: Optional[Callable[..., bool]] = None,
        verification_callback_args: Optional[Dict] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_instruction = system_instruction or GEMINI_SYSTEM_INSTRUCTION
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
        self.verification_callback = verification_callback
        self.verification_callback_args = verification_callback_args if verification_callback_args is not None else {}

@retry(
    stop=stop_after_attempt(5), # Max 5 retries for the direct API call itself
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.APIError)),
    reraise=True
)
async def _get_openai_response_direct(prompt: str, config: LLMConfig) -> Dict[str, Union[str, Dict]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    async with httpx.AsyncClient(timeout=1000.0) as client:
        aclient = openai.AsyncOpenAI(api_key=api_key, http_client=client)
        messages = [{"role": "system", "content": config.system_instruction}, {"role": "user", "content": prompt}]
        response = await aclient.chat.completions.create(
            model=config.model_name, messages=messages, temperature=config.temperature, max_tokens=config.max_tokens
        )
        usage_details = {
            "prompt_token_count": response.usage.prompt_tokens,
            "completion_token_count": response.usage.completion_tokens,
            "total_token_count": response.usage.total_tokens
        }
        return {"response_text": response.choices[0].message.content, "usage_details": usage_details}

@retry(
    stop=stop_after_attempt(5), # Max 5 retries for the direct API call itself
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((
        ConnectionError, 
        TimeoutError 
    )), 
    reraise=True
)
async def _get_gemini_response_direct(prompt: str, config: LLMConfig) -> Dict[str, Union[str, Dict]]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    g_client = genai.Client(api_key=api_key)
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: g_client.models.generate_content(
            model=config.model_name, contents=[prompt],
            generation_config=types.GenerationConfig(temperature=config.temperature),
        )
    )

    # Check for blocked prompt or other non-fatal issues in the response directly
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        return {
            "error": f"Prompt blocked by API. Reason: {response.prompt_feedback.block_reason.name}",
            "response_text": "",
            "usage_details": {"prompt_token_count": response.usage_metadata.prompt_token_count if response.usage_metadata else 0}
        }
    # Check if candidates are empty or finished for a non-retryable reason
    if not response.candidates or (response.candidates[0].finish_reason not in [types.FinishReason.STOP, types.FinishReason.MAX_TOKENS]):
        finish_reason_name = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN_NO_CANDIDATES"
        # If it's due to safety or other non-retryable reasons, treat as an error to avoid infinite loops
        if finish_reason_name in ["SAFETY", "RECITATION", "OTHER"]:
             return {
                "error": f"Generation stopped for a non-retryable reason: {finish_reason_name}",
                "response_text": "",
                "usage_details": {"prompt_token_count": response.usage_metadata.prompt_token_count if response.usage_metadata else 0}
            }

    usage_metadata = response.usage_metadata
    current_token_stats = {
        "prompt_token_count": usage_metadata.prompt_token_count if usage_metadata else 0,
        "candidates_token_count": usage_metadata.candidates_token_count if hasattr(usage_metadata, 'candidates_token_count') else 0,
        "thoughts_token_count": usage_metadata.thoughts_token_count if hasattr(usage_metadata, 'thoughts_token_count') else 0,
        "total_token_count": usage_metadata.total_token_count if hasattr(usage_metadata, 'total_token_count') else 0
    }
    if hasattr(usage_metadata, 'prompt_tokens_details') and usage_metadata.prompt_tokens_details:
        current_token_stats["prompt_tokens_details"] = [
            {"modality": detail.modality.name if detail.modality else "UNKNOWN", "token_count": detail.token_count}
            for detail in usage_metadata.prompt_tokens_details
        ]
    return {"response_text": response.text, "usage_details": current_token_stats}


async def get_llm_response_with_internal_retry(
    prompt: str, config: LLMConfig, provider: str
) -> Dict[str, Union[str, Dict]]:
    try:
        if provider.lower() == "gemini":
            return await _get_gemini_response_direct(prompt, config)
        elif provider.lower() == "openai":
            return await _get_openai_response_direct(prompt, config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e: # Catch exceptions from the direct calls after their retries
        return {"error": f"LLM API call failed after internal retries: {str(e)}", "provider": provider}


async def _process_single_prompt_attempt_with_verification(
    prompt_id: str,
    prompt_text: str,
    config: LLMConfig,
    provider: str,
    semaphore: asyncio.Semaphore
):
    async with semaphore: # Manage concurrency for LLM calls
        last_exception_details = None
        for attempt in range(config.max_retries):
            llm_response_data = None
            try:
                # Step 1: Get LLM response (this now has its own tenacity retry)
                llm_response_data = await get_llm_response_with_internal_retry(prompt_text, config, provider)

                if "error" in llm_response_data and llm_response_data["error"]:
                    # print(f"LLM call failed for {prompt_id} (attempt {attempt + 1}/{config.max_retries}): {llm_response_data['error']}. This is a final LLM error.")
                    last_exception_details = llm_response_data # Store this as the last error
                    # No more retries in *this* loop if the LLM call itself reported a final error.
                    return prompt_id, llm_response_data 


                # Step 2: Perform verification if callback is provided
                if config.verification_callback:
                    # The callback needs original_prompt_content. We should pass it if needed by callback_args.
                    # For now, assuming callback_args includes what's needed or callback gets it.
                    # It's better if callback_args contains static info, and dynamic info like llm_response_data is passed directly.
                    
                    # The callback must be synchronous as it deals with file I/O and potentially CPU-bound tasks from postprocess_responses
                    verified = await asyncio.to_thread(
                        config.verification_callback,
                        prompt_id,
                        llm_response_data, # Pass the successful LLM response
                        prompt_text, # Pass original prompt text for saving
                        **config.verification_callback_args 
                    )
                    if not verified:
                        # print(f"Verification failed for {prompt_id} (LLM attempt {attempt + 1}/{config.max_retries}). Retrying entire sequence...")
                        last_exception_details = {"error": f"Verification failed on attempt {attempt + 1}", "prompt_id": prompt_id, "llm_response_data": llm_response_data}
                        if attempt == config.max_retries - 1:
                            return prompt_id, last_exception_details # Final verification failure
                        # Wait before retrying the whole sequence (LLM call + verification)
                        await asyncio.sleep(min(2 * 2 ** attempt, 30))  # Simple exponential backoff
                        continue # Go to next attempt in the outer loop
                
                # If LLM call successful and (no verification OR verification successful)
                return prompt_id, llm_response_data

            except Exception as e: # Catch unexpected exceptions during the attempt
                # print(f"Unexpected error for {prompt_id} (attempt {attempt + 1}/{config.max_retries}): {str(e)}. Retrying...")
                last_exception_details = {"error": f"Unexpected error: {str(e)}", "prompt_id": prompt_id}
                if attempt == config.max_retries - 1:
                    return prompt_id, last_exception_details # Final unexpected error
                await asyncio.sleep(min(2 * 2 ** attempt, 30))  # Simple exponential backoff
                continue
        
        # Fallback if loop finishes without returning (should ideally be caught by attempt == config.max_retries - 1 checks)
        return prompt_id, last_exception_details if last_exception_details else {"error": f"Exhausted all {config.max_retries} retries for {prompt_id} with no specific final error."}


async def process_prompts_batch(
    prompts: List[Tuple[str, str]],
    config: LLMConfig,
    provider: str = "gemini",
    desc: Optional[str] = "Processing LLM prompts and verifying"
) -> Dict[str, Dict[str, Union[str, Dict]]]:
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    results = {}
    
    tasks = [
        _process_single_prompt_attempt_with_verification(pid, p_text, config, provider, semaphore)
        for pid, p_text in prompts
    ]
    
    for future in tqdm_asyncio(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
        prompt_id, response_data = await future
        results[prompt_id] = response_data
            
    return results

# Example usage (remains similar, but LLMConfig now takes callback info)
if __name__ == "__main__":
    async def mock_verification_callback(prompt_id, llm_response_data, original_prompt_text, **kwargs):
        print(f"  (Mock Verify for {prompt_id}): Called with LLM response containing '{llm_response_data.get('response_text', '')[:30]}...'. Args: {kwargs}")
        # Simulate saving
        # print(f"  (Mock Verify for {prompt_id}): Saving LLM response and original prompt '{original_prompt_text[:30]}...' to disk...")
        # Simulate verification failure for specific pids for testing
        if prompt_id == "p2" and kwargs.get("fail_p2_verification_once", False):
            kwargs["fail_p2_verification_once"] = False # So it passes on retry
            print(f"  (Mock Verify for {prompt_id}): Verification FAILED (simulated).")
            return False
        print(f"  (Mock Verify for {prompt_id}): Verification PASSED (simulated).")
        return True

    async def main():
        # Test Gemini
        gemini_config = LLMConfig(
            model_name="models/gemini-1.5-flash-preview-0514",
            temperature=0.7, max_tokens=50, max_retries=3, max_concurrent_requests=2,
            verification_callback=mock_verification_callback,
            verification_callback_args={"path_info": "/dummy/path", "fail_p2_verification_once": True} # Custom args for callback
        )
        prompts = [
            ("p1", "What is the capital of France? Answer concisely."),
            ("p2", "What is 2+2? Answer concisely."), # This one will fail verification once
            ("p3", "Who wrote 'Hamlet'? Answer concisely."),
            ("p4", "This prompt might be blocked for safety reasons.") # Test potential blocking
        ]
        
        print("\nProcessing Gemini prompts with mock verification...")
        results_gemini = await process_prompts_batch(prompts, gemini_config, provider="gemini", desc="Gemini Calls+Verify")
        for pid, resp in results_gemini.items():
            if "error" in resp and resp["error"]:
                print(f"Gemini - Prompt {pid} FINAL ERROR: {resp['error']}")
                if resp.get("llm_response_data") and resp["llm_response_data"] != resp: # Avoid printing self if error object *is* llm_response_data
                    print(f"  LLM data at failure: {resp['llm_response_data']}")
            else:
                print(f"Gemini - Prompt {pid} FINAL OK. Response: '{resp.get('response_text', '')[:50]}...' Tokens: {resp.get('usage_details', {}).get('total_token_count')}")
        
        # Test OpenAI
        # Ensure OPENAI_API_KEY is set if you uncomment this
        # openai_api_key = os.environ.get("OPENAI_API_KEY")
        # if openai_api_key:
        #     openai_config = LLMConfig(
        #         model_name="gpt-3.5-turbo",
        #         temperature=0.7, max_tokens=50, max_retries=2, max_concurrent_requests=2,
        #         verification_callback=mock_verification_callback,
        #         verification_callback_args={"path_info": "/dummy/path/openai", "another_arg": "test_val"}
        #     )
        #     print("\nProcessing OpenAI prompts with mock verification...")
        #     results_openai = await process_prompts_batch(prompts[:2], openai_config, provider="openai", desc="OpenAI Calls+Verify")
        #     for pid, resp in results_openai.items():
        #         if "error" in resp:
        #             print(f"OpenAI - Prompt {pid} FINAL ERROR: {resp['error']}")
        #         else:
        #             print(f"OpenAI - Prompt {pid} FINAL OK. Response: '{resp.get('response_text', '')[:50]}...' Tokens: {resp.get('usage_details', {}).get('total_token_count')}")
        # else:
        #     print("\nSkipping OpenAI test as OPENAI_API_KEY is not set.")

    asyncio.run(main()) 