from . import llm_config as config
import sys


def chat_with_llm(prompt: str) -> str:
    """
    Unified function to call LLM API based on configuration.
    Supports Expected Parrot, OpenAI, and DeepSeek APIs.
    """

    model_name = config.DEFAULT_MODEL_NAME
    api_mode = config.API_MODE

    print(f"[LLM_API] Mode={api_mode}, Model={model_name}", flush=True)
    if api_mode == "expected_parrot":
        return chat_with_parrot(prompt, model_name)
    elif api_mode == "openai":
        return chat_with_openai(prompt, model_name)
    elif api_mode == "deepseek":
        return chat_with_deepseek(prompt, model_name)
    else:
        return f"Error: Unknown API mode: {api_mode}"


def chat_with_openai(prompt: str, model_name: str) -> str:
    """
    Stream from OpenAI Chat Completions API and return final visible text.
    Print delimiters only at the beginning: stdout prints ”[OUTPUT] ==========”; stderr prints ”[REASON] ==========”.
    The final return value contains only the visible content text, not reasoning.
    """
    try:
        import openai  # type: ignore

        api_key = str(config.API_CONFIG.get("openai_api_key", "")).strip()
        if not api_key:
            return "Error: OPENAI_API_KEY is empty. Set it in .env"

        print("[OpenAI] creating client ...", flush=True)
        client = openai.OpenAI(api_key=api_key)
        print("[OpenAI] sending streaming request ...", flush=True)
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=180,
            stream=True,
        )
        print("[OpenAI] streaming started", flush=True)
        full_text_parts: list[str] = []
        reasoning_started = False
        output_started = False
        try:
            for event in stream:
                if not event or not getattr(event, "choices", None):
                    continue
                delta = event.choices[0].delta
                if delta is None:
                    continue
                # Visible content increment (stdout)
                token = getattr(delta, "content", None)
                if token:
                    if not output_started:
                        print("\n\n[OUTPUT] ==================================================\n", flush=True)
                        output_started = True
                    full_text_parts.append(token)
                    print(token, end="", flush=True)
                # Inference content increment (stderr real-time)
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    if not reasoning_started:
                        print("\n[REASON] ==================================================\n", file=sys.stderr, flush=True)
                        reasoning_started = True
                    print(reasoning, file=sys.stderr, end="", flush=True)
        finally:
            print("\n[OpenAI] streaming completed", flush=True)
        return "".join(full_text_parts)
    except Exception as e:  # pragma: no cover - defensive catch for runtime issues
        return f"Error: {e}"


def chat_with_parrot(prompt: str, model_name: str) -> str:
    """
    Send a prompt to Expected Parrot (EDSL) and return plain text.

    Parameters:
        prompt: User input text
        model_name: Parrot model name (EDSL)

    Returns:
        Response text or an error string prefixed with 'Error:'
    """
    try:
        import os
        from edsl import Model, QuestionFreeText, Survey  # type: ignore

        ep_api_key = str(config.API_CONFIG.get("expected_parrot_api_key", "")).strip()
        if not ep_api_key:
            return "Error: EXPECTED_PARROT_API_KEY is empty. Set it in .env"
        os.environ["EXPECTED_PARROT_API_KEY"] = ep_api_key

        service_name = getattr(config, "SERVICE_NAME", "")
        model = Model(model_name, service_name=service_name) if service_name else Model(model_name)

        question = QuestionFreeText(
            question_name="llm_response",
            question_text=prompt,
        )
        survey = Survey(questions=[question])
        results = survey.by(model).run()

        answer = results[0].answer
        if isinstance(answer, dict) and "llm_response" in answer:
            return answer["llm_response"]
        if answer is None:
            return "Error: Received None response from model"
        return str(answer)
    except Exception as e:  # pragma: no cover - defensive catch for runtime issues
        return f"Error: {e}"


def chat_with_deepseek(prompt: str, model_name: str) -> str:
    """
    Stream from DeepSeek (OpenAI-compatible) and return final visible text.
    Print delimiters only at the beginning: stdout prints ”[OUTPUT] ==========”; stderr prints ”[REASON] ==========”.
    The final return value contains only the visible content text, not reasoning.
    """
    try:
        import openai  # type: ignore

        api_key = str(config.API_CONFIG.get("deepseek_api_key", "")).strip()
        if not api_key:
            return "Error: DEEPSEEK_API_KEY is empty. Set it in .env"

        print("[DeepSeek] creating client ...", flush=True)
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
        print("[DeepSeek] sending streaming request ...", flush=True)
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=180,
            stream=True,
        )
        print("[DeepSeek] streaming started", flush=True)
        full_text_parts: list[str] = []
        reasoning_started = False
        output_started = False
        try:
            for event in stream:
                if not event or not getattr(event, "choices", None):
                    continue
                delta = event.choices[0].delta
                if delta is None:
                    continue
                # Visible content increment (stdout)
                token = getattr(delta, "content", None)
                if token:
                    if not output_started:
                        print("\n\n[OUTPUT] ==================================================\n", flush=True)
                        output_started = True
                    full_text_parts.append(token)
                    print(token, end="", flush=True)
                # Inference content increment (stderr real-time)
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    if not reasoning_started:
                        print("\n[REASON] ==================================================\n", file=sys.stderr, flush=True)
                        reasoning_started = True
                    print(reasoning, file=sys.stderr, end="", flush=True)
        finally:
            print("\n[DeepSeek] streaming completed", flush=True)
        return "".join(full_text_parts)
    except Exception as e:  # pragma: no cover
        return f"Error: {e}"
