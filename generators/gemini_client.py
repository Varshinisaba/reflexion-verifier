"""
Gemini client using the current `google-genai` SDK.

The older `google-generativeai` package (genai.GenerativeModel(...)) is
fully deprecated by Google — if you see a FutureWarning mentioning this,
you're on the old package. Install the current one instead:

    pip uninstall google-generativeai   # if you had the old one
    pip install google-genai
"""
import os
import time

_last_call_time = [0.0]
# Free-tier throttle; set GEMINI_MIN_INTERVAL=0 on a paid key for snappy live checks.
_MIN_INTERVAL = float(os.environ.get("GEMINI_MIN_INTERVAL", "13"))
_MAX_ATTEMPTS = int(os.environ.get("GEMINI_MAX_ATTEMPTS", "5"))

class GeminiModel:

    def __init__(self, model_name, api_key=None):
        from google import genai

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.name = model_name
        self.is_chat = False
        self.model_name = model_name
        self._client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        prompt,
        max_tokens=4096,
        stop_strs=None,
        temperature=0.0,
        num_comps=1
    ):
        elapsed = time.time() - _last_call_time[0]
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_call_time[0] = time.time()
        from google.genai import types

        last_error = None

        def _one_call():
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    
                ),
            )
            #print("RESPONSE:", response)
            #print("TEXT:", repr(response.text))
            text = response.text
            #print(f"\n[DEBUG raw Gemini output]\n{text!r}\n[end debug]\n")
            if not text:
                raise RuntimeError("Gemini returned no text. Check the response above.")
            
            if stop_strs:
                for stop in stop_strs:
                    if stop in text:
                        text = text.split(stop)[0]
            return text

        for attempt in range(_MAX_ATTEMPTS):
            try:
                if num_comps == 1:
                    return _one_call()
                return [_one_call() for _ in range(num_comps)]
            except Exception as e:
                last_error = e
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(min(60, 2 ** (attempt + 2)))  # 4s, 8s, 16s, 32s
        raise RuntimeError(
            f"Gemini call failed: {last_error}"
        )
