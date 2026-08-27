"""Carga e geração com o modelo ajustado, adaptando-se ao hardware da máquina.

O repositório publicado (`Grupo-de-Estudos-GPRSW/meditron-7b-finetuned-MedQuAD`) contém
apenas o **adapter LoRA** (~17 MB); os pesos vêm do modelo base `epfl-llm/meditron-7B`.
Este módulo carrega o base com a quantização adequada à GPU disponível — a mesma
`BitsAndBytesConfig` usada no treino (`finetuning/foundation_model.py`) — e aplica o
adapter por cima.

Toda degradação (queda de quantização, redução de tokens, ida para CPU) é registrada
em `self.notes` e vai para o JSON de resultados.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eval.config import EvalConfig, hardware_info, resolve_device, resolve_quantization

# O mesmo template usado no fine tuning (finetuning/dataset_formatting.py) e na
# inferência do pipeline (src/generation_nodes.py).
ALPACA_TEMPLATE = """###System:
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, don't share false information.

### User:
{question}

### Assistant:
"""

ASSISTANT_TAG = "### Assistant:"


@dataclass
class Generation:
    texto: str
    tokens_gerados: int
    parou_por_eos: bool
    segundos: float
    tokens_por_segundo: float


@dataclass
class ModelRunner:
    """Encapsula o modelo ajustado. Use `load()` antes de `generate()`."""

    config: EvalConfig
    device: str = ""
    quantization: str = ""
    notes: List[str] = field(default_factory=list)
    load_seconds: float = 0.0
    vram_pico_gb: Optional[float] = None
    model: Any = None
    tokenizer: Any = None

    # ------------------------------------------------------------------ #
    def describe(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "quantizacao": self.quantization,
            "segundos_para_carregar": round(self.load_seconds, 1),
            "vram_pico_gb": self.vram_pico_gb,
            "ajustes": list(self.notes),
        }

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as hf_logging

        # As barras de progresso do Hugging Face poluem o log da suíte; o progresso
        # relevante (item a item) é impresso pelas baterias.
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()

        hw = hardware_info()
        self.device = resolve_device(self.config, hw)
        self.quantization = resolve_quantization(self.config, self.device, hw)

        if self.config.quantization == "auto":
            self.notes.append(f"quantizacao escolhida automaticamente: {self.quantization}")
        if self.device == "cpu":
            self.notes.append("rodando em CPU: a geração é bem mais lenta")

        cache_dir = self.config.resolved_cache_dir()
        if cache_dir:
            self.notes.append(f"cache do Hugging Face: {cache_dir}")

        kwargs: Dict[str, Any] = {"cache_dir": cache_dir}
        if self.device == "cuda":
            kwargs["device_map"] = "cuda"
            kwargs["dtype"] = torch.float16
        else:
            kwargs["device_map"] = "cpu"
            kwargs["dtype"] = torch.float32

        quant_config = self._quantization_config()
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config

        started = time.perf_counter()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_id, cache_dir=cache_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.config.base_model_id, **kwargs)
        model = self._attach_adapter(model, cache_dir)
        model.eval()
        self.model = model

        self.load_seconds = time.perf_counter() - started
        if self.device == "cuda":
            self.vram_pico_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    def _quantization_config(self):
        if self.quantization == "none":
            return None
        import torch
        from transformers import BitsAndBytesConfig

        if self.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        # 4-bit: exatamente a configuração usada no treino
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    def _attach_adapter(self, model, cache_dir: Optional[str]):
        """Aplica o adapter LoRA publicado, com fallback para download manual."""
        from peft import PeftModel

        try:
            return PeftModel.from_pretrained(model, self.config.model_id, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"PeftModel.from_pretrained falhou ({type(exc).__name__}); "
                              "tentando baixar o adapter manualmente")
            from huggingface_hub import snapshot_download

            local = snapshot_download(
                repo_id=self.config.model_id,
                allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
                cache_dir=cache_dir,
            )
            return PeftModel.from_pretrained(model, local)

    # ------------------------------------------------------------------ #
    def generate(self, question: str, max_new_tokens: Optional[int] = None) -> Generation:
        """Geração determinística (greedy) com o template Alpaca do treino.

        Em caso de falta de memória, reduz `max_new_tokens` pela metade e tenta de novo
        (até duas vezes) antes de propagar o erro.
        """
        import torch

        budget = max_new_tokens or self.config.max_new_tokens
        prompt = ALPACA_TEMPLATE.format(question=question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        for attempt in range(3):
            try:
                started = time.perf_counter()
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=budget,
                        do_sample=False,                 # greedy: reprodutível entre execuções
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                elapsed = time.perf_counter() - started
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt == 2 or budget <= 32:
                    raise
                budget = max(32, budget // 2)
                self.notes.append(f"OOM na geração: max_new_tokens reduzido para {budget}")

        generated_ids = output[0][prompt_len:]
        parou_por_eos = bool(len(generated_ids)) and generated_ids[-1].item() == self.tokenizer.eos_token_id
        texto = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # o pipeline corta a resposta no marcador; aqui o prompt já foi removido pelo slice,
        # mas o modelo às vezes emite um novo turno "### User:" -- cortamos no primeiro.
        for marker in ("### User:", "###System:", ASSISTANT_TAG):
            if marker in texto:
                texto = texto.split(marker, 1)[0].strip()

        if self.device == "cuda":
            self.vram_pico_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)

        n_tokens = int(len(generated_ids))
        return Generation(
            texto=texto,
            tokens_gerados=n_tokens,
            parou_por_eos=parou_por_eos,
            segundos=round(elapsed, 2),
            tokens_por_segundo=round(n_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        )

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
