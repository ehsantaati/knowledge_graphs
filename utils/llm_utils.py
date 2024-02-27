from langchain_community.llms import LlamaCpp


def load_llamacpp_llm(
    llm_path: str, n_ctx: int = 4098, n_gpu_layers: int = 20) -> LlamaCpp:
  """Load an instance of an llm (gguf) in llamacpp  """
  n_gpu_layers = 20
  n_batch = 512
  n_ctx = n_ctx
  llm = LlamaCpp(
    model_path=llm_path,
    temperature=0.1,
    top_p=1,
    n_gpu_layers=n_gpu_layers,
    n_ctx=n_ctx,
    n_batch=n_batch)

  return llm
