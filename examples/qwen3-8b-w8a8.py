# isort: skip_file
import os

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams


def main():
    prompts = [
        "The president of the United States is Mr.",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=100, temperature=0.6, top_k=40, top_p=0.95)
    # Create an LLM.
    llm = LLM(
        model="/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-8B-W8A8",
              # enforce_eager=True,
              trust_remote_code=True,
            #   max_model_len=256,
              gpu_memory_utilization=0.7,
              quantization="ascend",
            #   block_size=64,
              )

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()
