import argparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
import readline  # enables command history and editing

def load_model(hf_model=None, pt_file=None):
    if hf_model:
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        
        if pt_file:
            print(f"Loading model architecture from Hugging Face: {hf_model} with custom weights from: {pt_file}")
            # Load the .pt state dict
            state_dict = torch.load(pt_file, map_location="cpu")["state"]
            
            # Try to load with state_dict directly (if supported)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model, 
                    state_dict=state_dict,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            except TypeError:
                # Fallback to config + load_state_dict method
                config = AutoConfig.from_pretrained(hf_model)
                model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                model.load_state_dict(state_dict)
        else:
            print(f"Loading pretrained model from Hugging Face: {hf_model}")
            model = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    else:
        raise ValueError("You must specify --hf-model")

    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def main():
    parser = argparse.ArgumentParser(description="🧠 Vibe Check CLI")
    parser.add_argument("--hf-model", type=str, required=True, help="Name of Hugging Face model (e.g., gpt2, EleutherAI/pythia-2.8b)")
    parser.add_argument("--pt-file", type=str, help="Optional path to .pt model file with custom weights")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate per response")
    args = parser.parse_args()

    llm:transformers.pipelines.text_generation.TextGenerationPipeline = load_model(hf_model=args.hf_model, pt_file=args.pt_file)
    print("Loaded pipeline for text generation: ", type(llm))
    print("🌈 Welcome to the Vibe Check CLI. Type your question and press Enter.")
    print("Type `exit` or press Ctrl+C to quit.\n")

    try:
        while True:
            try:
                question = input("🧠 Human: ").strip()
                if question.lower() in {"exit", "quit"}:
                    print("👋 Bye!")
                    break

                # Format in Anthropic HH style
                formatted_prompt = f"\n\nHuman: {question}\n\nAssistant:"
                
                # outputs = llm(formatted_prompt, max_new_tokens=args.max_tokens, do_sample=True, temperature=0.7)
                outputs = llm(formatted_prompt, max_new_tokens=args.max_tokens, do_sample=False, temperature=0.0)
                # Extract just the assistant's response (everything after the formatted prompt)
                response = outputs[0]['generated_text'][len(formatted_prompt):].strip()
                print(f"🤖 Assistant: {response}\n")
            except KeyboardInterrupt:
                print("\n👋 Bye!")
                break
            except EOFError:
                print("\n👋 Bye!")
                break
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
