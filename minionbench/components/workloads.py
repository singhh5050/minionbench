import tiktoken
from typing import List, Dict, Optional

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

class Workload:
    def __init__(self):
        # Use cl100k_base tokenizer (GPT-4 tokenizer) for consistent token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 4 characters per token
            return len(text) // 4
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens"""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self.tokenizer.decode(tokens[:max_tokens])
        else:
            # Rough approximation
            max_chars = max_tokens * 4
            return text[:max_chars] if len(text) > max_chars else text
    
    def get_data(self, limit=10) -> List[str]:
        """Return benchmark data for the workload"""
        pass

class Prefill(Workload):
    """Finance Bench workload focused on prefill (1200 input tokens)"""
    
    def get_data(self, limit=10) -> List[str]:
        """Return finance-related test prompts with heavy context"""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        try:
            ds = load_dataset("PatronusAI/financebench", split="train")
            prompts = []
            
            for i, example in enumerate(ds.select(range(min(limit, len(ds))))):
                context = example.get("context", "")
                question = example.get("question", "")
                
                # Calculate available space for context
                question_part = f"Question: {question}\nAnswer:"
                question_tokens = self.count_tokens(question_part)
                available_context_tokens = 1200 - question_tokens - 20  # 20 token buffer
                
                if available_context_tokens > 0:
                    context = self.truncate_to_tokens(context, available_context_tokens)
                
                prompt = f"{context}\n\n{question_part}"
                prompts.append(prompt)
                
            return prompts
            
        except Exception as e:
            raise RuntimeError(f"Error loading FinanceBench dataset: {e}")

class Balanced(Workload):
    """OpenAssistant balanced workload (750 input tokens)"""
    
    def get_data(self, limit=10) -> List[str]:
        """Return balanced test prompts"""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        try:
            ds = load_dataset("OpenAssistant/oasst1", split="train")
            
            # Filter for human prompter messages
            prompter_messages = [ex for ex in ds if ex["role"] == "prompter"]
            
            prompts = []
            for i, example in enumerate(prompter_messages[:limit]):
                prompt = example["text"]
                # Truncate to ~750 tokens
                prompt = self.truncate_to_tokens(prompt, 750)
                prompts.append(prompt)
                
            return prompts
            
        except Exception as e:
            raise RuntimeError(f"Error loading OpenAssistant dataset: {e}")

class Decode(Workload):
    """Writing/Math decode workload (300 input tokens)"""
    
    def get_data(self, limit=10) -> List[str]:
        """Return writing/math test prompts focused on generation"""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
            
            prompts = []
            for i, example in enumerate(ds.select(range(min(limit, len(ds))))):
                problem = example["problem"]
                
                # Keep problem short (~300 tokens)
                problem = self.truncate_to_tokens(problem, 280)  # Leave room for instruction
                prompt = f"Solve this step by step with detailed reasoning:\n\n{problem}"
                
                prompts.append(prompt)
                
            return prompts
            
        except Exception as e:
            raise RuntimeError(f"Error loading math dataset: {e}") 