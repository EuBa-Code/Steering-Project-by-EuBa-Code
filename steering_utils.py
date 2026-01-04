import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

class SteeringManager:
    def __init__(self, model_name="gpt2", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
            
        print(f"Loading model {model_name} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        # Storage for activations and hooks
        self._activations = {}
        self._hooks = []
        self._steering_vector = None
        self._target_layer_idx = None
        self._multiplier = 0.0

    def _get_layer(self, layer_idx):
        """
        Helper to get a specific transformer layer.
        This implementation is specific to GPT-2 structure (transformer.h).
        For other models (Llama, etc.), the path might differ.
        """
        # GPT-2 specific path
        return self.model.transformer.h[layer_idx]

    def _capture_hook(self, module, input, output, layer_idx):
        """
        Hook function to capture activations during the forward pass.
        'output' in GPT-2 block is usually a tuple (hidden_state, present).
        We want the hidden_state (index 0).
        """
        if isinstance(output, tuple):
            hidden_state = output[0]
        else:
            hidden_state = output
            
        # We store the activation of the last token
        # Shape: [batch_size, seq_len, hidden_dim]
        # We take the last token of the first sequence in batch for simplicity in extraction
        self._activations[layer_idx] = hidden_state[0, -1, :].detach().cpu()

    def _steering_hook(self, module, input, output, layer_idx):
        """
        Hook to modify activations during generation.
        Adds the steering vector to the hidden state.
        """
        if self._steering_vector is None or layer_idx != self._target_layer_idx:
            return output

        if isinstance(output, tuple):
            hidden_state = output[0]
            rest = output[1:]
        else:
            hidden_state = output
            rest = ()

        # Add the steering vector
        # self._steering_vector shape: [hidden_dim]
        # hidden_state shape: [batch, seq, hidden_dim]
        # We broadcast the vector to all tokens (or just the last one? usually all for consistent steering)
        
        steered_hidden = hidden_state + (self._steering_vector.to(hidden_state.device) * self._multiplier)
        
        if isinstance(output, tuple):
            return (steered_hidden,) + rest
        else:
            return steered_hidden

    def clear_hooks(self):
        """Remove all hooks to return model to normal."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._activations = {}
        self._target_layer_idx = None

    def get_activation(self, text, layer_idx):
        """
        Runs one forward pass and captures the activation at the specified layer
        for the last token of the input text.
        """
        self.clear_hooks()
        
        # Register capture hook
        layer = self._get_layer(layer_idx)
        hook_handle = layer.register_forward_hook(
            lambda m, i, o: self._capture_hook(m, i, o, layer_idx)
        )
        self._hooks.append(hook_handle)

        # Run model
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)
            
        return self._activations[layer_idx]

    def compute_steering_vector(self, positive_text, negative_text, layer_idx):
        """
        Computes the steering vector: Activation(Pos) - Activation(Neg)
        """
        pos_act = self.get_activation(positive_text, layer_idx)
        neg_act = self.get_activation(negative_text, layer_idx)
        
        self.steering_vector = pos_act - neg_act
        self.steering_layer = layer_idx
        return self.steering_vector

    def generate(self, prompt, max_new_tokens=20, steering_vector=None, layer_idx=None, multiplier=0.0):
        """
        Generates text. If a steering vector is provided (or stored), 
        it adds it to the activations at the specified layer.
        """
        self.clear_hooks()
        
        # Set up steering parameters
        self._steering_vector = steering_vector
        self._target_layer_idx = layer_idx
        self._multiplier = multiplier

        if steering_vector is not None and multiplier != 0:
            layer = self._get_layer(layer_idx)
            hook_handle = layer.register_forward_hook(
                lambda m, i, o: self._steering_hook(m, i, o, layer_idx)
            )
            self._hooks.append(hook_handle)

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            top_k=50, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
