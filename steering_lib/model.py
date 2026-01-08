import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .hooks import GestoreHooks
from .vectors import VettoreSteering

class ModelloSteerable:
    """
    Classe principale che avvolge il modello LLM e permette di eseguire operazioni di steering.
    Usa 'GestoreHooks' per manipolare le attivazioni e 'VettoreSteering' per i calcoli.
    """
    def __init__(self, model_name="gpt2", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda" and not torch.cuda.is_available():
            print("Attenzione: CUDA richiesto ma non disponibile (Torch not compiled with CUDA). Utilizzo CPU.")
            self.device = "cpu"
            
        print(f"Caricamento modello {model_name} su {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        # Inizializziamo il gestore degli hook
        self.gestore_hooks = GestoreHooks()

    def _get_layer_gpt2(self, layer_idx):
        """
        Recupera uno specifico blocco transformer.
        Nota: Questa implementazione è specifica per GPT-2. 
        Per Llama o altri modelli, il percorso 'transformer.h' cambia.
        """
        # La struttura di GPT-2 è model.transformer.h[indice files]
        return self.model.transformer.h[layer_idx]

    def estrai_vettore(self, testo_positivo, testo_negativo, layer_idx, normalize=True):
        """
        Calcola un vettore di steering basato sulla differenza tra due testi (o liste di testi).
        
        Args:
            testo_positivo (str | list[str]): Esempio/i del concetto verso cui vogliamo andare.
            testo_negativo (str | list[str]): Esempio/i del concetto opposto.
            layer_idx (int): Indice del layer da cui estrarre le attivazioni.
            normalize (bool): Se True, normalizza il vettore a lunghezza 1. Consigliato.
            
        Returns:
            torch.Tensor: Il vettore di steering calcolato.
        """
        # Helper interno per gestire input singolo o lista
        def _get_avg_activation(text_input):
            if isinstance(text_input, str):
                return self._esegui_e_cattura(text_input, layer_idx)
            elif isinstance(text_input, list):
                if not text_input:
                    raise ValueError("La lista degli input non può essere vuota")
                activations = [self._esegui_e_cattura(t, layer_idx) for t in text_input]
                # Stack su una nuova dimensione e calcola la media
                return torch.stack(activations).mean(dim=0)
            else:
                raise ValueError("Input deve essere stringa o lista di stringhe")

        # 1. Cattura attivazione Positiva (Media se è una lista)
        act_pos = _get_avg_activation(testo_positivo)
        
        # 2. Cattura attivazione Negativa (Media se è una lista)
        act_neg = _get_avg_activation(testo_negativo)
        
        # 3. Calcola differenza
        vettore = VettoreSteering.calcola_da_differenza(act_pos, act_neg)
        
        if normalize:
            vettore = torch.nn.functional.normalize(vettore, dim=-1)
            
        return vettore

    def _esegui_e_cattura(self, testo, layer_idx):
        """Helper interno per eseguire il modello e catturare l'attivazione a un layer."""
        self.gestore_hooks.rimuovi_tutti_hooks()
        
        layer = self._get_layer_gpt2(layer_idx)
        self.gestore_hooks.registra_hook_cattura(layer, layer_idx)
        
        inputs = self.tokenizer(testo, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)
            
        return self.gestore_hooks.get_attivazione(layer_idx)

    def genera(self, prompt, max_new_tokens=50, vettore_steering=None, layer_idx=None, moltiplicatore=0.0):
        """
        Genera testo a partire da un prompt, applicando opzionalmente lo steering.
        
        Args:
            prompt (str): Il testo iniziale.
            max_new_tokens (int): Lunghezza massima della generazione.
            vettore_steering (torch.Tensor, optional): Il vettore da iniettare.
            layer_idx (int, optional): Il layer dove iniettare il vettore.
            moltiplicatore (float): Forza dell'iniezione (positivo o negativo).
            
        Returns:
            str: Il testo generato decodificato.
        """
        self.gestore_hooks.rimuovi_tutti_hooks()
        
        # Se abbiamo parametri di steering, registriamo l'hook di modifica
        if vettore_steering is not None and layer_idx is not None and moltiplicatore != 0:
            layer = self._get_layer_gpt2(layer_idx)
            self.gestore_hooks.registra_hook_steering(layer, layer_idx, vettore_steering, moltiplicatore)

        # Generazione standard di Hugging Face
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            top_k=50, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
