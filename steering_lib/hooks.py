import torch

class GestoreHooks:
    """
    Gestisce gli hook di PyTorch per intercettare e modificare le attivazioni del modello.
    Questa classe si occupa della parte di 'basso livello' dell'interazione con il modello.
    """
    def __init__(self):
        self._hooks_registrati = []
        self._attivazioni_catturate = {}
        
        # Stato per lo steering attivo
        self._vettore_steering_attivo = None
        self._layer_target_idx = None
        self._moltiplicatore = 0.0

    def cattura_attivazione(self, modulo, input, output, layer_idx):
        """
        Hook di callback eseguito durante il forward pass per SALVARE le attivazioni.
        
        Args:
            modulo: Il layer PyTorch.
            input: Tuple degli input al layer.
            output: L'output del layer (spesso una tupla in GPT-2).
            layer_idx: L'indice del layer corrente.
        """
        # In GPT-2, l'output è una tupla (hidden_state, present_key_value_states)
        # A noi interessa hidden_state, che è il primo elemento.
        if isinstance(output, tuple):
            hidden_state = output[0]
        else:
            hidden_state = output
            
        # Salviamo solo l'attivazione dell'ULTIMO token della sequenza.
        # Shape tipica: [batch_size, seq_len, hidden_dim]
        # Prendiamo: batch 0, ultimo token (-1), tutte le feature (:).
        self._attivazioni_catturate[layer_idx] = hidden_state[0, -1, :].detach().cpu()

    def inietta_steering(self, modulo, input, output, layer_idx):
        """
        Hook di callback eseguito durante il forward pass per MODIFICARE le attivazioni.
        Aggiunge il vettore di steering all'output del layer specificato.
        """
        # Se non è il layer giusto o il vettore non c'è, non facciamo nulla.
        if self._vettore_steering_attivo is None or layer_idx != self._layer_target_idx:
            return output

        if isinstance(output, tuple):
            hidden_state = output[0]
            rest = output[1:]
        else:
            hidden_state = output
            rest = ()

        # Calcoliamo la perturbazione: Vettore * Moltiplicatore
        perturbazione = self._vettore_steering_attivo.to(hidden_state.device) * self._moltiplicatore
        
        # Aggiungiamo la perturbazione all'hidden state originale
        # PyTorch gestisce il broadcasting se le dimensioni sono compatibili.
        hidden_state_modificato = hidden_state + perturbazione
        
        if isinstance(output, tuple):
            return (hidden_state_modificato,) + rest
        else:
            return hidden_state_modificato

    def registra_hook_cattura(self, layer, layer_idx):
        """
        Registra un hook su un layer specifico per CATTURARE le attivazioni.
        """
        handle = layer.register_forward_hook(
            lambda m, i, o: self.cattura_attivazione(m, i, o, layer_idx)
        )
        self._hooks_registrati.append(handle)

    def registra_hook_steering(self, layer, layer_idx, vettore, moltiplicatore):
        """
        Registra un hook su un layer specifico per INIETTARE il vettore di steering.
        """
        self._vettore_steering_attivo = vettore
        self._layer_target_idx = layer_idx
        self._moltiplicatore = moltiplicatore
        
        handle = layer.register_forward_hook(
            lambda m, i, o: self.inietta_steering(m, i, o, layer_idx)
        )
        self._hooks_registrati.append(handle)

    def rimuovi_tutti_hooks(self):
        """Rimuove tutti gli hook registrati e pulisce lo stato."""
        for handle in self._hooks_registrati:
            handle.remove()
        self._hooks_registrati = []
        self._attivazioni_catturate = {}
        self._vettore_steering_attivo = None
        self._layer_target_idx = None
        self._moltiplicatore = 0.0
        
    def get_attivazione(self, layer_idx):
        """Restituisce l'attivazione catturata per un dato layer."""
        return self._attivazioni_catturate.get(layer_idx)
