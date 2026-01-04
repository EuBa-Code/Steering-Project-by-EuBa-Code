import torch

class VettoreSteering:
    """
    Classe che rappresenta un vettore di steering (attivazione).
    Gestisce le operazioni aritmetiche di base per calcolare la direzione di steering.
    """
    def __init__(self, vettore, strength=1.0):
        """
        Inizializza un oggetto VettoreSteering.
        
        Args:
            vettore (torch.Tensor): Il tensore che rappresenta la direzione di attivazione.
            strength (float): Un moltiplicatore di forza predefinito (opzionale).
        """
        self.vettore = vettore
        self.strength = strength

    @staticmethod
    def calcola_da_differenza(activations_pos, activations_neg):
        """
        Crea un vettore di steering calcolando la differenza tra due attivazioni.
        
        Formula: Vettore = Attivazione(Positivo) - Attivazione(Negativo)
        
        Args:
            activations_pos (torch.Tensor): Tensore delle attivazioni per il prompt positivo.
            activations_neg (torch.Tensor): Tensore delle attivazioni per il prompt negativo.
            
        Returns:
            torch.Tensor: Il vettore risultante dalla sottrazione.
        """
        # Assicuriamoci che siano sulla stessa device
        dev = activations_pos.device
        return activations_pos - activations_neg.to(dev)

    def to(self, device):
        """Sposta il tensore interno sul dispositivo specificato (CPU/GPU)."""
        self.vettore = self.vettore.to(device)
        return self
    
    def norm(self):
        """Restituisce la norma (lunghezza) del vettore."""
        return self.vettore.norm()

    def normalizza(self):
        """
        Normalizza il vettore rendendolo di lunghezza unitaria (norma = 1).
        Utile per avere un controllo consistente con il moltiplicatore 'forza'.
        """
        self.vettore = torch.nn.functional.normalize(self.vettore, dim=-1)
        return self
