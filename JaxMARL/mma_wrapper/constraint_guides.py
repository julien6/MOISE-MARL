import abc
from typing import List, Tuple, Any, Union

class BaseConstraintGuide(abc.ABC):
    """
    Classe de base pour tous les Constraint Guides (RAG, RRG, GRG).
    Gère le mapping (label) des observations et actions,
    et le stockage optionnel d'un Trajectory-based Pattern (TP).
    """

    def __init__(self, obs_label_fn=None, act_label_fn=None, pattern=None):
        """
        :param obs_label_fn: fonction(obs_onehot) -> str
        :param act_label_fn: fonction(act_onehot) -> str
        :param pattern: structure ou objet décrivant un Trajectory-based Pattern
        """
        self.obs_label_fn = obs_label_fn
        self.act_label_fn = act_label_fn
        self.pattern = pattern

    @abc.abstractmethod
    def apply(self, history, observation, action=None, **kwargs):
        """
        Méthode principale que les classes enfants doivent implémenter.
        - history: structure de l'historique (liste de (obs, act), ou tout autre format)
        - observation: observation courante (one-hot)
        - action: action courante (optionnelle selon le type de guide)
        - kwargs: tout paramètre supplémentaire (mission_id, role_name, etc.)
        """
        raise NotImplementedError


class RoleActionGuide(BaseConstraintGuide):
    """
    RAG : Contrôle la distribution d’actions autorisées/pondérées pour un rôle donné.
    """

    @abc.abstractmethod
    def apply(self, history, observation, action=None, **kwargs):
        """
        Doit retourner une liste de couples (action_index, weight)
        pour indiquer quelles actions sont autorisées et avec quel poids.

        Exemple de retour:
          [(0, 0.0), (1, 1.0), (2, 1.0), (3, 0.5), (4, 0.0)]
        signifiant que l'action 0 est interdite (poids=0),
        l'action 1 autorisée (poids=1), etc.
        """
        pass


class RoleRewardGuide(BaseConstraintGuide):
    """
    RRG : Modifie la récompense (bonus/malus) selon la conformité d’un agent à son rôle.
    """

    @abc.abstractmethod
    def apply(self, history, observation, action=None, **kwargs) -> float:
        """
        Retourne un score (float) à ajouter (ou soustraire) à la récompense.

        On peut se baser sur:
          - l'historique pour détecter un pattern
          - l'observation courante
          - l'action courante
        par ex. un +5 si l'action = 'shoot' et le pattern 'has_ball' est respecté,
        ou un -2 si hors zone.
        """
        pass


class GoalRewardGuide(BaseConstraintGuide):
    """
    GRG : Modifie la récompense pour encourager l’atteinte d’un objectif (mission).
    """

    @abc.abstractmethod
    def apply(self, history, observation=None, action=None, **kwargs) -> float:
        """
        Retourne un bonus/malus (float) selon le pattern.
        Par ex., +10 si la séquence menant à "objectif atteint" est détectée,
        ou si le 'score' est réalisé, etc.
        """
        pass
