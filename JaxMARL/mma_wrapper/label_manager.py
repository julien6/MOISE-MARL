class LabelMapping:
    """
    Exemples de fonctions de mapping, ici sous forme d'appels polymorphes.
    Dans la pratique, vous pourriez stocker un dict ou autre structure
    pour traduire 'one-hot -> label' pour observations et actions.
    """
    def obs_to_label(self, obs_onehot) -> str:
        # TODO: Retourne un label (str) à partir d'un vecteur one-hot
        return "any"  # ou un vrai label si match

    def act_to_label(self, act_onehot) -> str:
        # TODO: Retourne un label (str) à partir d'un vecteur one-hot
        return "any"
