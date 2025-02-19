import re

from typing import List, Tuple, Union, Optional, Literal
from label_manager import LabelMapping

class TPManager:
    """
    Gère la création de TP (Trajectory-based Patterns) depuis une expression string,
    et propose un LabelMapping (ou le reçoit en paramètre).
    """
    def __init__(self, label_mapping: Optional[LabelMapping] = None):
        self.label_mapping = label_mapping if label_mapping else LabelMapping()

    def create_tp(self, expression: str) -> TP:
        """
        Parse l'expression de type: "[[any,any](1,*)[o1|o2,a1|a2|a3](0,*)[o6,any](1,1)](1,1)"
        et génère une structure (pattern_struct) exploitable.
        """
        pattern_struct = self._parse_expression(expression)
        return TP(pattern_struct)

    def _parse_expression(self, expr: str):
        """
        Parse l'expression string pour en faire une structure Python
        (ex: tuples imbriqués décrivant les séquences et cardinalités).
        """
        # TODO: Implémenter un parseur plus sophistiqué
        # En attendant, on fait un placeholder
        #
        # On peut par exemple:
        # - Retirer les crochets
        # - Découper en sous-pattern
        # - Repérer (1,*) ou (0,1) etc.
        # - Extraire les listes d'obs [o1|o2] et d'actions [a1|a2|a3]
        #
        # Pour la démo, on renvoie juste une structure fictive
        return (
            (
              (["any"], ["any"]), (1, "*")
            ),
            (
              (["o1","o2"], ["a1","a2","a3"]), (0, "*")
            ),
            (
              (["o6"], ["any"]), (1, 1)
            )
        ), (1,1)


t = ((((["any"], ["any"]), (1,"*")), ((["o1","o2"], ["a1","a2","a3"]), (0,"*")), ((["o6"], ["any"]), (1,"*"))), (1,1))

