import re

from typing import List, Tuple, Union, Optional, Literal

# La cardinalité peut être un entier (0,1,2...) ou "*" pour signifier infini
CardValue = Union[int, Literal["*"]]

# Un sous-ensemble possible d'observations et d'actions
OnePattern = Tuple[List[str], List[str]]
# exemple : (["any"], ["any"])
#           (["o1","o2"], ["a1","a2","a3"])

# Un sous-pattern inclut un OnePattern + une cardinalité (ex. (1,"*"))
SubPattern = Tuple[OnePattern, Tuple[CardValue, CardValue]]
# exemple : ((["any"], ["any"]), (1,"*"))

# Le pattern complet (TP) : séquence de SubPatterns + cardinalité globale
TP = Tuple[
    Tuple[SubPattern, ...],            # suite de sous-patterns
    Tuple[CardValue, CardValue]        # cardinalité globale
]
# exemple :
# (
#   (
#     ((["any"], ["any"]), (1, "*")),
#     ((["o1","o2"], ["a1","a2","a3"]), (0, "*")),
#     ((["o6"], ["any"]), (1, 1))
#   ),
#   (1,1)
# )


def parse_tp(expression: str) -> TP:
    """
    Parse une chaîne de caractères du type :
      [[any,any](1,*)[o1|o2,a1|a2|a3](0,*)[o6,any](1,1)](1,1)
    pour produire une structure Python :
      (
        (
          ((["any"], ["any"]), (1, "*")),
          ((["o1","o2"], ["a1","a2","a3"]), (0, "*")),
          ((["o6"], ["any"]), (1, 1))
        ),
        (1,1)
      )
    """

    # Nettoyer les espaces éventuels
    expr = expression.strip()

    # 1) Récupérer la cardinalité globale (seqMin, seqMax)
    #    On s'attend à avoir le pattern principal entre crochets, puis
    #    un suffixe de la forme (X, Y) indiquant la cardinalité globale.
    #    Ex. ... ](1,1)
    #    On va localiser la dernière ']' et extraire ce qui suit
    last_bracket_idx = expr.rfind(']')
    if last_bracket_idx == -1:
        raise ValueError("Expression invalide : pas de ']' trouvé.")

    global_card_part = expr[last_bracket_idx+1 : ].strip()  # ex: '(1,1)'
    subpatterns_part = expr[1 : last_bracket_idx].strip()   # ex: 'any,any](1,*)[...]'

    # parse la cardinalité globale
    global_min, global_max = parse_card_range(global_card_part)

    # 2) On parse la partie subpatterns_part qui peut contenir
    #    plusieurs sous-patterns de la forme :
    #    [o1|o2,a1|a2|a3](0,*)
    #    [any,any](1,*) etc.

    # On va itérer tant qu'on trouve des sous-patterns [ ... ](m,n)
    # On stocke les SubPattern dans une liste
    subpatterns_list = []

    # On consomme la string subpatterns_part
    remaining = subpatterns_part

    # Tant qu'on trouve un '['
    while True:
        remaining = remaining.strip()
        if not remaining:
            # plus rien à parser
            break

        if not remaining.startswith('['):
            # Soit c'est vide, soit expression mal formée
            # si c'est vide => c'est normal, sinon c'est une erreur
            break

        # 2.1) Extraire le contenu entre '[' et ']'
        close_idx = remaining.find(']')
        if close_idx == -1:
            raise ValueError("Sous-pattern mal formé : ']' manquant.")
        content_in_brackets = remaining[1:close_idx].strip()
        # content_in_brackets ex : 'any,any' ou 'o1|o2,a1|a2|a3'

        # Ensuite, on s'attend à quelque chose comme ](m,n) ...
        after_bracket = remaining[close_idx+1:].strip()
        if not after_bracket.startswith('('):
            raise ValueError("Sous-pattern mal formé : pas de '(' après ']'.")
        # ex: '(1,*)[...]'
        # On va extraire jusqu'au premier ')'
        paren_close = after_bracket.find(')')
        if paren_close == -1:
            raise ValueError("Sous-pattern mal formé : parenthèse fermante manquante.")
        card_str = after_bracket[:paren_close+1].strip()  # ex '(1,*)'
        # parse card range
        sp_min, sp_max = parse_card_range(card_str)

        # Récupérer la portion string pour la suite
        remaining = after_bracket[paren_close+1:].strip()

        # => On parse le content_in_brackets => obsList, actList
        #    On autorise la syntaxe [obs1|obs2, act1|act2|act3]
        obs_list, act_list = parse_obs_act_block(content_in_brackets)

        # On crée le SubPattern
        subp: SubPattern = ((obs_list, act_list), (sp_min, sp_max))
        subpatterns_list.append(subp)

    # Convertir en tuple immuable
    subpatterns_tuple: Tuple[SubPattern, ...] = tuple(subpatterns_list)

    # 3) Construire le TP final
    #    ((subpattern1, subpattern2, ...), (globalMin, globalMax))
    return (subpatterns_tuple, (global_min, global_max))

def parse_card_range(card_part: str) -> Tuple[CardValue, CardValue]:
    """
    Parse une chaîne du genre '(1,*)' ou '(0,1)' pour renvoyer (min, max).
    """
    s = card_part.strip()
    if not (s.startswith('(') and s.endswith(')')):
        raise ValueError(f"Cardinalité mal formée: {card_part}")
    inner = s[1:-1].strip()  # ex: '1,*'
    # split par ','
    parts = [p.strip() for p in inner.split(',')]
    if len(parts) != 2:
        raise ValueError(f"Cardinalité mal formée, attend (x,y): {card_part}")

    def parse_val(x: str) -> CardValue:
        if x == '*':
            return '*'
        return int(x)

    min_val = parse_val(parts[0])
    max_val = parse_val(parts[1])
    return (min_val, max_val)

def parse_obs_act_block(block: str) -> Tuple[List[str], List[str]]:
    """
    Parse un bloc du genre 'o1|o2,a1|a2|a3' ou 'any,any'
    => renvoie (["o1","o2"], ["a1","a2","a3"])
    On autorise '|' comme séparateur multiple.
    """
    # On cherche la virgule qui sépare la partie obs de la partie act
    # ex: 'any,any' => obs_part='any', act_part='any'
    #     'o1|o2,a1|a2|a3' => obs_part='o1|o2', act_part='a1|a2|a3'
    block = block.strip()
    comma_idx = block.find(',')
    if comma_idx == -1:
        raise ValueError(f"Pas de virgule pour séparer obs/actions: '{block}'")
    obs_part = block[:comma_idx].strip()
    act_part = block[comma_idx+1:].strip()

    # séparer par '|'
    obs_list = [o.strip() for o in obs_part.split('|')] if obs_part else []
    act_list = [a.strip() for a in act_part.split('|')] if act_part else []

    # si c'est vide => on peut fallback sur ["any"] ou c'est une erreur, selon votre logique
    if not obs_list:
        obs_list = ["any"]
    if not act_list:
        act_list = ["any"]

    return obs_list, act_list


# =================================================
#  Exemple de test
# =================================================

if __name__ == "__main__":
    expr = "[[o7,a4|a7](1,*)[o1|o2,a1|a2|a3](0,*)[o6,any](1,1)](1,1)"
    tp_parsed = parse_tp(expr)

    print("=== Expression:", expr)
    print("=== Résultat (structure Python):")
    from pprint import pprint
    pprint(tp_parsed)
    # Attendu :
    # (
    #   (
    #     ( (["any"], ["any"]), (1, "*") ),
    #     ( (["o1","o2"], ["a1","a2","a3"]), (0, "*") ),
    #     ( (["o6"], ["any"]), (1, 1) )
    #   ),
    #   (1,1)
    # )
