
org_spec = {
    "structural": { 
        "roles": {
            "coach": "// TP or custom logic...",
            "player": "// TP or custom logic...",
            "goalkeeper": "// TP or custom logic...",
            "back": "// TP or custom logic...",
            "middle": "// TP or custom logic...",
            "attacker": "// TP or custom logic...",
            "leader": "// TP or custom logic..."
        },
        "role_inheritance": {
            "soc": ["coach", "player"],
            "player": ["back", "middle", "attacker", "leader"],
            "back": ["goalkeeper"]
        },
        "groups": {
            "team": {
                "roles": ["coach"],
                "subgroups": ["defense", "attack"],
                "links_inter": [
                  {
                      "source": "coach",
                      "target": "player",
                      "type": "aut"
                  }
                ],
                "links_intra": [],
                "compat_intra": [],
                "compat_inter": [],
                "cardinality_roles": {
                    "coach": [1, 2]
                },
                "cardinality_subgroups": {
                    "defense": [1, 1],
                    "attack": [1, 1]
                }
            },
            "defense": {
                "roles": ["goalkeeper", "back", "leader"],
                "subgroups": [],

                "links_intra": [
                    {
                        "source": "goalkeeper",
                        "target": "back",
                        "type": "aut"
                    }
                ],
                "links_inter": [],
                "compat_intra": [
                    {"roleA": "leader", "roleB": "back"},
                    {"roleA": "leader", "roleB": "goalkeeper"}
                ],
                "compat_inter": [],
                "cardinality_roles": {
                    "goalkeeper": [1, 1],
                    "back": [3, 3],
                    "leader": [0, 1]
                },
                "cardinality_subgroups": {}
            },
            "attack": {
                "roles": ["attacker", "middle", "leader"],
                "subgroups": [],

                "links_intra": [],
                "links_inter": [
                    {
                        "source": "leader",
                        "target": "player",
                        "type": "aut"
                    }
                ],
                "compat_intra": [
                    {"roleA": "leader", "roleB": "middle"},
                    {"roleA": "leader", "roleB": "attacker"}
                ],
                "compat_inter": [],

                "cardinality_roles": {
                    "attacker": [3, 4],
                    "middle": [2, 3],
                    "leader": [0, 1]
                },
                "cardinality_subgroups": {}
            }
        }
    },
    "functional": {
        "goals": {
            "g0": "// TP or custom logic...",
            "g2": "// TP or custom logic...",
            "g3": "// TP or custom logic...",
            "g4": "// TP or custom logic...",
            "g6": "// TP or custom logic...",
            "g7": "// TP or custom logic...",
            "g8": "// TP or custom logic...",
            "g9": "// TP or custom logic...",
            "g11": "// TP or custom logic...",
            "g13": "// TP or custom logic...",
            "g14": "// TP or custom logic...",
            "g16": "// TP or custom logic...",
            "g17": "// TP or custom logic...",
            "g18": "// TP or custom logic...",
            "g19": "// TP or custom logic...",
            "g21": "// TP or custom logic...",
            "g22": "// TP or custom logic...",
            "g24": "// TP or custom logic...",
            "g25": "// TP or custom logic..."
        },
        "schemes": {
            "ScoreGoal": {
                "goals": ["g0", "g2", "g3", "g4", "g6", "g7", "g8", "g9", "g11", "g13", "g14", "g16", "g17", "g18", "g19", "g21", "g22", "g24", "g25"],
                "plans": {
                    "g0": {
                        "sequence": [["g2", 0.7], ["g3", 0.9], ["g4", 0.5]]},
                    "g2": {
                        "sequence": [["g6", 1], ["g9", 1]]},
                    "g9": {
                        "choice": [["g7", 1], ["g8", 1]]},
                    "g3": {
                        "sequence": [
                            {
                                "parallel": [["g13", 1], ["g14", 1]]
                            },
                            ["g11", 1]
                        ]
                    },
                    "g4": {
                        "choice": [["g24", 1], ["g25", 1]],
                        "g11": {"choice": [["g21", 1], ["g22", 1]]},
                        "g13": {"choice": [["g16", 1], ["g17", 1]]},
                        "g14": {"parallel": [["g18", 1], ["g19", 1]]}
                    }
                },
                "missions": {
                    "m1": ["g2", "g6", "g7", "g8", "g13"],
                    "m2": ["g13", "g16", "g21", "g24"],
                    "m3": ["g13", "g17", "g22", "g25"],
                    "m7": ["g0"]
                },
                "cardinality_missions": {
                    "m1": [1, 4],
                    "m2": [1, 1],
                    "m3": [1, 1],
                    "m7": [1, 1]
                }
            },
            "mission_preferences": [
                {"prefer": "m1", "over": "m2"},
                {"prefer": "m1", "over": "m3"}
            ]
        }},
    "deontic": {
        "obligations": [
            {
                "agents": ["agent_0", "agent_1"],
                "role": "goalkeeper",
                "missions": ["m1"],
                "time_constraint": "any"
            },
            {
                "role": "back",
                "missions": ["m1"],
                "time_constraint": "weekdays(14h-18h)"
            },
            {
                "role": "attacker",
                "missions": ["m2", "m3"],
                "time_constraint": "any"
            }
        ],
        "permissions": [
            {
                "role": "goalkeeper",
                "missions": ["m7"],
                "time_constraint": "any"
            },
            {
                "role": "coach",
                "missions": ["m7"],
                "time_constraint": "any"
            }
        ]
    }
}
