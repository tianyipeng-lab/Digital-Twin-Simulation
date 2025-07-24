# Digital Twin Simulation Dataset - Documentation

## Introduction

This manual describes the hierarchical structure of the `persona_json` file that stores each individual response in the study. The structure is organized into two primary levels: **blocks**, and within each block, a set of **questions and corresponding answers**.

The survey is conducted across **four distinct waves**, and each block is associated with one of these waves. Blocks serve as thematic groupings of questions, which may vary in number and content. Blocks and questions can be presented in a fixed or randomized order. In some cases, blocks or individual questions may be randomly selected for inclusion based on experimental conditions or display logic (For the json file, this is especially true for the blocks in the fourth wave).

This manual provides a comprehensive listing of all possible question blocks and their contents. However, the actual set of questions encountered by each participant (or digital twin) may differ due to such randomization and conditional display mechanisms.

## Structure of the `persona_json` file

### Element 0:
```{toctree}
:maxdepth: 1

blocks/wave_1/Demographics_wave_1
```

### Element 1:
```{toctree}
:maxdepth: 1

blocks/wave_1/Personality_wave_1
```

### Element 2:
```{toctree}
:maxdepth: 1

blocks/wave_1/Cognitive_tests_wave_1
```

### Element 3:
```{toctree}
:maxdepth: 1

blocks/wave_1/Economic_preferences___intro_wave_1
```

### Element 4:
```{toctree}
:maxdepth: 1

blocks/wave_1/Economic_preferences_wave_1
```

### Element 5:
```{toctree}
:maxdepth: 1

blocks/wave_2/Personality_wave_2
```

### Element 6:
```{toctree}
:maxdepth: 1

blocks/wave_2/Cognitive_tests_wave_2
```

### Element 7:
```{toctree}
:maxdepth: 1

blocks/wave_2/Forward_Flow_wave_2
```

### Element 8:
```{toctree}
:maxdepth: 1

blocks/wave_2/Economic_preferences___intro_wave_2
```

### Element 9:
```{toctree}
:maxdepth: 1

blocks/wave_2/Economic_preferences_wave_2
```

### Element 10:
```{toctree}
:maxdepth: 1

blocks/wave_3/Personality_wave_3
```

### Element 11:
```{toctree}
:maxdepth: 1

blocks/wave_3/Cognitive_tests_wave_3
```

### Element 12:
```{toctree}
:maxdepth: 1

blocks/wave_3/Economic_preferences_wave_3
```

### Element 13:
```{toctree}
:maxdepth: 1

blocks/wave_4/False_consensus_wave_4
```

### Element 14:
:::{dropdown} Randomization Group of 2 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Base_rate_30_engineers_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Base_rate_70_engineers_wave_4
```


:::

### Element 15:
:::{dropdown} Randomization Group of 2 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Disease___gain_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Disease_loss_wave_4
```


:::

### Element 16:
:::{dropdown} Randomization Group of 2 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Linda__no_conjunction_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Linda_conjunction_wave_4
```


:::

### Element 17:
:::{dropdown} Randomization Group of 2 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Outcome_bias___failure_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Outcome_bias___success_wave_4
```


:::

### Element 18:
:::{dropdown} Randomization Group of 2 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Anchoring___African_countries_high_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Anchoring___African_countries_low_wave_4
```


:::

### Element 19:
:::{dropdown} Randomization Group of 2 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Anchoring___redwood_high_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Anchoring___redwood_low_wave_4
```


:::

### Element 20:
:::{dropdown} Randomization Group of 3 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Less_is_More_Gamble_A_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Less_is_More_Gamble_B_wave_4
```

Variation 3 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Less_is_More_Gamble_C_wave_4
```


:::

### Element 21:
:::{dropdown} Randomization Group of 3 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Proportion_dominance_1A_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Proportion_dominance_1B_wave_4
```

Variation 3 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Proportion_dominance_1C_wave_4
```


:::

### Element 22:
:::{dropdown} Randomization Group of 3 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Proportion_dominance_2A_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Proportion_dominance_2B_wave_4
```

Variation 3 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Proportion_dominance_2C_wave_4
```


:::

### Element 23:
:::{dropdown} Randomization Group of 2 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Sunk_cost___no_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Sunk_cost___yes_wave_4
```


:::

### Element 24:
:::{dropdown} Randomization Group of 2 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Absolute_vs__relative___calculator_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Absolute_vs__relative___jacket_wave_4
```


:::

### Element 25:
:::{dropdown} Randomization Group of 3 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/WTA_WTP_Thaler___WTP_noncertainty_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/WTA_WTP_Thaler_problem___WTA_certainty_wave_4
```

Variation 3 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/WTA_WTP_Thaler_problem___WTP_certainty_wave_4
```


:::

### Element 26:
:::{dropdown} Randomization Group of 2 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Allais_Form_1_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Allais_Form_2_wave_4
```


:::

### Element 27:
:::{dropdown} Randomization Group of 2 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Myside_Ford_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Myside_German_wave_4
```


:::

### Element 28:
:::{dropdown} Randomization Group of 2 blocks
:animate: fade-in
Variation 1 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Probability_matching_vs__maximizing___Problem_1_wave_4
```

Variation 2 of the Group:
```{toctree}
:maxdepth: 1

blocks/wave_4/Probability_matching_vs__maximizing___Problem_2_wave_4
```


:::

### Element 29:
```{toctree}
:maxdepth: 1

blocks/wave_4/Non_experimental_heuristics_and_biases_wave_4
```

### Element 30:
```{toctree}
:maxdepth: 1

blocks/wave_4/Product_Preferences___Pricing_wave_4
```


