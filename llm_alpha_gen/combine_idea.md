# Brain Alpha Development Guide  
## Typical Structures of Submitted Alphas and a Methodology for Developing New Alphas

This document summarizes the **structural commonalities** observed in alphas that are actually submitted to and pass on the Brain platform.  
Based on these observations, it presents a **general way of thinking and a practical methodology** for designing new alphas.

---

## 1. Typical Structural Pattern of Brain Alphas

Alphas that pass on Brain tend to follow the same **conceptual flow**, regardless of the specific operators used.

Source of information  
→ Extraction of change and context  
→ Relative comparison  
→ Bias removal and stabilization  

This flow appears consistently across almost all successful alphas.

---

### 1.1 Source of Information (What)
An alpha typically draws from one or more of the following information sources:

- Corporate fundamentals  
- Price movements in the market  
- Investor positioning and flow data  
- Changes in analyst expectations  
- Style factors (e.g., value, momentum)

The key point is that the focus is **not on absolute levels**,  
but on **how these quantities are changing**.

---

### 1.2 Extraction of Change and Context (When & How)
Raw information is rarely used directly.  
Instead, it is **reinterpreted within a temporal context**.

Common guiding questions include:

- How different is the current value from its usual level?
- Is the change occurring rapidly or gradually?
- Should recent information matter more than older observations?

As a result, most alphas encode not the level itself, but the **direction, speed, and abnormality of change**.

---

### 1.3 Relative Comparison (Compared to What)
Brain alphas emphasize **relative positioning across securities** rather than absolute magnitude.

This approach is preferred because it:
- Removes scale differences across assets  
- Is more robust to missing data and noise  
- Adapts better to changing market environments  

Ultimately, an alpha answers the question:  
*“Where does this asset stand relative to others?”*

---

### 1.4 Bias Removal and Stabilization (Risk Control)
The final stage focuses on controlling **concentration and instability** in the signal.

Typical issues addressed at this stage include:
- Concentration in specific industries, countries, or themes  
- Excessive weight assigned to a small number of securities  
- Unstable performance driven by extreme values  

To mitigate these risks, alphas are typically designed to:
- Restrict comparisons within well-defined groups, and/or  
- Compress extreme values in a smooth and controlled manner  

---

## 2. Common Thought Patterns in Successful Alphas

Regardless of implementation details, alphas that pass on Brain tend to share the following **conceptual principles**:

1. They focus on **changes**, not levels  
2. They rely on **relative positioning** rather than absolute values  
3. They combine information with **different economic interpretations**  
4. They incorporate stability and diversification from the outset  
5. They avoid betting on a small number of securities  

---

## 3. A General Methodology for Developing New Alphas

### STEP 1. Start with a Question
Every alpha begins with a hypothesis-driven question, such as:

- Which changes tend to precede future performance?
- Where does the market systematically overreact or underreact?
- Under what conditions do two signals reinforce each other?

The starting point is **a hypothesis, not a formula**.

---

### STEP 2. Interpret the Information
Rather than using raw inputs directly, define the **conditions under which the information becomes meaningful**.

This often involves thinking in terms of:
- Magnitude of change  
- Speed of change  
- Abnormality relative to history  
- Interaction with other information sources  

---

### STEP 3. Ensure Comparability
An alpha must always be able to answer the question,  
*“Which asset is stronger relative to others?”*

This requires a structure that supports comparison:
- Across securities  
- Across time  
- Across different market regimes  

Relative comparison should therefore be a core design principle.

---

### STEP 4. Actively Remove Bias
Continuously check whether the alpha is inadvertently capturing:
- Industry-specific effects  
- Country or regional exposures  
- Style-specific bets  
- Dominance by a small set of extreme observations  

Bias control should be **an explicit design choice**, not a post-hoc adjustment.

---

### STEP 5. Create Variants, Not Single Alphas
Alpha development is not about producing a single final expression,  
but about creating a **family of related variants**.

Typical variations include:
- Different observation windows  
- Different comparison groups  
- Different signal strengths or sensitivities  

This process is essential for improving robustness, managing correlation, and ensuring long-term stability.

---

## 4. Key Principles to Keep in Mind During Alpha Development

### Recommended
- Change-focused thinking  
- Relative comparison frameworks  
- Emphasis on diversification and stability  
- Combination of heterogeneous information sources  

### To Avoid
- Direct use of raw values  
- All-in bets on a single factor  
- Structures that concentrate on a few securities  
- Stabilization applied only as an afterthought  

---

## 5. Summary

**Successful alphas on Brain are not defined by clever use of specific operators,  
but by a way of thinking that interprets change, compares assets relatively,  
and is designed from the outset to avoid concentration and instability.**

Developing new alphas is the process of repeatedly applying this thinking  
to new data, new hypotheses, and new market contexts.
