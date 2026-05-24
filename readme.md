# GEO_CLASS — Experimento Público de Geometría Oculta

Repositorio experimental del marco **GEO** (*Hidden Geometry Framework*) aplicado al crecimiento cosmológico mediante una versión modificada de **CLASS**.

---

## Introducción

Mi nombre es **Leonel Torreblanca**. Soy desarrollador de software especializado en sistemas *custom*, SQL, PL/SQL, JavaScript y Oracle APEX. Mi enfoque principal es el análisis lógico y la construcción de sistemas complejos, aunque gran parte de este proyecto nace de un interés personal por la matemática, la geometría estructural y el comportamiento físico.

### La génesis de GEO
GEO no pretende reemplazar modelos cosmológicos existentes ni erigirse como una teoría física completa. El proyecto nace de una intuición simple: **la naturaleza parece operar mediante estructuras sutiles, eficientes y geométricamente estables.**

La pregunta fundamental que guía este trabajo es:
> *¿Y si la naturaleza no utiliza el 100% de sus grados de libertad de forma activa?*
> *¿Y si parte de la estructura gravitacional observable fuera geométricamente complementaria?*

A partir de esta premisa, realicé un estudio observacional sobre la eficiencia, dualidad, proporciones geométricas, transferencia y crecimiento.

---

## Evolución del Proyecto

### 1. GDD — Geometría Dual Dinámica
Las primeras formulaciones se agruparon bajo **GDD**. La idea central era que la naturaleza opera mediante una dualidad entre una región geométricamente activa y otra complementaria.

Se definieron las primeras variables geométricas:
- $f_c$: Fracción activa.
- $f_{out}$: Fracción complementaria.
- $\eta$: Eficiencia geométrica.

Durante el estudio, surgieron proporciones consistentes como:
$$\frac{3}{4}, \quad \sqrt{\frac{3}{5}}, \quad \frac{\pi}{4}$$

### 2. GEO — Hidden Geometry Framework
Con el tiempo, el enfoque se trasladó a contrastar estas relaciones contra datos reales de cosmología observacional (como $S_8$, $f\sigma_8$, BAO, Pantheon+SH0ES y *weak lensing*), evolucionando hacia el marco **GEO**.

---

## GEO_CLASS: El Laboratorio Experimental

**GEO_CLASS** es el laboratorio principal del proyecto, construido sobre una modificación pública y reproducible de *CLASS*. Introduce un operador geométrico efectivo sobre el crecimiento perturbativo cosmológico.

### Parámetros Principales
- `geo_xi`: Fracción geométrica activa.
- `geo_mu`
- `geo_mode`

### Relaciones Operativas
- $f_{out} = 1 - geo\_xi$
- $\eta = geo\_xi^2$

**Comportamiento:**
* Cuando $geo\_xi = 1$: El comportamiento se aproxima al modelo $\Lambda$CDM estándar.
* Cuando $geo\_xi < 1$: La contribución efectiva al crecimiento observable se reduce.

---

## Estructura Experimental

El estudio **GEO_CLASS** se divide en cinco etapas diseñadas para responder preguntas específicas sobre la estructura geométrica en el crecimiento cosmológico. El flujo completo es:

text
GEO 01 → Prueba estadística principal
GEO 02 → Análisis de nodos geométricos
GEO 03 → Prueba fuerte de arquitectura
GEO 04 → Ley predictiva geométrica
GEO 05 → Puente geométrico final

Cada paso genera automáticamente:

* `resultados/csv/`
* `resultados/logs/`
* `docs/plots/`

Permitiendo la reproducibilidad completa del experimento.

---

## GEO 01 — Master Test

### Objetivo
El primer paso del estudio busca responder la pregunta más importante: **¿GEO_CLASS mejora el comportamiento observacional respecto a $\Lambda$CDM?**

Para esto, se comparan tres modelos sobre exactamente el mismo entorno cosmológico:
1. **$\Lambda$CDM**
2. **GEO_CLASS**
3. **FREE_MU**

### Qué compara
El análisis utiliza distintos bloques observacionales:
$f\sigma_8$, $S_8$, BAO, Pantheon+SH0ES, $r_d$, $\Omega_b h^2$, $\Omega_m h^2$ y $A_s$.

El objetivo no es optimizar únicamente un observable aislado, sino analizar si la geometría efectiva puede mantenerse estable frente a múltiples restricciones cosmológicas simultáneamente.

### ¿Qué es FREE_MU?
**FREE_MU** es un modelo comparativo interno. No representa una geometría explícita; representa un parámetro libre de crecimiento introducido únicamente para responder:

> *¿La mejora observacional de GEO_CLASS es realmente geométrica o simplemente cualquier parámetro libre mejora el ajuste?*

Esto es fundamental para evitar construir una narrativa geométrica artificial sobre una mejora estadística trivial.

### Deducción Experimental
El experimento GEO 01 busca observar tres posibilidades:

* **Escenario 1:** GEO_CLASS no mejora frente a $\Lambda$CDM.
  * *Resultado:* La hipótesis geométrica pierde plausibilidad.
* **Escenario 2:** FREE_MU destrona completamente a GEO_CLASS.
  * *Resultado:* La geometría observada probablemente sea accidental.
* **Escenario 3:** GEO_CLASS mejora significativamente y FREE_MU no obtiene ventaja fuerte sobre GEO_CLASS.
  * *Resultado:* La estructura geométrica gana plausibilidad física.

### Resultado Observado
En los tests de crecimiento y $S_8$, GEO_CLASS produce consistentemente:
* Reducción de $\sigma_8$ y $S_8$.
* Mejora en $\chi^2$, AIC y mejora o empate técnico en BIC.

Además, **FREE_MU no destrona significativamente a GEO_CLASS**, lo que sugiere que `FREE_MU` equivale a una reinterpretación libre del mismo efecto geométrico.

### Lectura física
La interpretación experimental de GEO_CLASS es: el crecimiento observable parece comportarse como si sólo una fracción geométrica efectiva del sector oscuro participara activamente. Esto **no** significa "menos materia oscura", sino **menor contribución efectiva al crecimiento observable**.

### Resultado conceptual
El paso GEO 01 establece algo fundamental: **la señal geométrica sobrevive al contraste observacional inicial**. Ese resultado habilita el resto del experimento.

### Outputs
Los resultados se almacenan en:
* `resultados/csv/`
* `resultados/logs/`
* `docs/plots/`

### Gráficos sugeridos
Imágenes generadas en `docs/plots/geo_01_master_test/`:
* `01_delta_bic.png`
* `02_s8_geo_class.png`
* `03_geo_xi.png`
* `04_f_out.png`
* `05_omega_growth.png`

## GEO 02 — Análisis de Nodos Geométricos

### Objetivo
El segundo paso del estudio intenta responder una pregunta fundamental:
> *¿Los valores obtenidos por GEO_CLASS son arbitrarios o tienden naturalmente hacia ciertos nodos geométricos?*

Tras el ajuste cosmológico inicial, aparecieron valores extremadamente similares entre distintos escenarios. La pregunta evolucionó: de "¿GEO_CLASS funciona?" a **"¿GEO_CLASS converge hacia geometrías específicas?"**

### Hipótesis Geométrica
La hipótesis central de GEO 02 es que ciertas estructuras geométricas actúan como **estados preferidos de eficiencia**. En lugar de mostrar cualquier valor continuo, el sistema tiende hacia proporciones específicas. Los principales candidatos estudiados son:

* $\frac{3}{4}$
* $\sqrt{\frac{3}{5}}$
* $\frac{\pi}{4}$
* $\frac{1}{\sqrt{2}}$
* $\frac{2}{\pi}$

Estas relaciones no son arbitrarias; aparecieron repetidamente durante el desarrollo del marco GEO y reaparecen constantemente dentro del ajuste cosmológico.

### Qué analiza GEO 02
El script toma los valores de `geo_xi` obtenidos en GEO 01 y calcula:
* Distancia absoluta y error relativo.
* RMSE (Root Mean Square Error) y MAE (Mean Absolute Error).
* Estabilidad geométrica.
* Proximidad a nodos candidatos.

El objetivo no es demostrar una identidad matemática exacta, sino verificar si existen **atractores geométricos naturales**.

### Concepto de Nodo Geométrico
Dentro de GEO, un nodo geométrico representa una **región de estabilidad efectiva** donde la geometría minimiza la tensión estructural. En términos simples: ciertas relaciones podrían ser más eficientes, estables, reproducibles o naturales para el crecimiento observable.

### Resultados Observados
El análisis reveló dos bandas dominantes:
1. **Banda base:** $\approx \frac{3}{4}$
2. **Banda nodo:** $\approx \sqrt{\frac{3}{5}}$

El comportamiento no fue aleatorio; distintos escenarios cosmológicos convergieron hacia estas regiones. El ranking final determinó que `best_global_geometry` $\approx \sqrt{\frac{3}{5}}$, mientras que $\frac{3}{4}$ se consolidó como la estructura base recurrente.

### Interpretación Física Experimental
La lectura experimental de GEO 02 es que el crecimiento cosmológico observable parece tender hacia estados geométricos preferidos. Esto **no significa** necesariamente que $\sqrt{\frac{3}{5}}$ sea una constante física fundamental; significa que la **geometría efectiva observada converge repetidamente hacia esa región**.

### Resultado Conceptual
GEO 02 cambia la naturaleza del experimento: tras este punto, GEO deja de ser únicamente un ajuste libre y comienza a comportarse como una **arquitectura geométrica emergente**.

### Outputs
Los resultados se almacenan en:
* `resultados/csv/`
* `resultados/logs/`
* `docs/plots/`

### Gráficos sugeridos
Imágenes generadas en `docs/plots/geo_02_geometric_node_analysis/`:
* `01_geo_xi_vs_candidates.png`
* `02_distance_candidates.png`
* `03_geo_xi_vs_f_out.png`
* `04_hist_geo_xi.png`
* `05_rmse_candidates.png`
---


## GEO 03 — Architecture Strong Test

### Objetivo
Después de detectar estructuras geométricas recurrentes en GEO 02, el paso siguiente fue crítico: 
> *¿Las geometrías observadas sobreviven cuando son forzadas explícitamente?*

Hasta este punto, `geo_xi` era un parámetro libre. GEO 03 cambia la lógica: el sistema fuerza arquitecturas geométricas específicas para analizar si mantienen la compatibilidad con los datos cosmológicos.

### Geometrías Probadas
El estudio fuerza los siguientes valores:
* `geo_xi = 1` ($\Lambda$CDM estándar)
* `geo_xi = 3/4` (Estructura base GEO)
* `geo_xi = \sqrt{3/5}` (Nodo geométrico principal)
* `geo_xi = \pi/4` (Estructura angular alternativa)

### ¿Qué intenta responder GEO 03?
El objetivo es separar dos posibilidades fundamentales:
1. **Posibilidad 1:** La mejora de GEO_CLASS depende totalmente de ajustar libremente `geo_xi` (las geometrías fijas fallarían rápidamente).
2. **Posibilidad 2:** Ciertas geometrías fijas sobreviven naturalmente (la estructura geométrica gana plausibilidad física).

### Qué compara GEO 03
Cada arquitectura fija se compara contra $\Lambda$CDM fijo, GEO_CLASS libre y FREE_MU, utilizando:
* **Observables:** Crecimiento, $S_8$, BAO, Pantheon+SH0ES.
* **Métricas:** $\chi^2$, AIC, BIC, $\mu_{eff}$, $\Omega_{growth}$, $S_8$.

### Resultado Observado
El resultado fue inesperadamente fuerte: **las arquitecturas geométricas NO colapsaron**. 
En varios escenarios, el nodo $\sqrt{3/5}$ sobrevivió extremadamente cerca del rendimiento de GEO_CLASS libre, mientras que $3/4$ continuó apareciendo como estructura base estable.

### Interpretación Física Experimental
La lectura experimental de GEO 03 es que el crecimiento cosmológico observable parece compatible con arquitecturas geométricas discretas. Esto no significa necesariamente que el universo esté cuantizado, pero sí sugiere que **ciertas proporciones actúan como estados efectivos preferidos**.

### El Nodo $\sqrt{3/5}$
Este caso es el más relevante del estudio porque:
* Reaparece desde GEO 02.
* Sobrevive en condiciones de rigidez en GEO 03.
* Mantiene un comportamiento estadístico robusto.
* Produce una geometría extremadamente estable.

En este punto del experimento, $\sqrt{3/5}$ deja de ser una coincidencia aislada y se comporta como un **nodo geométrico dominante**.

### Resultado Conceptual
GEO 03 transforma el problema: pasamos de una **optimización estadística** hacia una **estructura geométrica estable**. La señal no desaparece al eliminar grados de libertad, representando uno de los hitos más importantes del proyecto.

### Outputs
Los resultados se almacenan en:
* `resultados/csv/`
* `resultados/logs/`
* `docs/plots/`

### Gráficos sugeridos
Imágenes generadas en `docs/plots/geo_03_architecture_strong_test/`:
* `01_delta_bic_node_vs_free_geo.png`
* `02_delta_bic_base_vs_free_geo.png`
* `03_s8_node_vs_free_geo.png`
* `04_mu_eff_node_vs_free_geo.png`
* `05_best_fixed_vs_lcdm.png`

## GEO 04 — Ley Predictiva Geométrica

### Objetivo
Tras detectar estabilidad geométrica, convergencia hacia nodos y supervivencia de arquitecturas fijas en pasos anteriores, el siguiente paso fue intentar responder la pregunta fundamental del proyecto:
> *¿La geometría observada puede transformarse en una ley predictiva?*

Hasta GEO 03, el proyecto describía un comportamiento emergente; GEO 04 busca ir más allá, transformando la geometría en una **predicción cuantitativa directa**.

### Idea Central
Durante el desarrollo se consolidaron tres estructuras clave:
* $f_c$: Fracción activa geométrica.
* $\eta$: Eficiencia geométrica.
* $\mu_{eff}$: Acople efectivo del crecimiento.

Las observaciones mostraban una recurrencia constante de $f_c \approx \sqrt{3/5}$ y $\eta \approx 3/5$. La pregunta central pasó a ser: **¿Puede derivarse una relación matemática simple que reproduzca directamente la supresión observada?**

### La Ley Geométrica
GEO 04 introduce la relación:
$$R = \mu_{eff}^{1/3}$$

Donde $R$ representa la supresión relativa del crecimiento observable frente a $\Lambda$CDM. La hipótesis física es simple: si el crecimiento efectivo está controlado geométricamente, la respuesta observable debe escalar con una potencia efectiva del acople geométrico.

### ¿Qué analiza GEO 04?
El script compara la **predicción geométrica** contra la **respuesta real** obtenida desde *CLASS GEO* en diversos escenarios: crecimiento, *weak lensing*, BAO + SN y combinaciones cosmológicas, evaluando el error absoluto, relativo y el *prediction score*.

### Resultados Observados
El resultado fue extremadamente fuerte: la relación $R = \mu_{eff}^{1/3}$ reproduce la respuesta de *CLASS GEO* con una precisión notable:
* Error relativo: $< 0.1\%$
* *Prediction score*: $\approx 95\% - 99\%$

### Interpretación Física Experimental
La lectura experimental de GEO 04 es que la supresión del crecimiento observable parece seguir una ley geométrica simple. Aunque esto no implica necesariamente que $\mu_{eff}^{1/3}$ sea una ley física definitiva, demuestra que **la geometría observada deja de ser meramente descriptiva para producir predicciones cuantitativas reproducibles**.

### Relación con $\sqrt{3/5}$
El nodo $f_c = \sqrt{3/5}$ continúa reapareciendo naturalmente dentro de la ley predictiva. Esto logra conectar los nodos geométricos, la eficiencia, la supresión observable y la arquitectura estable dentro de una misma estructura matemática.

### Resultado Conceptual
GEO 04 cambia la naturaleza del proyecto: después de este paso, el marco GEO deja de ser una modificación cosmológica experimental y comienza a comportarse como una **arquitectura geométrica predictiva**. La aparición repetida de $\sqrt{3/5}$, $3/5$ y $\mu_{eff}^{1/3}$ no es una coincidencia, sino parte de una estructura coherente.

### Outputs
Los resultados se almacenan en:
* `resultados/csv/`
* `resultados/logs/`
* `docs/plots/`

### Gráficos sugeridos
Imágenes generadas en `docs/plots/geo_04_prediction_law_base/`:
* `01_R_pred_vs_CLASS.png`
* `02_S8_prediction.png`
* `03_relative_error.png`
* `04_growth_suppression.png`


## GEO 05 — Puente Geométrico Final

### Objetivo
Tras detectar estructuras, forzar arquitecturas y derivar leyes predictivas, el cierre del experimento busca responder: 
> *¿Todas las piezas del sistema forman una geometría coherente?*

El objetivo de GEO 05 no es solo medir ajustes cosmológicos, sino construir un puente geométrico completo entre crecimiento, eficiencia, dualidad y estructura observable.

### Idea Central
El estudio conecta cantidades clave como $f_c$ (fracción activa), $f_{out}$ (fracción complementaria), $\eta$ (eficiencia), $\mu_{eff}$ (acople efectivo) y $\Omega_{growth}$ (crecimiento efectivo) dentro de una arquitectura única. El crecimiento observable deja de interpretarse solo como densidad gravitacional para entenderse como una **transferencia geométrica efectiva**.

### Resultado Geométrico Global
El análisis final confirma la dominancia del nodo $\sqrt{3/5}$:
* `mean_fc` $\approx 0.7609$
* `mean_eta` $\approx 0.5792$
* `mean_mu_eff` $\approx 0.7989$
* `mean_Omega_growth` $\approx 0.2516$

Al comparar distancias, la relación `distance_to_node` resulta consistentemente menor que `distance_to_base` o `distance_to_pi4`, consolidando a $\sqrt{3/5}$ como el nodo geométrico más cercano al conjunto completo.

### Interpretación Física Experimental
El crecimiento observable parece comportarse como una transferencia geométrica efectiva. Esto no implica que la gravedad esté resuelta ni que GEO reemplace automáticamente a $\Lambda$CDM, pero demuestra que **la geometría observada forma una estructura reproducible, coherente y predictiva**.

El hecho de que $\sqrt{3/5}$ emerja, sobreviva y domine el puente final —sin haber sido impuesto inicialmente— es el resultado conceptual más robusto del experimento.

---

## Conclusión del Experimento
**GEO_CLASS** demuestra consistentemente:
1. Reducción de $S_8$ y del crecimiento observable.
2. Estabilidad geométrica y supervivencia de arquitecturas rígidas.
3. Existencia de una ley predictiva reproducible.
4. Coherencia en una arquitectura global.

Aunque no constituye una validación física definitiva, la geometría observada es suficientemente estable y predictiva para justificar una auditoría externa rigurosa. Este repositorio permite: reproducibilidad, falsación pública y la separación clara entre geometría observada y narrativa especulativa.

---

## Ejecución del Laboratorio
Todo el entorno experimental puede ejecutarse de forma secuencial:

bash
python scripts/geo_01_master_test.py
python scripts/geo_02_geometric_node_analysis.py
python scripts/geo_03_architecture_strong_test.py
python scripts/geo_04_prediction_law_base.py
python scripts/geo_05_final_geometry_bridge.py



# GEO_CLASS — Experimento Público de Geometría Oculta

Repositorio experimental del marco **GEO** (*Hidden Geometry Framework*) aplicado al crecimiento cosmológico mediante una versión modificada de **CLASS**.

---

## Introducción

Mi nombre es **Leonel Torreblanca**. Soy desarrollador de software especializado en sistemas *custom*, SQL, PL/SQL, JavaScript y Oracle APEX. Mi enfoque principal es el análisis lógico y la construcción de sistemas complejos, aunque gran parte de este proyecto nace de un interés personal por la matemática, la geometría estructural y el comportamiento físico.

### La génesis de GEO
GEO no pretende reemplazar modelos cosmológicos existentes ni erigirse como una teoría física completa. El proyecto nace de una intuición simple: **la naturaleza parece operar mediante estructuras sutiles, eficientes y geométricamente estables.**

La pregunta fundamental que guía este trabajo es:
> *¿Y si la naturaleza no utiliza el 100% de sus grados de libertad de forma activa?*
> *¿Y si parte de la estructura gravitacional observable fuera geométricamente complementaria?*

---

## Evolución del Proyecto

### 1. GDD — Geometría Dual Dinámica
Las primeras formulaciones se agruparon bajo **GDD**. La idea central era que la naturaleza opera mediante una dualidad entre una región geométricamente activa y otra complementaria.

### 2. GEO — Hidden Geometry Framework
Con el tiempo, el enfoque se trasladó a contrastar estas relaciones contra datos reales de cosmología observacional (como $S_8$, $f\sigma_8$, BAO, Pantheon+SH0ES y *weak lensing*), evolucionando hacia el marco **GEO**.

---

## GEO_CLASS: El Laboratorio Experimental

**GEO_CLASS** es el laboratorio principal del proyecto, construido sobre una modificación pública y reproducible de *CLASS*. Introduce un operador geométrico efectivo sobre el crecimiento perturbativo cosmológico.

### Parámetros Principales
- `geo_xi`: Fracción geométrica activa.
- `geo_mu`
- `geo_mode`

### Relaciones Operativas
- $f_{out} = 1 - geo\_xi$
- $\eta = geo\_xi^2$

---

## Estructura Experimental

El estudio **GEO_CLASS** se divide en cinco etapas diseñadas para responder preguntas específicas sobre la estructura geométrica en el crecimiento cosmológico. Cada etapa genera automáticamente los datos necesarios en `resultados/csv/`, `resultados/logs/` y `docs/plots/`.

### Etapas del Flujo
1. **GEO 01:** Prueba estadística principal.
2. **GEO 02:** Análisis de nodos geométricos.
3. **GEO 03:** Prueba fuerte de arquitectura.
4. **GEO 04:** Ley predictiva geométrica.
5. **GEO 05:** Puente geométrico final.

---

## Conclusión Final

GEO comenzó como una observación simple, nacida de una intuición geométrica: **la naturaleza parece operar mediante estructuras sutiles, eficientes y organizadas.**

Lo que empezó como una tendencia numérica evolucionó mediante un viaje conceptual claro que transita desde la observación, pasando por la relación geométrica y el operador efectivo, hasta consolidarse finalmente como un entorno experimental.

### Del Ajuste a la Predicción
El punto de inflexión del proyecto ocurrió cuando GEO dejó de describir observaciones para producir predicciones reproducibles. La relación $R = \mu_{eff}^{1/3}$ no solo ajustó los datos de *CLASS GEO*, sino que se consolidó como una ley geométrica efectiva capaz de predecir la supresión observable del crecimiento cosmológico.

### El Objetivo del Repositorio
Este repositorio **NO** existe para declarar una nueva cosmología ni para reemplazar a $\Lambda$CDM. Su objetivo es concreto y técnico:
* Preservar y organizar el experimento **GEO_CLASS**.
* Permitir auditoría externa y reproducción independiente.
* Separar los resultados observables de la interpretación especulativa.
* Ofrecer un entorno listo para la validación o falsación pública.

### Filosofía del Proyecto
Todo el marco GEO se sostiene sobre una premisa simple: **si una estructura es real, debe sobrevivir a una auditoría externa.** Por ello, todos los scripts, datos, gráficos y resultados son públicos, verificables y regenerables.

---

## Licencia
**MIT License.** El proyecto es completamente abierto para inspección, reproducción, auditoría y análisis independiente.

## Autor
**Leonel Torreblanca** — Proyecto GEO
