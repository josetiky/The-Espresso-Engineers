# Metodología y fundamentos matemáticos — Copa FutBotMX

> Capítulo Visión por Computadora · Categoría Amateur
> Pipeline: **Detección → Tracking → Homografía → Segmentación → Analítica**

Este documento explica, con ecuaciones, **cada etapa** del pipeline y por qué
funciona. La meta no es solo que el código corra, sino poder **explicar el
conjunto del entregable** (como pide la sección 3.1.3 de la convocatoria).

Las fórmulas usan notación LaTeX (GitHub las renderiza con `$...$` / `$$...$$`).

---

## 0. Notación

| Símbolo | Significado |
|---|---|
| $\mathbf{p}=(u,v)$ | punto en píxeles de la **cámara** |
| $\mathbf{p}'=(x,y)$ | punto en el **lienzo cenital** (vista de pájaro) |
| $\mathbf{P}=(X,Y)$ | posición en **metros** sobre el campo real |
| $H$ | matriz de homografía $3\times3$ |
| $f$ | índice de frame; $\text{fps}$ = cuadros por segundo |
| $\Delta t = 1/\text{fps}$ | duración de un frame |

---

## 1. Detección de objetos (YOLO)

Se usa un detector **YOLO entrenado con datos propios** (`futbotmx_v1.pt`,
clases `robot` y `balón`). YOLO divide la imagen en una rejilla y, por cada
celda, predice cajas $(b_x,b_y,b_w,b_h)$, una confianza $c$ y la clase. Se
conservan las detecciones con

$$c \ge \tau_{\text{clase}},\qquad \tau_{\text{robot}}=0.40,\quad \tau_{\text{balón}}=0.15 .$$

**Por qué dos umbrales.** El balón es pequeño y de baja resolución, por lo
que su confianza media es menor. Un único umbral alto (0.40) lo eliminaba casi
siempre — esa fue una de las causas de que las estadísticas del balón salieran
en cero. Bajar **solo** el umbral del balón recupera sus detecciones sin
ensuciar las de robots.

**Supresión de no-máximos (NMS).** Para eliminar cajas duplicadas se descartan
las que se solapan demasiado con otra de mayor confianza:

$$\text{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}\;\ge\;0.45 \;\Rightarrow\; \text{se elimina la de menor } c .$$

### 1.1 Respaldo del balón por color (HSV)

Cuando YOLO pierde el balón en un frame, se activa un detector clásico por
color. Se convierte el frame a HSV y se umbraliza el rango naranja:

$$M_{\text{balón}}(u,v)=\mathbb{1}\!\left[H\in[5,22]\,\wedge\,S\in[120,255]\,\wedge\,V\in[120,255]\right].$$

Tras una apertura morfológica para quitar ruido, se toma el **blob de mayor
área** dentro de la cancha como balón. Esto aporta robustez sin entrenamiento
adicional.

---

## 2. ROI dinámico de la cancha (la cámara es móvil)

Como la cámara se mueve, **no** se puede usar un polígono fijo. En cada frame se
recalcula la máscara verde del césped en HSV:

$$M_{\text{campo}}(u,v)=\mathbb{1}\!\left[H\in[35,85]\,\wedge\,S\ge40\,\wedge\,V\ge40\right],$$

seguida de cierre y apertura morfológicos (kernel $9\times9$) para tapar las
líneas blancas y quitar ruido del público. Las detecciones de **robots** cuyo
centro cae fuera de $M_{\text{campo}}$ se descartan (jueces, mesas, público).
El **balón nunca se descarta** por este filtro, porque legítimamente puede
estar sobre una línea blanca o encima de un robot.

---

## 3. Seguimiento (ByteTrack)

ByteTrack asocia detecciones entre frames para dar a cada objeto un
`tracker_id` persistente. Predice la posición siguiente con un **filtro de
Kalman** (modelo de velocidad constante) y empareja con la detección actual
mediante el costo

$$\text{costo}(i,j)=1-\text{IoU}\big(\hat{b}_i,\,b_j\big),$$

resuelto con el algoritmo húngaro. La clave de ByteTrack es usar **también** las
detecciones de baja confianza en una segunda ronda de asociación, lo que reduce
los IDs perdidos cuando un robot se ocluye un instante.

---

## 4. Homografía: de la cámara a la vista cenital

Una **homografía** $H$ relaciona el plano de la cancha visto por la cámara con
un plano cenital (vista de pájaro). En coordenadas homogéneas:

$$
\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix}
= H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix},
\qquad
H=\begin{bmatrix} h_{11}&h_{12}&h_{13}\\ h_{21}&h_{22}&h_{23}\\ h_{31}&h_{32}&h_{33}\end{bmatrix},
\qquad
x=\frac{x'}{w'},\; y=\frac{y'}{w'} .
$$

$H$ tiene **8 grados de libertad** ($h_{33}$ se fija a 1), así que bastan **4
correspondencias** de puntos para resolverla. Cada par $(\mathbf{p}_i \to \mathbf{p}_i')$
aporta dos ecuaciones lineales:

$$
\begin{aligned}
x_i'(h_{31}u_i+h_{32}v_i+1) &= h_{11}u_i+h_{12}v_i+h_{13},\\
y_i'(h_{31}u_i+h_{32}v_i+1) &= h_{21}u_i+h_{22}v_i+h_{23}.
\end{aligned}
$$

Con 4 puntos se forma un sistema $A\mathbf{h}=\mathbf{0}$ que `cv2.getPerspectiveTransform`
resuelve. Las **4 esquinas de la cancha** se obtienen automáticamente del
contorno verde más grande con `cv2.minAreaRect` (rectángulo rotado mínimo),
ordenadas como (sup-izq, sup-der, inf-der, inf-izq). Como la cámara se mueve,
$H$ se **recalcula en cada frame**; si un frame no da esquinas válidas, se
conserva la última $H$ válida.

### 4.1 Escala métrica real

El campo de la **RoboCup Junior Soccer (2023)** mide $1.82 \times 2.43$ m. El
lienzo cenital mide $W\times H$ px. Para convertir píxeles cenitales a metros se
usa un factor invariante a la rotación basado en la diagonal:

$$
\text{mpp}=\frac{\sqrt{1.82^2+2.43^2}}{\sqrt{W^2+H^2}}\quad[\text{m/px}],
\qquad
\mathbf{P}=\text{mpp}\cdot\mathbf{p}' .
$$

Así todas las métricas (distancia, velocidad, posesión) quedan en **metros
reales**, no en una escala fija inventada.

---

## 5. Segmentación (SAM)

SAM recibe las cajas de YOLO como *prompts* y devuelve una **máscara binaria**
$S_k\in\{0,1\}^{H\times W}$ por objeto. La caja $b_k$ guía a SAM hacia el objeto
correcto:

$$S_k=\operatorname{SAM}(I,\;\text{prompt}=b_k).$$

El backend es intercambiable (`seg_backend`): `sam2` (SAM 2.1, rápido), `sam3`
(SAM 3 vía Ultralytics) o `sam3_hf` (SAM 3 de Meta con *prompts de texto*, p. ej.
`"soccer robot"`, mediante `transformers`). La línea de innovación 3.7.3
(*prompts y contexto*) se cubre cambiando este parámetro.

---

## 6. Cinemática: distancia y velocidad

Sea $\mathbf{P}_f$ la posición métrica de un objeto en el frame $f$. La
**distancia recorrida** es la suma de los desplazamientos:

$$
d=\sum_{f} \big\lVert \mathbf{P}_f-\mathbf{P}_{f-1} \big\rVert_2 .
$$

La **velocidad instantánea** y su conversión a km/h:

$$
v_f=\frac{\lVert \mathbf{P}_f-\mathbf{P}_{f-1}\rVert_2}{\Delta t},
\qquad
v_f^{\text{km/h}} = v_f \cdot 3.6 .
$$

**Filtro anti-salto.** La homografía es ruidosa; un salto irreal
($>0.6$ m/frame para robots, $>1.2$ m/frame para el balón) se descarta para no
inflar la distancia. Es un filtrado de *outliers* por umbral físico.

---

## 7. Posesión con histéresis

El "dueño" del balón es el robot más cercano, pero con **dos radios** distintos
para evitar parpadeo (histéresis tipo Schmitt):

$$
\text{dueño}_{f}=
\begin{cases}
\arg\min_i \lVert \mathbf{P}^{\text{rob}}_i-\mathbf{P}^{\text{balón}}\rVert & \text{si } d_{\min} < r_{\text{captura}}=0.22\text{ m}\\[4pt]
\text{dueño}_{f-1} & \text{si sigue y } d < r_{\text{libera}}=0.40\text{ m}\\[4pt]
\varnothing & \text{si } d_{\min} > r_{\text{libera}}
\end{cases}
$$

La **posesión por equipo** es la fracción de frames en que el dueño pertenece a
ese equipo:

$$
\text{Pos}_{T}=\frac{\#\{f:\;\text{equipo}(\text{dueño}_f)=T\}}{\#\text{frames}}\times100\%.
$$

---

## 8. Detección de eventos (máquina de estados)

Sobre la secuencia de dueños se detectan eventos por **cambio de dueño**
$A\to B$:

$$
\text{evento}=
\begin{cases}
\textbf{pase} & \text{si } \text{equipo}(A)=\text{equipo}(B),\; A\neq B\\
\textbf{intercepción} & \text{si } \text{equipo}(A)\neq \text{equipo}(B)\\
\end{cases}
$$

Un **tiro a gol** se registra cuando la rapidez del balón supera un umbral:

$$v^{\text{balón}}_f > v_{\text{tiro}} = 6\ \text{km/h}.$$

Cada pase $A\to B$ incrementa el **grafo dirigido de pases** $G$, usado luego en
la red de interacción.

---

## 9. Mapas de calor (estimación de densidad)

Cada posición deposita un **núcleo gaussiano** sobre un acumulador 2D; la suma
es una estimación de densidad por kernel (KDE):

$$
\mathcal{H}(x,y)=\sum_{f}\exp\!\left(-\frac{(x-x_f)^2+(y-y_f)^2}{2\sigma^2}\right),
$$

con $\sigma$ controlando el "radio de influencia". Los heatmaps se dibujan sobre
la **plantilla cenital limpia** (no sobre el frame inclinado), de modo que las
zonas calientes correspondan a posiciones reales del campo.

---

## 10. Control de cancha (Voronoi)

A cada punto del campo se le asigna el robot más cercano (celda de Voronoi):

$$
\text{región}(i)=\{\mathbf{P}:\;\lVert \mathbf{P}-\mathbf{P}_i\rVert \le \lVert \mathbf{P}-\mathbf{P}_j\rVert\ \forall j\}.
$$

El **control de un equipo** es la fracción del área cuya celda pertenece a un
robot de ese equipo:

$$
\text{Ctrl}_T=\frac{\text{Área}\{\mathbf{P}:\ \text{equipo(robot más cercano)}=T\}}{\text{Área total del campo}} .
$$

Se aproxima discretizando el campo en una malla de $220\times220$ celdas. Es una
medida directa de **dominio espacial**, complementaria a la posesión del balón.

---

## 11. Red de pases (grafo de interacción)

El grafo dirigido $G=(V,E)$ tiene un nodo por robot y una arista $A\to B$ con
peso = número de pases. El tamaño del nodo refleja su **grado de salida
ponderado** (pases originados):

$$
\text{out}(A)=\sum_{B} w_{A\to B}.
$$

Es la versión "robótica" de las redes de pases del fútbol profesional y revela
qué robot es el principal **distribuidor** del equipo.

---

## 12. Asignación de equipos (k-means + voto temporal)

El color del marcador de cada robot (tercio superior de la caja, en HSV) se
agrupa con **k-means** ($k=2$):

$$
\min_{\{\mu_0,\mu_1\}} \sum_{i}\big\lVert \mathbf{x}_i-\mu_{c(i)}\big\rVert_2^2 .
$$

Como el color puede leerse mal en frames aislados (brillo, sombra), el equipo
**reportado** por `tracker_id` es la **moda temporal** de sus votos:

$$
\text{equipo}(\text{id})=\arg\max_{T}\ \#\{f:\ c_f(\text{id})=T\}.
$$

Esto estabiliza la etiqueta y evita que robots vistos antes de inicializar
k-means queden sin equipo ($-1$).

---

## 13. Resumen de parámetros

| Parámetro | Valor | Sección |
|---|---|---|
| $\tau_{\text{robot}}$ / $\tau_{\text{balón}}$ | 0.40 / 0.15 | §1 |
| IoU (NMS) | 0.45 | §1 |
| Rango HSV césped | H∈[35,85] | §2 |
| Rango HSV balón | H∈[5,22] | §1.1 |
| Campo real | 1.82 × 2.43 m | §4.1 |
| $r_{\text{captura}}$ / $r_{\text{libera}}$ | 0.22 / 0.40 m | §7 |
| $v_{\text{tiro}}$ | 6 km/h | §8 |
| Malla Voronoi | 220 × 220 | §10 |

---

### Créditos de las técnicas

Homografía y visión clásica (OpenCV) · Detección YOLO y SAM (Ultralytics) ·
Tracking ByteTrack y anotadores (Roboflow Supervision) · k-means
(scikit-learn). Todas las dependencias se citan en el `README.md`.
