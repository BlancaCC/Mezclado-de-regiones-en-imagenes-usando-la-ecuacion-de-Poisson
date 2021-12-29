# Poisson Image Editing   

Esto es un resumen y una ampliación de conceptos del artículo *Poisson Image Editing* 
Patrick Pérez, Michel Gangnet, and Andrew Blake. 2003. Poisson image editing. ACM Trans. Graph. 22, 3 (July 2003), 313–318. DOI:https://doi.org/10.1145/882262.882269

Que puede  descargar en  https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf  o consultar o ver un vídeo de los propios autores en 
https://dl.acm.org/doi/10.1145/882262.882269 . 

**TODO : Estaría interesante buscar cómo se hace en la actualidad, ya que el artículo es del 2003** [2]


## Abstract   

Se expone en este paper: 
1. Importación de regiones de imágenes **perfecta**. 
2. Afectar a la textura e iluminación de una zona concreta. 

## Introducción 

Hasta el momento el pegado de regiones de fotografías de una a otra se podía realizar mediante un recortado pixel perfecto de la imagen. O mediante técnicas imperfectas

**TODO : Podría ser interesante comentar las técnicas vistas en clase y sus problemas y cómo este artículo lo mejor**  [1]

Se proponen herramientas de pegado que sean más eficaces, basadas en herramientas matemáticas como: 

- Ecuación diferencial de Poisson con condiciones en la frontera de Dirichlet los cuales especifican la Laplaciana de una función desconocida. 

### Conceptos matemáticos   

Notas:
-  Las funciones pueden estar definida de $\mathbb R^n \longrightarrow K$ con $K$ el cuerpo de los números reales o complejos. Por simplicidad y contexto donde se trabaja, haremos $n=3$. 

Dadas $\phi , f$ funciones de clase dos definidas en un dominio de $\mathbb{R}^3$.   

Se define la ecuación de Poisson como 

\begin{equation}\label{eq:poisson}
    \Delta \phi = f
\end{equation}

Con $\Delta$ el operador Laplaciona que es la traza de la matriz Hessiana, esto es 
$$ \Delta \phi(x,y,z) = \frac{\partial ^ 2 \phi}{\partial x^2} + \frac{\partial ^ 2 \phi}{\partial y^2} + \frac{\partial ^ 2 \phi}{\partial z^2})(x,y,z)$$  


Existe una gama de ecuaciones ampliamente conocidas que surgen de imponer condiciones a la ecuación de Poisson \ref{eq:poisson} estas son:   

1. **Ecuación de Laplace**. Se hace $f=0$ la función continuamente nula.   

Cuando una función cumple que su laplaciano es siempre nulo se  le llama funciones **armónicas** \label{def:armonica}. Ejemplo de funciones de este tipo son 
$f(x,y) = x^2 - y^2$ (o cualquier función holomorfa, las derivables en los complejos).  

2. **Problema o ecuación de Dirichlet**   

Consiste en encontrar una función armónica \ref{def:armonica} sobre un dominio y que en la frontera o borde (en nuestro caso será el borde de la región que queramos pegar en otra imagen) sea igual a otra función.   
De existir solución a este tipo de ecuaciones esta es única. (TODO: Buscar teorema donde se dice). 

**TODO: Comentar conceptos matemáticos y cómo se relacionan con las imágenes** [3].   


## Motivación   

Se tiene una doble motivación:  

-  Edwin H. Land and John J. McCann, "Lightness and Retinex Theory," J. Opt. Soc. Am. 61, 1-11 (1971) que se puede encontrar en https://www.osapublishing.org/josa/abstract.cfm?uri=josa-61-1-1  

1. A nivel fisiológico por cómo el ojo capta las imágenes este tratamiento es correcto. 
2. La función escalar que se utiliza para la interpolación es solución única del problema planteado.  

Lo que aporta este artículo es que trata el problema como uno de minimización (encontrar la función que minimice tal operación) 

## Trabajos relacionados   

Comentar otras técnicas utilizadas hasta la fecha en que se publicó el artículo, está relacionado con [1].   

## Solución de Poisson para la interpolación guiada  

En ese capítulo se detalla la interpolación de una imagen usando campo de vector guía. (TODO buscar una traducción de Guidance vector field).  

### Hipótesis  

- Se trata cada canal de la imagen por separado (**TODO** ya que según el artículo es suficiente ¿por qué? Quizás sea suficiente por los resultados que da visualmente, podría plantearse esta pregunta en el propio trabajo a presentar  y cómo afectaría si se utilizaran varios canales simultáneamente. [4])  

(Nota, como estamos trabajando con un canal, la imágenes se representa en dominio de R^2) 

- Sea $S$ un subconjunto cerrado de $R^2$ y sea $\Omega$ un subconjunto cerrado de $S$ con borde $\partial \Omega$. 
-  Denotemos por $f^*$ a la función $f^*: S - \Omega \longrightarrow R$ (Representa la imagen sobre la que queremos poner la otra). ([6] Nótese que para resolver el problema no se está utilizando información sobre qué había en la imagen original cuando pegamos algo. Para ciertos problemas esta información podría ser relevante y podría afinar el resultado. Esto se podría comentar en el proyecto a entregar. )  
- Sea $f$ la función desconocida  $f: \Omega ^ \circ \longrightarrow R$    
- Finalmente se define $v$ como el campo de vectores en $\Omega$ ([7] *TODO: Todavía no entiendo muy bien cómo se podría explicar de manera intuitiva este concepto* ).   

El interpolante más simple que se podría definir sería 

\begin{equation}
min_ f \int \int_\Omega | \nabla f| ^2 \text{con } f|_{\partial \Omega} = f^* | _{\partial \Omega}, 
\end{equation}
donde  además el operador debe satisfacer 
\begin{equation}
 \Delta f = 0 \text{sobre } f|_{\partial \Omega} = f^* | _{\partial \Omega}
\end{equation}

$\Delta$ representa el operador Laplaciano, (Suma de la traza del Hessiano).  

Este método no produce resultados satisfactorios y es por ello que se intenta resolver con ecuaciones diferenciales más complejas, como en impating 

https://dl.acm.org/doi/10.1145/344779.344972
https://www.researchgate.net/publication/17641495_Lightness_and_Retinex_Theory

  
En su lugar se introduce un vector guía de la forma 

\begin{equation}
min_ f \int \int_\Omega | \nabla f - v | ^2 \text{con } f|_{\partial \Omega} = f^* | _{\partial \Omega}, 
\end{equation}  

Con las condiciones de borde que   


\begin{equation}
\Delta f = div(v) \text{ en } \Omega, \text{con condiciones en borde } f|_{\partial \Omega} = f^* | _{\partial \Omega}, 
\end{equation}    

TODO: 
-  No entiendo la intuición tras esta definición. 
-  ¿Cómo se calcula v?


La estrategia fundamental consiste pues en resolver tres veces ecuaciones de este tipo. 

Ha tenido resultados buenos en RGB, para el que se han hecho los experimentos, y también otros similares para CIE-LAB (es una escala de colores que presenta ventajas a la hora de aplicar efectos a la imagen https://es.wikipedia.org/wiki/Espacio_de_color_Lab) . 

Cuando el campo es conservativo ( es decir es el gradiente de una función (Nota Blanca: Esto es un teorema)) se puede entender este algoritmo como una función que parte de la original con cierta corrección.  

Hasta aquí se ha mostrado la teoría subyaciente en el caso continuo, ahora es necesario implementar el caso discreto para poder aplicarse a las imágenes.  

# Discrete Poisson solver (No sé traducir esto pa que no suene cateto xD ¿Solución discreta a la ecuación de Poisson?)  

La ecuación expuesta (TODO añadir referencia cuando se pase a LaTEX) puede ser discretizada de diferentes método: exponemos uno.   

Utilizando una rejilla de píxeles (Tampoco sé traducir **discrete pixel grid** para que no quede cateto xD ) ([7] como siempre se podría  comparar qué otros métodos existen para discretizar esto. ¿qué tipo de interpolación y cómo afecta?) ( [8] ¿Cómo se podrían comparar en este tipo de problemas mejores resultados? ¿Con un humano que te diga que es real? ¿con una red que sea capaz de distinguir objetos?) 


Sin pérdida de generalidad se mantiene la misma notación para el caso discreto que para el continuo. 

Donde ahora $S y \Omega$ se han convertido en conjuntos finitos de píxeles definidos en nuestra rejilla finita.   

Se define ahora la función $N_p$ que consiste en el conjunto de sus cuatro vecinos conectados en cruz del pixel $p$ ([9] Comentario sobre por qué no se ha tomado como $N_p$ los 8 vecinos que se encuentra en cruz y equis o solo las equis, ¿mejoraría el resultado considerar más? O por cómo se define el borde con estos  se contemplan todos los casos).

Denotemos por $\textlangle p, q \textrangle$ denotado por el conjunto de pares de vecinos con $q \in N_p$.  

El borde se define ahora como 
TODO formalizar
Los puntos $p$ tales que algún punto de $N_p$ esté en $\Omega$ (la imagen en la que se pega [9] se ha descrito entonces por economía de comprobaciones ).
Ejemplo en los que no se considera borde (s puntos de S y o puntos de $\Omega$) $p$ en un sentido más estricto debería ser considerado borde, sin embargo con este método no
s | s_1 | o
s | p | s_2|
s | s | s 

Se podría experimentar  (¿[8] cómo se podría comparar ?) cómo afecta al resultado visual [o si por estar próximo a puntos que serían bordes ($s_1, s_2$) se podría suponer que toma valores próximos y se puede omitir.  


(TODO completar )

Ahora basta con discretizar dichas ecuaciones. 
Est se podría hacer con Gauss-Seidel iteration

TODO: El algoritmo viene explicado aquí, 
- abría que explicarlo, implementarlo
 https://es.wikipedia.org/wiki/Método_de_Gauss-Seidel

También sería enteresante comentar su costo computacionalm

with successive overrelaxation or V-cycle multigrid.
[9] Si sobra tiempo (que no creo) pero estaría chulísimo hacer una app funcional con todo esto y no solo un programa en pitoncillo. 






# Conceptos candidatos a añadir al trabajo   

La numeración indica tan solo dónde surgió la idea. 

[1] Comentar, implementar o comparar técnicas de mezclado tradicionales o vistas en clase. 

[2] El artículo se remonta a 2003, ¿Cómo se hacen estas cosas actualmente? Estaría incluso genial experimentar con ellas si no es mu difícil, o incluso implementarlas o proponer las nuestras. 
¿se podrá hacer con redes neuronales? ¿Con una GAN? (Muy relacionada con [5])

[3] Introducir y explicar conceptos matemáticos usados y cómo se relacionan estos con nuestro problema de mezcla de regiones. 

[4] Se trata cada canal de la imagen por separado ya que según el artículo es suficiente ¿por qué? Quizás sea suficiente por los resultados que da visualmente, podría plantearse esta pregunta en el propio trabajo a  presentar y cómo afectaría si se utilizaran varios canales simultáneamente. 

[5] Investigar un poco sobre este pensamiento: 
- Este artículo fue presentado en 2003 poco antes del boom de las RRNN, ¿Pueden los problemas que resuelve ser resueltos con RRNN? ¿o existe un área de visión por computador 
que debe hacerse todavía mediante técnicas "artesanales" como estas?  

[6] Nótese que para resolver el problema no se está utilizando información sobre qué había en la imagen original cuando pegamos algo. Para ciertos problemas esta información podría ser relevante y podría afinar el resultado. Esto se podría comentar en el proyecto a entregar. 


[7]  Sobre el comentario hecho sobre la discretización con rejilla como siempre se podría  comparar qué otros métodos existen para discretizar esto. ¿qué tipo de interpolación y cómo afecta?.  

[8] ¿Cómo se podrían comparar en este tipo de problemas mejores resultados? ¿Con un humano que te diga que es real? ¿con una red que sea capaz de distinguir objetos?
Una apreciación que se podría hacer el hacer pegado, el resultado ideal a buscar es que una persona lo reconozca como "real" y esto no es algo matemático riguroso y comparable matemáticamente (o eso creo). Podríamos proponer una formulación matemática de una *máquina* que sea capaz de reconocer imágenes al mismo nivel que una persona.  
¿esto sería posible? 

[9] Si sobra tiempo (que no creo) pero estaría chulísimo hacer una app funcional con todo esto y no solo un programa en pitoncillo. 
