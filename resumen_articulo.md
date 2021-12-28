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

**TODO: Comentar conceptos matemático y cómo se relacionan con las imágenes** [3].   




## Motivación   

Se tiene una doble motivación:  

-  Edwin H. Land and John J. McCann, "Lightness and Retinex Theory," J. Opt. Soc. Am. 61, 1-11 (1971) que se puede encontrar en https://www.osapublishing.org/josa/abstract.cfm?uri=josa-61-1-1  
  




# Conceptos candidatos a añadir al trabajo   

La numeración indica tan solo dónde surgió la idea. 

[1] Comentar, implementar o comparar técnicas de mezclado tradicionales o vistas en clase. 

[2] El artículo se remonta a 2003, ¿Cómo se hacen estas cosas actualmente? Estaría incluso genial experimentar con ellas si no es mu difícil, o incluso implementarlas o proponer las nuestras. 
¿se podrá hacer con redes neuronales? ¿Con una GAN?   

[3] Introducir y explicar conceptos matemáticos usados y cómo se relacionan estos con nuestro problema de mezcla de regiones. 

