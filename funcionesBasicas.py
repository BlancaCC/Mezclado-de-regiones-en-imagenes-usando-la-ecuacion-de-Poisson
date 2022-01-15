# Código: 0   Import básico y otras definiciones globales auxiliares
import numpy as np
from itertools import chain # Herramienta de programación funcional

# Para la lectura de las imágenes
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Para las matrices dispersas
from scipy.sparse import lil_matrix
import scipy.sparse.linalg

# ___ Lectura de imágenes___
# path carpeta con imágenes
path = 'images/'

#Código: 1 Implementación de la funciones de cálculo de vecinos y borde

def Neighborhood(pixel : list, inputChannel : np.ndarray )->list:
    '''
    pixel: tupla o lista de dos números naturales que indican  las coordenadas del pixel. Se comprobará que que es coherente con las dimensiones de la imagen
    inputChannel: imagen a la que pertenece el píxel p. Debe ser solo un canal. Es decir, será una matriz de dos dimensiones. 
    '''
    x,y = pixel
    # Dimensiones de inputChannel coherente con el pixel demandado
    x_max , y_max = np.shape(inputChannel) 
    DimensionChecker = lambda x,y: 0 <= x < x_max and 0 <= y < y_max
    assert DimensionChecker(x,y), f' El pixel {pixel} se sale de las dimensiones (0 <= x <{x_max}, 0 <= y < {y_max})'

    # Se devuelven vecino válidos
    neighborhoodCoordinates = filter(
        lambda tuple : DimensionChecker(tuple[0], tuple[1]),
        list( chain.from_iterable(([x+i,y], [x, y+i]) for i in [-1,1])) 
    )
   
    return list(neighborhoodCoordinates)


def SourceBoundary(inputChannel: np.ndarray, O: set )-> set : 
    '''
    Se pretenda calcular el borde de imagen destino. 
    inputChannel: imagen a la que pertenece el píxel p. Debe ser solo un canal. Es decir, será una matriz de dos dimensiones. 
    O: Supondremos que O son n conjunto de pares de tuplas de coordenadas. 
    Devuelve un conjunto de coordenadas en tuplas que son el borde de O.
    ''' 
    # Calculamos el conjunto de puntos S
    x_len, y_len = np.shape(inputChannel)
    S = set( (i,j) for i in range(x_len) for j in range(y_len)) 
    # Calculamos la diferencia 
    difference = S.difference(O)
    # Filtramos los puntos que cumplen la condición de que sus vecinos están en O (Omega)
    boundary = set(filter(
        lambda p : len(set(
            map(
                lambda l : tuple(l),
                Neighborhood(p, inputChannel)
            )
        ).intersection(O))>0,
        difference
    ))

    return boundary

    # Código: 2 Implementación del campo de guía  

def GuidanceFieldGenerator(inputChannel : np.ndarray, targetChannel: np.ndarray):
    ''' Devuelve la función v_{p,q} = g_p - g_q$
    f_ = f^* Para el caso borde. 
    Para el caso particular en que g es inputChannel
    '''
    x_len, y_len = inputChannel.shape 

    def V (p,q):
       # Calculamos g_p (nunca en el borde, luego)
        g_p = inputChannel[p[0]][p[1]]
        if q[0] < x_len and q[1] < y_len:
            g_q = inputChannel[q[0]][q[1]]
        else:
            g_q = targetChannel[q[0]][q[1]]
        return g_p - g_q
    return V
   

# Código: 3 Funciones lectura de imágenes 
'''
Salvo tener la necesidad de cambiar la ruta donde se leen las imágenes, no es necesario la lectura de ésta
celda, ya que contiene: 
1. Bibliotecas necesarias.
2. Indicación de la ruta de donde se toman las imágenes y la lectura de las imágenes.
3. Conjunto de funciones auxiliaras como las utilizadas para mostrar imágenes en pantalla. 
Éstas están tomadas de la práctica 0 y a lo sumo tienen ligeras modificaciones.  
'''

# Devuelve si una imagen está en blanco o negro 
def IsGreyScale(img):
    '''Devuelve si una imagen está en blanco y negro'''
    return len(img.shape) == 2

# Reutilizamos código de la práctica inicial
def ReadImage(filename, flagColor):
  '''
  @param filename: nombre de la foto 
  @para flagColor: boolean escala de grises o color
  '''
  return np.asarray(cv.imread(filename, flagColor), dtype=float)
## Flags
flagColor = cv.IMREAD_ANYCOLOR
flagGrey = cv.IMREAD_GRAYSCALE


## Algunas funciones para pintar imágenes 
# Para pintar imágenes
def Normalize (img):
    ''' Transforma una imagen de números reales
    al intervalo [0,1] 
    '''
    min = np.min(img)
    max = np.max(img)

    normalized_img = np.copy(img)

    if max - min > 0:
        normalized_img = (normalized_img - min) / (max - min)
    else: 
        normalized_img *= 0 # suponemos todo blanca
    return normalized_img

def PrintOneImage( img, title=None, normalize= True, size = (13,13)):
    '''Muestra una imagen usando imshow'''

    plt.figure(figsize=size)
    if normalize:
        img = Normalize(img)
    if IsGreyScale(img):
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img[:,:,::-1])
    if title:
        plt.title(title)
    plt.show()


## Leemos algunas imágenes y las mostramos  
pathPielRugosa = path + 'PielRugosa.png'
pathPintadaPared = path + 'PintadaEnPared.png'
imgPielRugosa = ReadImage(pathPielRugosa, flagColor) 
imgPintadaPared = ReadImage(pathPintadaPared, flagColor)


# Playa 
pathPlaya = path + 'Playa2.png'
imgPlaya = ReadImage(pathPlaya, flagColor) 


# Código 4: Selección de regiones 

def SelectedRegion( originalImage: np.ndarray, selectedImage : np.ndarray, showImage = False, title = 'Regiones seleccionadas') -> set :
    '''
    Devuelve un conjunto de pares de coordenadas que han sido modificadas en alguno de los tres canales. 
    Su correspondencia teórica sería $\Omega$. 
    originalImage: Imagen en la que no se ha seleccionado ningún área. 
    selectedImage: Imagen con el área seleccionada. 
    showImage: Booleano si True muestra las regiones seleccionadas. 
    '''

    assert originalImage.shape == selectedImage.shape, f'El tamaño de las imágenes no son iguales, son de tamaño: {originalImage.shape} y {selectedImage.shape}'
    differenceMatrix = originalImage - selectedImage
    if(showImage):
        PrintOneImage(differenceMatrix, title= title,size=(7,7))
    # Seleccionamos las coordenadas que han cambiado
    x_len, y_len, _ = differenceMatrix.shape
    selectedRegion = set()
    for x in range(x_len):
        for y in range(y_len): 
            if (
                differenceMatrix[x][y][0] != 0
                or differenceMatrix[x][y][1] != 0
                or differenceMatrix[x][y][2] != 0 
            ):
                selectedRegion.add((x,y))

    return selectedRegion

# Código 8: Cálculo de la función de interpolación

def OneChannelDiscretePoissonSolver( source, targetImg, selectedSet, V = ""):
    ''' Devuelve f, la función interpolación para los canales dados
    source: Imagen de la que se extrae la región que se quiere clonar
    targetImg:  Imagen donde se va a posicionar la nueva región 
    selectedSet: Conjunto de puntos con las región selecciónada, a de calcularse previamente
    V: Campo de vectores guía, debe de ser una función que reciba dos píxeles y devuelva dos números reales. Si none se toma la divergencia de targetImg. 
    '''
    # Calculamos borde del arcoiris 
    sourceBoundary = SourceBoundary(targetImg, selectedSet)
    
    # Calculamos el campo de escalares 
    if(V == "" or V == "originalGuidance"):
        V = GuidanceFieldGenerator(source,targetImg) # imagen original de la que se extrae la región seleccionada
                                                 # Nótese que aquí se trabaja con las coordenadas de esos píxeles
    elif (V == "mixingGradients"):
        V = GuidanceFieldMixingGradients(source, targetImg)
                                                 

    # Cálculo de valores necesarios
    omega_len = len(selectedSet)
    equation = lil_matrix((omega_len, omega_len), dtype = np.int8)
    terms = np.zeros(omega_len)

    column = {}
    # Se almacenan los índices de los puntos en un diccionario: p -> índice matriz
    for index,p in enumerate(selectedSet):
        column[p] = index
    
    # Escribimos el sistema de ecuaciones
    for index,p in enumerate(selectedSet):
        # Cálculo de N_p, vecinos
        Np = Neighborhood(p, targetImg)
        Np_cardinality = len(Np)
        # Entrada para |N_p| f_p 
        equation[index, index] = Np_cardinality
        # Operaciones que involucran vecinos
        for q in Np:
            q = tuple(q)
            # Entrada para  - \sum_{q \in N_p \cap \Omega} f_q 
            if q in selectedSet:
                equation[index, column[q]] = -1
            # Término independiente equación: \sum_{q \in N_p} v_{pq}
            terms[index] += V(p,q)
            # Término independiente \sum _{q \in N_p \cap \partial \Omega} f_q ^* 
            if q in sourceBoundary:
                terms[index] += targetImg[q[0]][q[1]]
    # Resolvemos sistema
    f = scipy.sparse.linalg.spsolve(equation.tocsr(True), terms)
    return f

def MultipleChannelDiscretePoissonSolver( multiChannelSource, multiChannelTargetImg, selectedSet, V = ""):
    '''
    Se calcula f, la función de interpolación para cada cada canal, y se realiza la sustitución de los valores
    de los píxeles de la región seleccionada por los de la función de interpolación.
    multiChannelSource: Imagen multicanal que se desea copiar.
    multiChannelTargetImg: Imagen multicanal sobre la que se desea insertar la imagen fuente.
    selectedSet: Conjunto de puntos que definen la región de la imagen destino donde se va a realizar la inserción de la imagen fuente.
    V: Campo de vectores guía, debe de ser una función que reciba dos píxeles y devuelva dos números reales. Si es none se toma la divergencia de multiChannelTargetImg
    '''
    # Copiado de la imagen para no transformar la imagen original
    seamlessImg = np.copy(multiChannelTargetImg)

    # Se itera sobre todos los canales de la imagen
    for i in range(3):
        source = multiChannelSource[:,:,i] # Imagen de la que se extrae la región que se quiere clonar
        targetImg = multiChannelTargetImg[:,:,i]  # Imagen donde se va a posicionar la nueva región 
        f = OneChannelDiscretePoissonSolver(source, targetImg, selectedSet, V) # Cómputo de la función de interpolación para el canal

        # Se actualizan los valores de la región
        for index,p in enumerate(selectedSet):
            seamlessImg[p[0], p[1],i] = f[index]
    
    return seamlessImg

# Código 10: Matriz de transformación

def GetMatrixTransformation(traslationX: int, traslationY: int, degreesRot: float, simmetryX: bool, simmetryY: bool, region = None, centroide = None) -> np.ndarray:
    '''Devuelve una transformación rígida dirigida a aplicarse a los puntos seleccionados
    Parámetros:
    1. `traslationX`: traslación, en píxeles, en el eje X de la imagen.
    2. `traslationY`: traslación, en píxeles, en el eje Y de la imagen.
    3. `degreesRot`: ángulo a rotar, en grados, de la región.
    4. `symmetryX`: valor booleano indicando si se desea que la región final sea simétrica, en el eje X, con la región inicial.
    5. `symmetryY`: valor booleano indicando si se desea que la región final sea simétrica, en el eje Y, con la región inicial.
    6. `region`: conjunto de puntos necesario para el cálculo del centroide. Puede ser None, en tal caso será el valor del centroide o (0,0)
    7. `centroide`: Valor de centroide, puede ser `None` y en tal caso se calculará. 
    '''

    # Calcular el centroide de la región
    centerX = 0
    centerY = 0
    
    if(centroide == None ):
        if( region != None):
            centerX = np.sum([p[0] for p in region]) / len(region)
            centerY = np.sum([p[1] for p in region]) / len(region)
    else: 
        centerX = centroide[0]
        centerY = centroide[1]

    # Cálculo de matriz de traslación al inicio
    initMatrix = np.zeros((3,3))
    for i in range(3):
        initMatrix[i, i] = 1
    initMatrix[0, 2] = -centerX
    initMatrix[1, 2] = -centerY

    # Cálculo de la matriz de simetría
    simmetryMatrix = np.zeros((3,3))
    simXVal = -1 if simmetryX else 1
    simYVal = -1 if simmetryY else 1
    simmetryMatrix[0, 0] = simXVal
    simmetryMatrix[1, 1] = simYVal
    simmetryMatrix[2, 2] = 1

    # Cálculo de la matriz de rotación
    rotateMatrix = np.zeros((3,3))
    cosVal = np.cos(degreesRot * np.pi / 180)
    sinVal = np.sin(degreesRot * np.pi / 180)
    rotateMatrix[2, 2] = 1
    rotateMatrix[0, 0] = cosVal
    rotateMatrix[0, 1] = -sinVal
    rotateMatrix[1, 0] = sinVal
    rotateMatrix[1, 1] = cosVal

    # Cálculo de la matriz de traslación según los parámetros
    traslateMatrix = np.zeros((3,3))
    for i in range(3):
        traslateMatrix[i, i] = 1
    traslateMatrix[0, 2] = traslationX
    traslateMatrix[1, 2] = traslationY

    # Cálculo de matriz de transformación mediante composición #issue 18 
    finalMatrix = np.dot(np.dot(np.dot(traslateMatrix, rotateMatrix), simmetryMatrix), initMatrix)
    # Truncamos # TODO comentar el motivo ¿añadir una función como argumento de qué se le va a hacer a esto? -> ¿Vale con lo de abajo?
    finalMatrix = np.round(finalMatrix)
    return finalMatrix

    # Código 12: Función de clonado directo sin modificar

def DirectedCloning ( multiChannelSource, multiChannelTargetImg, selectedSet, transformation = [[1,0,0], [0,1,0], [0,0,1]]): 
    """
    Realiza el clonado directo de la región de unos píxeles seleccionados de una imagen fuente en una imagen objetivo.
    1. `multiChannelSource`: Imagen fuente que se va a clonar.
    2. `multiChannelTargetImg`: Imagen objetivo en la que se va a realizar el clonado.
    3. `selectedSet`: Región de píxeles que se desean clonar de la imagen fuente.
    4. `transformation`: Movimiento rígido a realizar a la región de píxeles a clonar.
    """
    transformation = np.array(transformation)
    newImg = np.copy( multiChannelTargetImg)
    for p in selectedSet:
        # Obtener transformación
        x,y,_ = transformation.dot([p[0],p[1], 1])
        x = int(x)
        y = int(y)

        # Copiar valores pertinentes
        for i in range(3):
            newImg[x,y, i] = multiChannelSource[p[0],p[1], i]
    return newImg


# Código 17: Cálculo de la función de interpolación generalizado movimiento rígido
def Transform (coordinates, transformation):
    """
    Función que dadas unas coordenadas y una matriz de transformación, computa el resultado de aplicar a dichas
    coordenadas la transformación indicada.
    """
    x,y,_ = transformation.dot([coordinates[0], coordinates[1], 1])
    return (int(x),int(y))

def MultipleChannelDiscretePoissonSolverTransformed( multiChannelSource, multiChannelTargetImg, selectedSet, V = "", transformation = np.array([[1,0,0], [0,1,0], [0,0,1]])):
    '''
    Se calcula f, la función de interpolación para cada cada canal, y se realiza la sustitución de los valores
    de los píxeles de la región seleccionada por los de la función de interpolación teniendo en cuenta el movimiento
    rígido que se realiza sobre la región seleccionada en la imagen fuente.
    multiChannelSource: imagen multicanal que se desea copiar.
    multiChannelTargetImg: imagen multicanal sobre la que se desea insertar la imagen fuente.
    selectedSet: conjunto de puntos que definen la región de la imagen destino donde se va a realizar la inserción de la imagen fuente.
    V: Campo de vectores guía, debe de ser una función que reciba dos píxeles y devuelva dos números reales. Si es none se toma la divergencia de multiChannelTargetImg
    transformation: np.array con un transformación rígida
    '''
    # Copiado de la imagen para no transformar la imagen original
    seamlessImg = np.copy(multiChannelTargetImg)

    # Se forma una imagen intermedia copiando sólo los píxeles necesarios: la región y el borde
    # de la imagen fuente en la imagen destino
    boundary = SourceBoundary(multiChannelSource[:,:,0], selectedSet)
    regionToCopy = selectedSet.union(boundary)
    newMultiChannelSource = DirectedCloning( multiChannelSource, multiChannelTargetImg, regionToCopy , transformation)

    # Calcular región de puntos después de transformación
    newSelectedSet = set( 
        map(
            lambda t : Transform(t, transformation),
            selectedSet
        )
    )

    # Se itera sobre todos los canales de la imagen
    for i in range(3):
        source = newMultiChannelSource[:,:,i] # Imagen de la que se extrae la región que se quiere clonar
        targetImg = multiChannelTargetImg[:,:,i]  # Imagen donde se va a posicionar la nueva región 
        f = OneChannelDiscretePoissonSolver(source, targetImg, newSelectedSet, V) # Cómputo de la función de interpolación para el canal

        # Se actualizan los valores de la región
        for index,p in enumerate(newSelectedSet):
            seamlessImg[p[0], p[1],i] = f[index]
    
    return seamlessImg


def GuidanceFieldMixingGradients(inputChannel : np.ndarray, targetChannel: np.ndarray):
    ''' Devuelve la función v_{p,q} = f^*_p - f^*_q si |f^*_p - f^*_q| > |g_p - g_q|
    g_p - g_q en el caso contrario
    f_ = f^* Para el caso borde. 
    Para el caso particular en que g es inputChannel
    '''
    x_len, y_len = inputChannel.shape 

    def V (p,q):
        # Calculamos g_p (nunca en el borde, luego)
        g_p = inputChannel[p[0]][p[1]]
        if q[0] < x_len and q[1] < y_len:
            g_q = inputChannel[q[0]][q[1]]
        else:
            g_q = targetChannel[q[0]][q[1]]
        
        # Se computa g_p - g_q
        gDifference = g_p - g_q
        
        # Se computa f^*_p - f^*_q
        pDifference = targetChannel[p[0], p[1]] - targetChannel[q[0], q[1]]

        # Según el valor absoluto se devuelve un valor u otro
        if (np.abs(pDifference) > np.abs(gDifference)):
            return pDifference
        else:
            return gDifference
            
    return V