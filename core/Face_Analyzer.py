import onnxruntime as ort
import cv2
import numpy as np
import insightface
from core.runtime import get_onnx_providers

class Analyzer_Faces:

    '''

        FaceAnalysis de la biblioteca insightface proporciona una interfaz de alto nivel para realizar análisis faciales, incluyendo detección de rostros, 
        extracción de características,estimación de edad y género, y alineación facial. A continuación, vamos a manejar lo que son los "faces" ,cual teine siguieten atributos
       
        
                bbox: Coordenadas de la caja delimitadora del rostro.

                landmark: Coordenadas de los puntos clave faciales (ojos, nariz, boca, etc.).

                embedding: Vector de características del rostro para tareas de reconocimiento.

                normed_embedding: Versión normalizada del vector de características.

                gender: Predicción del género (0 para femenino, 1 para masculino).

                age: Estimación de la edad.


     '''
    
    def __init__(self, device="auto", providers=None):

        '''Creamos en el constructor o incializamos el modelo Analyzer_Faces  solo una vez'''

        if providers is None:
            providers, device = get_onnx_providers(device)
        self.device = device
        self._analizador_cara = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=providers,
        )
        ctx_id = 0 if device == "cuda" else -1
        self._analizador_cara.prepare(ctx_id=ctx_id, det_size=(480, 480))

    def preparar_modelo(self,ctx_id=0,dimension=480):
       
        self._analizador_cara.prepare(ctx_id, det_size=(dimension, dimension))


    def obtener_caras(self, frame , ordenamiento="iz-de"):

        '''Retornaremos el conjunto de caras que detecta nuestro analizador  usaremos get(img) de Anylzer_Faces cual devuelve un arrray de Faces'''
        caras=self._analizador_cara.get(frame)

        caras_ordenadas=self.ordenamiento_caras(caras,ordenamiento)

        return caras_ordenadas



    def obtener_cara(self, frame, posicion):

        '''Retornaremos la posicion de una cara de un conjuto de Faces'''

        caras=self.obtener_caras(frame)

        return caras[posicion]
    

    def ordenamiento_caras(self,faces,ordenamiento="iz-de"):

        ''' Ordenar de izquierda a derecha (menor x del bounding box) o derecha izquierda de las caras'''
         
        faces_ordenadas=faces

        if ordenamiento == "iz-de":
         
         faces_ordenadas = sorted(faces, key=lambda face: face.bbox[0])
        
        elif ordenamiento == "de-iz":
         
         faces_ordenadas = sorted(faces, key=lambda face: face.bbox[0], reverse=True)

        return faces_ordenadas 
    

    def print_info_faces(self,img):

        '''Funcion que usa draw_on(img, faces) cual dibuja los resultados del análisis facial (como cajas delimitadoras y puntos clave)
           sobre la imagen original.Este devolvera una iamgen con su respectivo dibujos (cajas y puntos claves) en el'''
        
        faces=self.obtener_caras(img)  
        
        return self._analizador_cara.draw_on(img,faces)
        
         