import cv2
import numpy as np

# Caminhos dos arquivos
ARQUIVO_VIDEO = 'velocidade-f1/speedsense.mp4'
ARQUIVO_MODELO = 'velocidade-f1/frozen_inference_graph.pb'
ARQUIVO_CFG = 'velocidade-f1/ssd_mobilenet_v2_coco.pbtxt'

def carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG):
    '''
    Carrega o modelo de detecção de objetos usando OpenCV.
    ARQUIVO_MODELO: Caminho para o arquivo .pb contendo os pesos do modelo.
    ARQUIVO_CFG: Caminho para o arquivo .pbtxt contendo a configuração do modelo.
    Retorna o modelo carregado.
    '''
    try:
        modelo = cv2.dnn_DetectionModel(ARQUIVO_MODELO, ARQUIVO_CFG)
        modelo.setInputSize(320, 320)
        modelo.setInputScale(1.0 / 127.5)
        modelo.setInputMean((127.5, 127.5, 127.5))
        modelo.setInputSwapRB(True)
    except cv2.error as erro:
        print(f"Erro ao carregar o modelo: {erro}")
        exit()
    return modelo

def aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf, limiar_supr):
    '''
    Aplica a Supressão Não Máxima para reduzir o número de caixas delimitadoras sobrepostas.
    caixas: Lista de caixas delimitadoras.
    confiancas: Lista de confianças de cada caixa.
    limiar_conf: Limiar de confiança para considerar detecções.
    limiar_supr: Limiar de sobreposição para suprimir caixas redundantes.
    Retorna uma lista de caixas após aplicar a supressão.
    '''
    indices = cv2.dnn.NMSBoxes(caixas, confiancas, limiar_conf, limiar_supr)
    return [caixas[i] for i in indices.flatten()] if len(indices) > 0 else []

def main():
    '''
    Função principal que executa a detecção de carros no vídeo de Formula 1.
    '''
    captura = cv2.VideoCapture(ARQUIVO_VIDEO)
    detector_carros = carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG)
    pausado = False

    while True:
        if not pausado:
            ret, frame = captura.read()
            if not ret:
                break
                #teste

            # Criação do blob a partir do frame e realização da detecção
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            detector_carros.setInput(blob)
            deteccoes = detector_carros.forward()

            caixas = []
            confiancas = []

            # Extração das caixas delimitadoras e confianças das detecções
            for i in range(deteccoes.shape[2]):
                confianca = float(deteccoes[0, 0, i, 2])
                if confianca > 0.5:
                    (altura, largura) = frame.shape[:2]
                    caixa = (deteccoes[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])).astype("int")
                    (inicioX, inicioY, fimX, fimY) = caixa
                    caixas.append([inicioX, inicioY, fimX - inicioX, fimY - inicioY])
                    confiancas.append(confianca)

            # Aplicação da supressão não máxima para finalizar as caixas delimitadoras
            caixas_finais = aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf=0.5, limiar_supr=0.4)
            numero_carros = len(caixas_finais)

            # Desenho das caixas e exibição do número de carros detectados
            for (inicioX, inicioY, largura, altura) in caixas_finais:
                cv2.rectangle(frame, (inicioX, inicioY), (inicioX + largura, inicioY + altura), (0, 255, 0), 2)
            cv2.putText(frame, f"Carros: {numero_carros}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Exibição do frame processado e controle de pausa/play
        cv2.imshow("Detecção de Carros", frame)
        
        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('p'):
            pausado = not pausado

    # Liberação dos recursos ao finalizar
    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
