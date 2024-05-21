from Hipotesis import Hipotesis as Hipotesis

def main():
    tipoDePrueba = "bilateral"
    mediaHipotesis = 7
    mediaMuestral = 6.69
    desviacionEstandar = 1.037
    tamanoMuestral = 13
    nivelSignificancia = 0.25
    hipotesis = Hipotesis(tipoDePrueba, mediaHipotesis, mediaMuestral, desviacionEstandar, tamanoMuestral, nivelSignificancia)
    
main()