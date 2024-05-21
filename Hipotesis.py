
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

class Hipotesis:
    def __init__(self, tipoDePrueba, mediaHipotesis, mediaMuestral, desviacionEstandar, tamanoMuestral, nivelSignificancia):
        self.tipoDePrueba = tipoDePrueba
        self.mediaHipotesis = mediaHipotesis
        self.mediaMuestral = mediaMuestral
        self.desviacionEstandar = desviacionEstandar
        self.tamanoMuestral = tamanoMuestral
        self.nivelSignificancia = nivelSignificancia
        self.alfa = 0.5 - nivelSignificancia
        self.grado_libertad = tamanoMuestral-1
        self.pruebaDeHipotesis()
        
    tablaDistribucionNormal = [
        (0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09),
        (0.0, 0.0000, 0.004, 0.008, 0.012, 0.016, 0.0199, 0.0239, 0.0279, 0.0319, 0.0359),
        (0.1, 0.0398, 0.0438, 0.0478, 0.0517, 0.0557, 0.0596, 0.0636, 0.0675, 0.0714, 0.0753),
        (0.2, 0.0793, 0.0832, 0.0871, 0.091, 0.0948, 0.0987, 0.1026, 0.1064, 0.1103, 0.1141),
        (0.3, 0.1179, 0.1217, 0.1255, 0.1293, 0.1331, 0.1368, 0.1406, 0.1443, 0.148, 0.1517),
        (0.4, 0.1554, 0.1591, 0.1628, 0.1664, 0.17, 0.1736, 0.1772, 0.1808, 0.1844, 0.1879),
        (0.5, 0.1915, 0.195, 0.1985, 0.2019, 0.2054, 0.2088, 0.2123, 0.2157, 0.219, 0.2224),
        (0.6, 0.2257, 0.2291, 0.2324, 0.2357, 0.2389, 0.2422, 0.2454, 0.2486, 0.2517, 0.2549),
        (0.7, 0.258, 0.2611, 0.2642, 0.2673, 0.2704, 0.2734, 0.2764, 0.2794, 0.2823, 0.2852),
        (0.8, 0.2881, 0.291, 0.2939, 0.2967, 0.2995, 0.3023, 0.3051, 0.3078, 0.3106, 0.3133),
        (0.9, 0.3159, 0.3186, 0.3212, 0.3238, 0.3264, 0.3289, 0.3315, 0.334, 0.3365, 0.3389),
        (1.0, 0.3413, 0.3438, 0.3461, 0.3485, 0.3508, 0.3531, 0.3554, 0.3577, 0.3599, 0.3621),
        (1.1, 0.3643, 0.3665, 0.3686, 0.3708, 0.3729, 0.3749, 0.377, 0.379, 0.381, 0.383),
        (1.2, 0.3849, 0.3869, 0.3888, 0.3907, 0.3925, 0.3944, 0.3962, 0.398, 0.3997, 0.4015),
        (1.3, 0.4032, 0.4049, 0.4066, 0.4082, 0.4099, 0.4115, 0.4131, 0.4147, 0.4162, 0.4177),
        (1.4, 0.4192, 0.4207, 0.4222, 0.4236, 0.4251, 0.4265, 0.4279, 0.4292, 0.4306, 0.4319),
        (1.5, 0.4332, 0.4345, 0.4357, 0.437, 0.4382, 0.4394, 0.4406, 0.4418, 0.4429, 0.4441),
        (1.6, 0.4452, 0.4463, 0.4474, 0.4484, 0.4495, 0.4505, 0.4515, 0.4525, 0.4535, 0.4545),
        (1.7, 0.4554, 0.4564, 0.4573, 0.4582, 0.4591, 0.4599, 0.4608, 0.4616, 0.4625, 0.4633),
        (1.8, 0.4641, 0.4649, 0.4656, 0.4664, 0.4671, 0.4678, 0.4686, 0.4693, 0.4699, 0.4706),
        (1.9, 0.4713, 0.4719, 0.4726, 0.4732, 0.4738, 0.4744, 0.475, 0.4756, 0.4761, 0.4767),
        (2.0, 0.4772, 0.4778, 0.4783, 0.4788, 0.4793, 0.4798, 0.4803, 0.4808, 0.4812, 0.4817),
        (2.1, 0.4821, 0.4826, 0.483, 0.4834, 0.4838, 0.4842, 0.4846, 0.485, 0.4854, 0.4857),
        (2.2, 0.4861, 0.4864, 0.4868, 0.4871, 0.4875, 0.4878, 0.4881, 0.4884, 0.4887, 0.489),
        (2.3, 0.4893, 0.4896, 0.4898, 0.4901, 0.4904, 0.4906, 0.4909, 0.4911, 0.4913, 0.4916),
        (2.4, 0.4918, 0.492, 0.4922, 0.4925, 0.4927, 0.4929, 0.4931, 0.4932, 0.4934, 0.4936),
        (2.5, 0.4938, 0.494, 0.4941, 0.4943, 0.4945, 0.4946, 0.4948, 0.4949, 0.4951, 0.4952),
        (2.6, 0.4953, 0.4955, 0.4956, 0.4957, 0.4959, 0.496, 0.4961, 0.4962, 0.4963, 0.4964),
        (2.7, 0.4965, 0.4966, 0.4967, 0.4968, 0.4969, 0.497, 0.4971, 0.4972, 0.4973, 0.4974),
        (2.8, 0.4974, 0.4975, 0.4976, 0.4977, 0.4977, 0.4978, 0.4979, 0.4979, 0.498, 0.4981),
        (2.9, 0.4981, 0.4982, 0.4982, 0.4983, 0.4984, 0.4984, 0.4985, 0.4985, 0.4986, 0.4986),
        (3.0, 0.4987, 0.4987, 0.4987, 0.4988, 0.4988, 0.4989, 0.4989, 0.4989, 0.499, 0.499)
    ]
    
    tablaTStudent = [
        (0.25, 0.1, 0.05, 0.025, 0.01, 0.005),#valores de significancia 
        (1, 1.0000, 3.0777, 6.3137, 12.7062, 31.8210, 63.6559),#(posicon,area bajo curva)
        (2, 0.8165, 1.8856, 2.9200, 4.3027, 6.9645, 9.9250),
        (3, 0.7649, 1.6377, 2.3534, 3.1824, 4.5407, 5.8408),
        (4, 0.7407, 1.5332, 2.1318, 2.7765, 3.7469, 4.6041),
        (5, 0.7267, 1.4759, 2.0150, 2.5706, 3.3649, 4.0321),
        (6, 0.7176, 1.4398, 1.9432, 2.4469, 3.1427, 3.7074),
        (7, 0.7111, 1.4149, 1.8946, 2.3646, 2.9979, 3.4995),
        (8, 0.7064, 1.3968, 1.8595, 2.3060, 2.8965, 3.3554),
        (9, 0.7027, 1.3830, 1.8331, 2.2622, 2.8214, 3.2498),
        (10, 0.6998, 1.3722, 1.8125, 2.2281, 2.7638, 3.1693),
        (11, 0.6974, 1.3634, 1.7959, 2.2010, 2.7181, 3.1058),
        (12, 0.6955, 1.3562, 1.7823, 2.1788, 2.6810, 3.0545),
        (13, 0.6938, 1.3502, 1.7709, 2.1604, 2.6503, 3.0123),
        (14, 0.6924, 1.3450, 1.7613, 2.1448, 2.6245, 2.9768),
        (15, 0.6912, 1.3406, 1.7531, 2.1315, 2.6025, 2.9467),
        (16, 0.6901, 1.3368, 1.7459, 2.1199, 2.5835, 2.9208),
        (17, 0.6892, 1.3334, 1.7396, 2.1098, 2.5669, 2.8982),
        (18, 0.6884, 1.3304, 1.7341, 2.1009, 2.5524, 2.8784),
        (19, 0.6876, 1.3277, 1.7291, 2.0930, 2.5395, 2.8609),
        (20, 0.6870, 1.3253, 1.7247, 2.0860, 2.5280, 2.8453),
        (21, 0.6864, 1.3232, 1.7207, 2.0796, 2.5176, 2.8314),
        (22, 0.6858, 1.3212, 1.7171, 2.0739, 2.5083, 2.8188),
        (23, 0.6853, 1.3195, 1.7139, 2.0687, 2.4999, 2.8073),
        (24, 0.6848, 1.3178, 1.7109, 2.0639, 2.4922, 2.7970),
        (25, 0.6844, 1.3163, 1.7081, 2.0595, 2.4851, 2.7874),
        (26, 0.6840, 1.3150, 1.7056, 2.0555, 2.4786, 2.7787),
        (27, 0.6837, 1.3137, 1.7033, 2.0518, 2.4727, 2.7707),
        (28, 0.6834, 1.3125, 1.7011, 2.0484, 2.4671, 2.7633),
        (29, 0.6830, 1.3114, 1.6991, 2.0452, 2.4620, 2.7564),
        (30, 0.6828, 1.3104, 1.6973, 2.0423, 2.4573, 2.7500)
    ]
    
    def comprobarTipoDePrueba(self, valorCalculado, valorTabular):
        if self.tipoDePrueba == "bilateral":
            if valorCalculado < valorTabular and valorCalculado > (-1)*valorTabular:
                print("Se acepta la hipótesis nula")
            else:
                print("Se rechaza la hipótesis nula")
            return
        
        # implementar el argumento de porque se acepta o rechaza la hipotesis nula 
        # para los casos de prueba unilaterales y bilaterales       
        if abs(valorCalculado) <= valorTabular:
            print("Se acepta la hipótesis nula")
        else:
            print("Se rechaza la hipótesis nula")
    
    def pruebaZ(self):
        valorZCalculada = (self.mediaMuestral - self.mediaHipotesis) / (self.desviacionEstandar / (self.tamanoMuestral ** 0.5))
        
        if self.tipoDePrueba == "bilateral":
            self.nivelSignificancia = self.nivelSignificancia / 2
            
        valorZTabular = self.valorZTabular()
        
        self.comprobarTipoDePrueba(valorZCalculada, valorZTabular)
    
    def pruebaT(self):
        valorTCalculada = (self.mediaMuestral - self.mediaHipotesis) / (self.desviacionEstandar / (self.tamanoMuestral ** 0.5))
        
        if self.tipoDePrueba == "bilateral":
            self.nivelSignificancia = self.nivelSignificancia / 2
            
        valorTTabular = self.valorTtabular()
        
        self.comprobarTipoDePrueba(valorTCalculada, valorTTabular)
        
        self.graph(valorTTabular)
                
    
    def valorZTabular(self):
        valorMasCercanoAlfa = 5.0
        apoyo = 0
        posiciones = (0,0)
        for i in range(1,len(self.tablaDistribucionNormal)):
            for j in range(1,len(self.tablaDistribucionNormal[i])):
                apoyo = abs(self.alfa - self.tablaDistribucionNormal[i][j])
                if valorMasCercanoAlfa > apoyo:
                    valorMasCercanoAlfa = apoyo
                    posiciones = (i,j)
                    
        valorZTabla = round(self.tablaDistribucionNormal[posiciones[0]][0] + (self.tablaDistribucionNormal[0][posiciones[1]-1]),2)
                    
        return valorZTabla
                
    
    def valorTtabular(self):
        posicion = 0
        for i in range(1, len(self.tablaTStudent)):
            if self.tablaTStudent[i][0] == self.grado_libertad:
                posicion = i
        
        valorTTabular = self.tablaTStudent[posicion][self.tablaTStudent[0].index(self.nivelSignificancia) + 1]
        print("Valor t tabular: ", valorTTabular)
        
        return valorTTabular
        
        
    
    def pruebaDeHipotesis(self):
        
        if (self.desviacionEstandar is None or self.desviacionEstandar == 0):
            print("No se puede realizar la prueba de hipótesis")
            return
        
        
        if(self.tamanoMuestral > 30):
            self.pruebaZ()
        else:
            self.pruebaT()
        


        
    def graph(self, crit):
        
        plt.figure()
        xs = np.linspace(-10, 10, 1000)
        plt.plot(xs, stats.t.pdf(xs, self.grado_libertad), 'k', label="T-Distribution PDF")
        ## Plot some vertical lines representing critical t-score cutoff
        if self.tipoDePrueba == "bilateral":
            critline = np.linspace(0, self.nivelSignificancia / 2)  # y range for critical line, AKA probability from 0-p*
            xs_2 = len(critline) * [crit]
        else: 
            critline = np.linspace(0, self.nivelSignificancia)  # y range for critical line, AKA probability from 0-p*
            xs_2 = len(critline) * [crit]
        #xs_2 = np.array(xs_2) + 0.3
        plt.plot(xs_2, critline, 'r', label=f"t* for dof={self.grado_libertad}")
        plt.axhline(self.nivelSignificancia, color='g', linestyle='-.')

        plt.legend()
        plt.show()
        
