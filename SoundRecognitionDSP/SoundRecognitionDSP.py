# Proyecto Reconocimiento de Sonido por DSP UV 2020
# Modulo PC, 4to de 4 modulos, a cargo de:
# Omar Jordan, Harold Medina, Pablo Torres

"""
para compilar un ejecutable .exe:
* instalar libreria de compilacion (si no la tiene):
    pip install pyinstaller
* luego ubique la consola en la carpeta de proyecto:
    cd ruta_de_carpeta_sin_incluir_el_.py
* la carpeta contiene:
    SoundRecognitionDSP.py (este codigo), icono.ico, img*.png (* de 0 a 17)
* ejecutar comando generado con funcion: compilador(18), algo asi:
    pyinstaller -y -F -i "icono.ico" "SoundRecognitionDSP.py" ... etc
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox,\
    QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton, QLineEdit,\
    QGridLayout, QLabel, QFileDialog, QSizePolicy
from PyQt5.QtCore import Qt, QMargins, QThread, QPointF
from PyQt5.QtGui import QIcon
from PyQt5.QtChart import QChartView, QLineSeries
import numpy as np
from numpy.fft import fftn
from scipy.fftpack import dct
from python_speech_features import mfcc
import sounddevice as sd
import soundfile as sf

# la funcion principal o inicializadora
def main():
    # para que al ejecutar en forma .exe, se hallen los assets
    if getattr(sys, "frozen", False):
        os.chdir(sys._MEIPASS)

    # lanzar la GUI
    app = QApplication(sys.argv)
    gui = GUI("v1.0.0")
    gui.setStyleSheet(estilo(app.desktop().screenGeometry().height()))
    gui.show()

    # ejecutar aplicacion hasta que no haya ventana visible
    sys.exit(app.exec_())

class GUI(QWidget):

    def __init__(self, version):
        QWidget.__init__(self)
        self.version = version
        self.setWindowTitle("Sound Recognition DSP UV")
        self.setWindowIcon(QIcon("img0.png"))

        # crear hilos
        self.hiloTestOpt = HiloTestOpt()
        self.hiloTestOpt.finished.connect(self.finHiloTestOpt)
        self.hiloTestLow = HiloTestLow()
        self.hiloTestLow.finished.connect(self.finHiloTestLow)
        self.hiloRecord = HiloRecord()
        self.hiloRecord.finished.connect(self.finHiloRecord)
        self.hiloExtractOpt = HiloExtractOpt()
        self.hiloExtractOpt.finished.connect(self.finHiloExtractOpt)
        self.hiloExtractLow = HiloExtractLow()
        self.hiloExtractLow.finished.connect(self.finHiloExtractLow)
        self.hiloNewNet = HiloNewNet()
        self.hiloNewNet.finished.connect(self.finHiloNewNet)
        self.hiloTrainNet = HiloTrainNet()
        self.hiloTrainNet.finished.connect(self.finHiloTrainNet)
        self.hiloAccuracyNet = HiloAccuracyNet()
        self.hiloAccuracyNet.finished.connect(self.finHiloAccuracyNet)

        # variables del programa
        self.ejecucion = False
        self.Fs = 16000
        self.voz = np.zeros(0, dtype=float)
        # 1 de pitch, 13 datos de MFCC y 1 es la clase
        self.patrones = np.zeros((0, 15), dtype=float)
        # pesos sinapticos de red DMNN y numero de dendritas / neurona
        self.pesW = np.array([0.0])
        self.numK = np.array([0])
        # variables de la GUI
        self.turnoBajoCut = True
        self.abortar = False

        # crear la GUI como tal
        self.crearGUI(True, 16)

    def crearGUI(self, artesanal, cuantos):
        fondo = QHBoxLayout()
        subfondo = QVBoxLayout()
        subfondo.addWidget(self.moduloGUIaudio(artesanal))
        subfondo.addWidget(self.moduloGUItrain(artesanal))
        fondo.addLayout(subfondo)
        fondo.addWidget(self.moduloGUIpatterns(cuantos))

        # escalamiento
        fondo.setStretch(0, 3)
        fondo.setStretch(1, 1)
        subfondo.setStretch(0, 2)
        subfondo.setStretch(1, 1)

        self.setLayout(fondo)

    def moduloGUIaudio(self, artesanal):
        # crear cajon de grupo con titulo
        grupo = QGroupBox("AUDIO")
        fondo1 = QVBoxLayout()
        fondo2 = QHBoxLayout()

        # grupo de botones de administracion
        admin = QGroupBox("Administration")
        fondo3 = QHBoxLayout()
        # boton importacion
        aux = QPushButton(QIcon("img1.png"), "")
        aux.clicked.connect(self.importAudio)
        fondo3.addWidget(aux)
        # boton exportacion
        aux = QPushButton(QIcon("img2.png"), "")
        aux.clicked.connect(self.exportAudio)
        fondo3.addWidget(aux)
        # boton acerca de
        aux = QPushButton(QIcon("img8.png"), "")
        aux.setToolTip(tooltips("about"))
        aux.clicked.connect(self.acercade)
        fondo3.addWidget(aux)
        # boton reproducir audio
        aux = QPushButton(QIcon("img5.png"), "")
        aux.setToolTip(tooltips("play"))
        aux.clicked.connect(self.play)
        fondo3.addWidget(aux)
        # boton detener audio
        aux = QPushButton(QIcon("img6.png"), "")
        aux.setToolTip(tooltips("stop"))
        aux.clicked.connect(self.stop)
        fondo3.addWidget(aux)
        # agregar a grupo superior
        pedazo = QVBoxLayout()
        pedazo.addSpacing(10)
        pedazo.addLayout(fondo3)
        admin.setLayout(pedazo)
        fondo2.addWidget(admin)

        # grupo para manejar la clase
        etiqueta = QGroupBox("Class")
        fondo3 = QHBoxLayout()
        # boton extraccion optimizada
        aux = QPushButton(QIcon("img3.png"), "")
        aux.setToolTip(tooltips("extractOpt"))
        aux.clicked.connect(self.extractOpt)
        fondo3.addWidget(aux)
        # boton extraccion artesanal
        if artesanal:
            aux = QPushButton(QIcon("img4.png"), "")
            aux.setToolTip(tooltips("extractLow"))
            aux.clicked.connect(self.extractLow)
            fondo3.addWidget(aux)
        # texto con el nombre de la clase
        self.textEtiqueta = QLineEdit("")
        self.textEtiqueta.setAlignment(Qt.AlignCenter)
        self.textEtiqueta.setMaxLength(12)
        self.textEtiqueta.setToolTip(tooltips("clase"))
        fondo3.addWidget(self.textEtiqueta)
        # texto de cantidad de muestras a unir
        self.textCompact = QLineEdit("10")
        self.textCompact.setAlignment(Qt.AlignRight)
        self.textCompact.setMaxLength(6)
        self.textCompact.setToolTip(tooltips("compact"))
        fondo3.addWidget(self.textCompact)
        # agregar a grupo superior
        pedazo = QVBoxLayout()
        pedazo.addSpacing(10)
        pedazo.addLayout(fondo3)
        etiqueta.setLayout(pedazo)
        fondo2.addWidget(etiqueta)

        # grupo de corte de signal
        corte = QGroupBox("Cut Signal")
        fondo3 = QHBoxLayout()
        # boton de cortar banda
        aux = QPushButton(QIcon("img11.png"), "")
        aux.setToolTip(tooltips("cutBand"))
        aux.clicked.connect(self.cutSignalBand)
        fondo3.addWidget(aux)
        # boton de cortar extremos
        aux = QPushButton(QIcon("img12.png"), "")
        aux.setToolTip(tooltips("cutOuter"))
        aux.clicked.connect(self.cutSignalOuter)
        fondo3.addWidget(aux)
        # texto de minimo corte
        self.textCutMin = QLineEdit("0")
        self.textCutMin.setAlignment(Qt.AlignRight)
        self.textCutMin.setMaxLength(6)
        self.textCutMin.setToolTip(tooltips("cutMin"))
        fondo3.addWidget(self.textCutMin)
        # texto de maximo corte
        self.textCutMax = QLineEdit("1")
        self.textCutMax.setAlignment(Qt.AlignRight)
        self.textCutMax.setMaxLength(6)
        self.textCutMax.setToolTip(tooltips("cutMax"))
        fondo3.addWidget(self.textCutMax)
        # agregar a grupo superior
        pedazo = QVBoxLayout()
        pedazo.addSpacing(10)
        pedazo.addLayout(fondo3)
        corte.setLayout(pedazo)
        fondo2.addWidget(corte)

        # grupo de grabacion por microfono
        microfono = QGroupBox("Record")
        fondo3 = QHBoxLayout()
        # boton de grabacion
        aux = QPushButton(QIcon("img7.png"), "")
        aux.setToolTip(tooltips("record"))
        aux.clicked.connect(self.recordSignal)
        fondo3.addWidget(aux)
        # texto de segundos de grabacion
        self.textRecS = QLineEdit("3")
        self.textRecS.setAlignment(Qt.AlignRight)
        self.textRecS.setMaxLength(6)
        self.textRecS.setToolTip(tooltips("recTime"))
        fondo3.addWidget(self.textRecS)
        # agregar a grupo superior
        pedazo = QVBoxLayout()
        pedazo.addSpacing(10)
        pedazo.addLayout(fondo3)
        microfono.setLayout(pedazo)
        fondo2.addWidget(microfono)

        # agregar la grafica de audio en segundos
        grafica = QGroupBox("Signal vs Seconds")
        self.plotAudio = QChartView()
        self.plotAudio.chart().setDropShadowEnabled(False)
        self.plotAudio.chart().setMargins(QMargins(0, 0, 0, 0))

        # juntar las cosas al final
        fondo1.addLayout(fondo2)
        pedazo = QVBoxLayout()
        pedazo.addSpacing(10)
        pedazo.addWidget(self.plotAudio)
        grafica.setLayout(pedazo)
        fondo1.addWidget(grafica)
        pedazo = QVBoxLayout()
        pedazo.addSpacing(10)
        pedazo.addLayout(fondo1)
        grupo.setLayout(pedazo)

        # escalamiento
        aux = QSizePolicy()
        aux.setHorizontalPolicy(QSizePolicy.MinimumExpanding)
        self.textEtiqueta.setSizePolicy(aux)
        fondo1.setStretch(0, 1)
        fondo1.setStretch(1, 3)
        fondo2.setStretch(0, 1)
        fondo2.setStretch(1, 3)
        fondo2.setStretch(2, 1)
        fondo2.setStretch(3, 1)

        return grupo

    def moduloGUItrain(self, artesanal):
        # crear cajon de grupo con titulo
        grupo = QGroupBox("TRAIN / TEST")
        fondo1 = QHBoxLayout()
        fondo2 = QVBoxLayout()

        # grupo de botones de administracion
        admin = QGroupBox("Administration")
        fondo3 = QHBoxLayout()
        # boton importacion
        aux = QPushButton(QIcon("img1.png"), "")
        aux.clicked.connect(self.importNet)
        fondo3.addWidget(aux)
        # boton exportacion
        aux = QPushButton(QIcon("img2.png"), "")
        aux.clicked.connect(self.exportNet)
        fondo3.addWidget(aux)
        # boton ejecutar inicializacion
        aux = QPushButton(QIcon("img17.png"), "")
        aux.setToolTip(tooltips("netNew"))
        aux.clicked.connect(self.newNet)
        fondo3.addWidget(aux)
        # boton ejecutar entrenamiento
        aux = QPushButton(QIcon("img10.png"), "")
        aux.setToolTip(tooltips("netTrain"))
        aux.clicked.connect(self.trainNet)
        fondo3.addWidget(aux)
        # boton hallar precision usando todos los patrones
        aux = QPushButton(QIcon("img16.png"), "")
        aux.setToolTip(tooltips("accuracy"))
        aux.clicked.connect(self.accuracyNet)
        fondo3.addWidget(aux)
        # boton testear audio actual mediante optimo
        aux = QPushButton(QIcon("img14.png"), "")
        aux.setToolTip(tooltips("netTestOpt"))
        aux.clicked.connect(self.testNetOpt)
        fondo3.addWidget(aux)
        # boton testear audio actual mediante artesanal
        if artesanal:
            aux = QPushButton(QIcon("img15.png"), "")
            aux.setToolTip(tooltips("netTestLow"))
            aux.clicked.connect(self.testNetLow)
            fondo3.addWidget(aux)
        # agregar a grupo superior
        pedazo = QVBoxLayout()
        pedazo.addSpacing(10)
        pedazo.addLayout(fondo3)
        admin.setLayout(pedazo)
        fondo2.addWidget(admin)

        # grupo de parametros de entreno e informacion
        parametrous = QGroupBox("Train Parameters")
        fondo3 = QGridLayout()
        # arriba
        # texto de parametro clusters
        self.textClusters = QLineEdit("10")
        self.textClusters.setAlignment(Qt.AlignRight)
        self.textClusters.setMaxLength(6)
        self.textClusters.setToolTip(tooltips("netClusters"))
        fondo3.addWidget(self.textClusters, 0, 0)
        # texto de parametro dimension hipercajas
        self.textHipercaja = QLineEdit("10")
        self.textHipercaja.setAlignment(Qt.AlignRight)
        self.textHipercaja.setMaxLength(6)
        self.textHipercaja.setToolTip(tooltips("netBoxSize"))
        fondo3.addWidget(self.textHipercaja, 0, 1)
        # texto de parametro mutacion genetica
        self.textMutacion = QLineEdit("1")
        self.textMutacion.setAlignment(Qt.AlignRight)
        self.textMutacion.setMaxLength(6)
        self.textMutacion.setToolTip(tooltips("netMutar"))
        fondo3.addWidget(self.textMutacion, 0, 2)
        # texto de parametro de iteraciones
        self.textIteracion = QLineEdit("100")
        self.textIteracion.setAlignment(Qt.AlignRight)
        self.textIteracion.setMaxLength(6)
        self.textIteracion.setToolTip(tooltips("netItera"))
        fondo3.addWidget(self.textIteracion, 0, 3)
        # abajo
        # texto de numero de pesos sinapticos
        self.textPesoW = QLabel("W: 0")
        self.textPesoW.setToolTip(tooltips("infoW"))
        self.textPesoW.setAlignment(Qt.AlignLeft)
        fondo3.addWidget(self.textPesoW, 1, 0)
        # texto de accuracy global
        self.textAccuracy = QLabel("Acc%: 0")
        self.textAccuracy.setToolTip(tooltips("infoAcc"))
        self.textAccuracy.setAlignment(Qt.AlignLeft)
        fondo3.addWidget(self.textAccuracy, 1, 1)
        # texto de alguna otra cosa
        self.textEstado = QLabel("...")
        self.textEstado.setToolTip(tooltips("estado"))
        self.textEstado.setAlignment(Qt.AlignLeft)
        fondo3.addWidget(self.textEstado, 1, 2)
        # texto de porcentaje entrenado
        self.textGo = QLabel("Go%: 0")
        self.textGo.setToolTip(tooltips("infoGo"))
        self.textGo.setAlignment(Qt.AlignLeft)
        fondo3.addWidget(self.textGo, 1, 3)
        # agregar a grupo superior
        pedazo = QVBoxLayout()
        pedazo.addSpacing(10)
        pedazo.addLayout(fondo3)
        parametrous.setLayout(pedazo)
        fondo2.addWidget(parametrous)

        # agregar la grafica de entreno
        grafica = QGroupBox("Error vs Iterations")
        self.plotTrain = QChartView()
        self.plotTrain.chart().setDropShadowEnabled(False)
        self.plotTrain.chart().setMargins(QMargins(0, 0, 0, 0))

        # juntar las cosas al final
        fondo1.addLayout(fondo2)
        pedazo = QVBoxLayout()
        pedazo.addSpacing(10)
        pedazo.addWidget(self.plotTrain)
        grafica.setLayout(pedazo)
        fondo1.addWidget(grafica)
        pedazo = QVBoxLayout()
        pedazo.addSpacing(10)
        pedazo.addLayout(fondo1)
        grupo.setLayout(pedazo)

        # escalamiento
        fondo1.setStretch(0, 1)
        fondo1.setStretch(1, 3)
        fondo2.setStretch(0, 1)
        fondo2.setStretch(1, 2)

        return grupo

    def moduloGUIpatterns(self, cuantos):
        # crear cajon de grupo con titulo
        grupo = QGroupBox("PATTERNS")
        fondo1 = QVBoxLayout()

        # grupo de botones de administracion
        admin = QGroupBox("Administration")
        fondo3 = QHBoxLayout()
        # boton importacion
        aux = QPushButton(QIcon("img1.png"), "")
        aux.clicked.connect(self.importPatterns)
        fondo3.addWidget(aux)
        # boton exportacion
        aux = QPushButton(QIcon("img2.png"), "")
        aux.clicked.connect(self.exportPatterns)
        fondo3.addWidget(aux)
        # boton cortar parte de los patrones al azar
        aux = QPushButton(QIcon("img13.png"), "")
        aux.setToolTip(tooltips("patCut"))
        aux.clicked.connect(self.patternsCut)
        fondo3.addWidget(aux)
        # boton eliminar patrones
        aux = QPushButton(QIcon("img9.png"), "")
        aux.setToolTip(tooltips("patClean"))
        aux.clicked.connect(self.patternsClean)
        fondo3.addWidget(aux)
        # agregar a grupo superior
        pedazo = QVBoxLayout()
        pedazo.addSpacing(10)
        pedazo.addLayout(fondo3)
        admin.setLayout(pedazo)
        fondo1.addWidget(admin)

        # agregar titulo de los patrones
        self.textTituPat = QLabel("(0) ...")
        self.textTituPat.setToolTip(tooltips("infoTitle"))
        self.textTituPat.setAlignment(Qt.AlignCenter)
        fondo1.addWidget(self.textTituPat)

        # crear ciclicamente la matrix de labels para mostrar info
        matrix = QGridLayout()
        aux = QLabel("Class Name")
        aux.setAlignment(Qt.AlignLeft)
        matrix.addWidget(aux, 0, 0)
        aux = QLabel("Tot%")
        aux.setToolTip(tooltips("infoPercent"))
        aux.setAlignment(Qt.AlignRight)
        matrix.addWidget(aux, 0, 1)
        aux = QLabel("Pre%")
        aux.setToolTip(tooltips("infoExac"))
        aux.setAlignment(Qt.AlignRight)
        matrix.addWidget(aux, 0, 2)
        aux = QLabel("Sen%")
        aux.setToolTip(tooltips("infoSens"))
        aux.setAlignment(Qt.AlignRight)
        matrix.addWidget(aux, 0, 3)
        aux = QLabel("Res%")
        aux.setToolTip(tooltips("infoTest"))
        aux.setAlignment(Qt.AlignRight)
        matrix.addWidget(aux, 0, 4)
        # empezar ciclo de agregar labels
        self.className = []
        self.classNumber = []
        self.classExacti = []
        self.classSensi = []
        self.classResult = []
        for c in range(1, cuantos + 1):
            self.className.append(QLabel("..."))
            self.className[-1].setAlignment(Qt.AlignLeft)
            matrix.addWidget(self.className[-1], c, 0)
            self.classNumber.append(QLabel(""))
            self.classNumber[-1].setAlignment(Qt.AlignRight)
            matrix.addWidget(self.classNumber[-1], c, 1)
            self.classExacti.append(QLabel(""))
            self.classExacti[-1].setAlignment(Qt.AlignRight)
            matrix.addWidget(self.classExacti[-1], c, 2)
            self.classSensi.append(QLabel(""))
            self.classSensi[-1].setAlignment(Qt.AlignRight)
            matrix.addWidget(self.classSensi[-1], c, 3)
            self.classResult.append(QLabel(""))
            self.classResult[-1].setAlignment(Qt.AlignRight)
            matrix.addWidget(self.classResult[-1], c, 4)
        # agregar a grupo superior
        fondo1.addLayout(matrix)

        # juntar las cosas al final
        pedazo = QVBoxLayout()
        pedazo.addSpacing(10)
        pedazo.addLayout(fondo1)
        grupo.setLayout(pedazo)

        # escalamiento
        fondo1.setStretch(0, 1)
        fondo1.setStretch(1, 1)
        fondo1.setStretch(2, 10)

        return grupo

    def mousePressEvent(self, e):
        if self.plotAudio.underMouse():
            serie = self.plotAudio.chart().series()
            if len(serie) > 0:
                des = 0
                punto = QPointF(e.pos().x(), 0)
                charX = self.plotAudio.chart().mapToValue(punto, serie[0]).x() - des
                if self.turnoBajoCut:
                    self.turnoBajoCut = False
                    self.textCutMin.setText(str(round(charX, 2)))
                else:
                    self.turnoBajoCut = True
                    self.textCutMax.setText(str(round(charX, 2)))

    def importAudio(self):
        fileDir, _ = QFileDialog.getOpenFileName(caption="Import Audio",
                                                 filter="Audio File (*.wav)")
        if fileDir:
            try:
                self.voz, self.Fs = sf.read(fileDir)
                try:
                    self.voz = self.voz[:, 0]
                except:
                    pass
                self.graphLine(self.plotAudio, self.voz, self.Fs)
                titulo = os.path.basename(fileDir).replace(".wav", "")
                self.textEtiqueta.setText("".join(k for k in titulo if k.isalpha()))
                self.cambiarFrecuencia(16000)
            except:
                self.Fs = 16000
                self.voz = np.zeros(0, dtype=float)
                QMessageBox.about(self, "Error!", "cant open the file...")

    def cambiarFrecuencia(self, Fs):
        if Fs != self.Fs:
            oriY = self.voz.copy()
            oriX = np.arange(oriY.size) / self.Fs
            newX = np.arange(oriY.size) / Fs
            self.voz = np.interp(newX, oriX, oriY)
        self.Fs = Fs

    def exportAudio(self):
        fileDir, _ = QFileDialog.getSaveFileName(caption="Export Audio",
                                                 filter="Audio File (*.wav)")
        if fileDir:
            try:
                sf.write(fileDir, self.voz, self.Fs)
            except:
                QMessageBox.about(self, "Error!", "cant export audio...")

    def extractOpt(self):
        self.generalExtract(self.hiloExtractOpt)

    def extractLow(self):
        self.generalExtract(self.hiloExtractLow)

    def generalExtract(self, hilo):
        if self.voz.size == 0:
            QMessageBox.about(self, "Advice", "need audio to work...")
        else:
            if self.textEtiqueta.text() == "":
                QMessageBox.about(self, "Advice", "write some class name...")
            else:
                if not self.ejecucion:
                    self.ejecucion = True
                    self.textEstado.setText("Ext...")
                    hilo.voz = self.voz.copy()
                    hilo.Fs = self.Fs
                    hilo.name = self.textEtiqueta.text()
                    try:
                        hilo.particion = int(self.textCompact.text())
                    except:
                        hilo.particion = 10
                    hilo.start()
                else:
                    QMessageBox.about(self, "Wait!", "process in execution...")

    def play(self):
        try:
            sd.play(self.voz.copy(), self.Fs)
        except:
            QMessageBox.about(self, "Error!", "cant play...")

    def stop(self):
        sd.stop(True)
        self.abortar = True

    def cutSignalBand(self):
        self.cutSignal(True)

    def cutSignalOuter(self):
        self.cutSignal(False)

    def cutSignal(self, isBand):
        try:
            if self.textCutMin.text() == "":
                limInf = 0
            else:
                limInf = int(float(self.textCutMin.text()) * self.Fs)
            if self.textCutMax.text() == "":
                limSup = self.voz.size
            else:
                limSup = int(float(self.textCutMax.text()) * self.Fs)
            limInf = min(max(limInf, 0), self.voz.size - 1)
            limSup = min(max(limSup, 1), self.voz.size)
            if limInf >= limSup:
                limSup = limInf + 1
            if isBand:
                self.voz = self.voz[limInf: limSup]
            else:
                self.voz = np.append(self.voz[0: limInf], self.voz[limSup:])
            self.graphLine(self.plotAudio, self.voz, self.Fs)
        except:
            QMessageBox.about(self, "Error!", "cant cut signal...")

    def recordSignal(self):
        if not self.ejecucion:
            self.ejecucion = True
            self.textEstado.setText("Rec...")
            try:
                self.hiloRecord.tiempo = min(30.0, max(1.0, float(self.textRecS.text())))
            except:
                self.hiloRecord.tiempo = 1.0
            self.hiloRecord.start()
        else:
            QMessageBox.about(self, "Wait!", "process in execution...")

    def importNet(self):
        fileDir, _ = QFileDialog.getOpenFileName(caption="Import DMNN",
                                                 filter="Text File (*.txt)")
        if fileDir:
            file = open(fileDir, "r")
            txt = file.read().split("\n")
            file.close()
            try:
                if txt[0].find("DMNN: ") == 0:
                    pesos = txt[4].split(",")
                    pesW = []
                    for p in pesos:
                        pesW.append(float(p))
                    self.pesW = np.array(pesW, dtype=float)
                    dendritas = txt[6].split(",")
                    numK = []
                    for d in dendritas:
                        numK.append(int(d))
                    self.numK = np.array(numK, dtype=int)
                    self.textPesoW.setText("W: " + str(self.pesW.size))
                    self.hiloTrainNet.error = np.zeros(0, dtype=float)
            except:
                QMessageBox.about(self, "Error!", "file cant be open...")

    def exportNet(self):
        fileDir, _ = QFileDialog.getSaveFileName(caption="Export DMNN",
                                                 filter="Text File (*.txt)")
        if fileDir:
            # crear el archivo y escribir las cabeceras
            file = open(fileDir, "w")
            try:
                file.write("DMNN: SoundRecognitionDSP\n")
                file.write("Dimension: Entradas, Clases\n")
                file.write("14,3\n")
                file.write("Pesos\n")
                txt = ""
                for p in range(self.pesW.size):
                    txt += str(self.pesW[p]) + ","
                txt = txt[:-1] + "\n"
                file.write(txt)
                file.write("DendritasPorClase\n")
                txt = ""
                for n in range(self.numK.size):
                    txt += str(self.numK[n]) + ","
                txt = txt[:-1] + "\n"
                file.write(txt)
                file.write("Activas\n")
                txt = ""
                for i in range(self.numK.sum()):
                    txt += "1,"
                txt = txt[:-1] + "\n"
                file.write(txt)
                file.write("NormalizacionH\n")
                txt = ""
                for i in range(14):
                    txt += "1.,"
                txt = txt[:-1] + "\n"
                file.write(txt)
                file.write("NormalizacionL\n")
                txt = ""
                for i in range(14):
                    txt += "-1.,"
                txt = txt[:-1] + "\n"
                file.write(txt)
                file.write("NormalizacionN: 0.0\n")
                file.write("NombresSalidas: ")
                txt = ""
                for i in range(self.numK.size):
                    txt += "Class" + str(i) + ", "
                txt = txt[:-2] + "\n"
                file.write(txt)
                file.write("NombresEntradas: Pitch, n0, n1, n2, n3, n4, n5, " +
                           "n6, n7, n8, n9, n10, n11, n12\n")
            except:
                QMessageBox.about(self, "Error!", "invalid data to write...")
            file.close()

    def newNet(self):
        if self.patrones.size == 0:
            QMessageBox.about(self, "Advice!", "need patterns to run...")
        else:
            if not self.ejecucion:
                self.ejecucion = True
                self.textEstado.setText("Ini...")
                self.hiloNewNet.patrones = self.patrones.copy()
                try:
                    self.hiloNewNet.clusters = int(self.textClusters.text())
                except:
                    self.hiloNewNet.clusters = 1
                try:
                    self.hiloNewNet.iteraciones = int(self.textIteracion.text())
                except:
                    self.hiloNewNet.iteraciones = 100
                try:
                    self.hiloNewNet.dimCajas = float(self.textHipercaja.text())
                except:
                    self.hiloNewNet.dimCajas = 10.0
                self.hiloNewNet.start()
            else:
                QMessageBox.about(self, "Wait!", "process in execution...")

    def trainNet(self):
        if self.pesW.size == 0 or self.patrones.size == 0:
            QMessageBox.about(self, "Advice!", "need data to work...")
        else:
            if not self.ejecucion:
                self.ejecucion = True
                self.textEstado.setText("Tra...")
                self.hiloTrainNet.patrones = self.patrones.copy()
                self.hiloTrainNet.pesW = self.pesW.copy()
                self.hiloTrainNet.numK = self.numK.copy()
                try:
                    muta = float(self.textMutacion.text()) / 100.0
                except:
                    muta = 0.01
                self.hiloTrainNet.muta = np.max(self.patrones[:, :-1]) * muta
                try:
                    self.hiloTrainNet.iteracion = [0, max(1, int(self.textIteracion.text()))]
                except:
                    self.hiloTrainNet.iteracion = [0, 100]
                if self.hiloTrainNet.error.size == 0:
                    e = self.hiloTrainNet.funError(self.hiloTrainNet.pesW)
                    self.hiloTrainNet.error = np.array([0, e], dtype=float)
                self.abortar = False
                self.hiloTrainNet.start()
            else:
                QMessageBox.about(self, "Wait!", "process in execution...")

    def accuracyNet(self):
        if not self.ejecucion:
            self.ejecucion = True
            self.textEstado.setText("Acc...")
            self.hiloAccuracyNet.pesW = self.pesW.copy()
            self.hiloAccuracyNet.numK = self.numK.copy()
            self.hiloAccuracyNet.patrones = self.patrones.copy()
            self.hiloAccuracyNet.start()
        else:
            QMessageBox.about(self, "Wait!", "process in execution...")

    def testNetOpt(self):
        self.generalTest(self.hiloTestOpt)

    def testNetLow(self):
        self.generalTest(self.hiloTestLow)

    def generalTest(self, hilo):
        if self.voz.size == 0 or np.shape(self.patrones)[0] == 0 or self.pesW.size == 0:
            QMessageBox.about(self, "Advice", "need audio or net to work...")
        else:
            if not self.ejecucion:
                self.ejecucion = True
                self.textEstado.setText("Tes...")
                hilo.voz = self.voz.copy()
                hilo.Fs = self.Fs
                hilo.pesW = self.pesW
                hilo.numK = self.numK
                try:
                    hilo.particion = int(self.textCompact.text())
                except:
                    hilo.particion = 10
                hilo.start()
            else:
                QMessageBox.about(self, "Wait!", "process in execution...")

    def importPatterns(self):
        fileDir, _ = QFileDialog.getOpenFileName(caption="Import Patterns",
                                                 filter="Text File (*.txt)")
        if fileDir:
            file = open(fileDir, "r")
            txt = file.read().split("\n")
            file.close()
            try:
                if txt[0].find("Patrones: ") == 0 and txt[1].find("Salidas: ") == 0 and\
                        txt[2].find("Entradas: ") == 0:
                    self.limpiarInfo(True)
                    self.textTituPat.setText("(0) " + txt[0][10:])
                    names = txt[1][9:].split(", ")
                    for i in range(len(names)):
                        self.className[i].setText(names[i])
                    patrones = []
                    txt = txt[3:]
                    for p in range(len(txt)):
                        if txt[p] != "":
                            patrones.append([])
                            nums = txt[p].split(", ")
                            for n in range(15):
                                patrones[-1].append(float(nums[n]))
                    self.patrones = np.array(patrones)
                    self.hiloTrainNet.error = np.zeros(0, dtype=float)
                    self.calculaInfoPatrones()
            except:
                QMessageBox.about(self, "Error!", "invalid format...")

    def exportPatterns(self):
        fileDir, _ = QFileDialog.getSaveFileName(caption="Export Patterns",
                                                 filter="Text File (*.txt)")
        if fileDir:
            # obtener el nombre del set de patrones y ponerlo en GUI
            titulo = os.path.basename(fileDir).replace(".txt", "")
            total = self.textTituPat.text().split(")")
            self.textTituPat.setText(total[0] + ") " + titulo)
            # crear el archivo y escribir las cabeceras
            file = open(fileDir, "w")
            try:
                file.write("Patrones: " + titulo + "\n")
                file.write("Salidas: ")
                names = ""
                for i in range(len(self.className)):
                    if self.className[i].text() != "...":
                        names += self.className[i].text() + ", "
                names = names[:-2] + "\n"
                file.write(names)
                file.write("Entradas: Pitch")
                for i in range(13):
                    file.write(", n" + str(i))
                file.write("\n")
                # escribir los datos como tal
                np.random.shuffle(self.patrones)
                txt = ""
                for p in range(np.shape(self.patrones)[0]):
                    for n in range(15):
                        txt += str(self.patrones[p, n]) + ", "
                    txt = txt[:-2] + "\n"
                file.write(txt)
            except:
                QMessageBox.about(self, "Error!", "invalid data to write...")
            file.close()

    def patternsCut(self):
        np.random.shuffle(self.patrones)
        antik = self.patrones.copy()
        inicio = int(np.shape(self.patrones)[0] * 0.1)
        self.patrones = self.patrones[inicio:, :]
        ok = True
        for i in range(len(self.className)):
            if self.className[i].text() != "...":
                if np.sum(self.patrones[:, -1] == i) == 0:
                    ok = False
                    break
        if ok:
            self.calculaInfoPatrones()
            self.hiloTrainNet.error = np.zeros(0, dtype=float)
            self.limpiarInfo(False)
        else:
            self.patrones = antik
            QMessageBox.about(self, "Advice!", "a class can be destroy...")

    def patternsClean(self):
        self.textTituPat.setText("(0) ...")
        self.hiloTrainNet.error = np.zeros(0, dtype=float)
        self.patrones = np.zeros((0, 15), dtype=float)
        self.limpiarInfo(True)

    def graphLine(self, axes, data, Fs):
        try:
            axes.chart().removeAllSeries()
            L = data.size
            tiempo = np.arange(L) / Fs
            paso = int(np.ceil(L / 3000))
            linea = QLineSeries()
            linea.setColor(Qt.blue)
            for x in range(0, L, paso):
                linea.append(tiempo[x], data[x])
            axes.chart().addSeries(linea)
            axes.chart().createDefaultAxes()
            axes.chart().legend().setVisible(False)
        except:
            pass

    def acercade(self):
        txt = "($$$) Software for Sound Recognition, here you\n" \
             "import or record audio, next a classification system\n" \
             "is trained to perform test, made for DSP class of\n" \
             "electronic engineery in University of Valle (Cali\n" \
             "Colombia 2020), creators:\n" \
             "- Omar Jordan\n" \
             "- Pablo Torres\n" \
             "- Harold Medina"
        txt = txt.replace("$$$", self.version)
        QMessageBox.about(self, "Acerca de SoundRecognitionDSP", txt)

    def finHiloRecord(self):
        self.ejecucion = False
        self.textEstado.setText("...")
        self.voz = self.hiloRecord.voz.copy()
        self.Fs = self.hiloRecord.Fs
        self.graphLine(self.plotAudio, self.voz, self.Fs)

    def finHiloExtractOpt(self):
        self.finGeneralExtract(self.hiloExtractOpt)

    def finHiloExtractLow(self):
        self.finGeneralExtract(self.hiloExtractLow)

    def finGeneralExtract(self, hilo):
        self.ejecucion = False
        self.textEstado.setText("...")
        # buscar si ya existe la clase, sino crearla
        ind = -1
        for i in range(len(self.className)):
            if self.className[i].text() == hilo.name:
                ind = i
                break
        if ind == -1:
            for i in range(len(self.className)):
                if self.className[i].text() == "...":
                    self.className[i].setText(hilo.name)
                    ind = i
                    break
        if ind == -1:
            QMessageBox.about(self, "Advice!", "no more slots for classes...")
        else:
            # agregar los datos a los patrones
            bloque = hilo.param.copy()
            aux = np.ones((np.shape(bloque)[0], 1)) * hilo.tono
            bloque = np.concatenate((aux, bloque), axis=1)
            aux = np.ones((np.shape(bloque)[0], 1)) * ind
            bloque = np.concatenate((bloque, aux), axis=1)
            self.patrones = np.concatenate((self.patrones, bloque), axis=0)
            # modifica los datos de informacion
            self.hiloTrainNet.error = np.zeros(0, dtype=float)
            self.calculaInfoPatrones()
            self.limpiarInfo(False)

    def finHiloNewNet(self):
        self.ejecucion = False
        self.textEstado.setText("...")
        self.pesW = self.hiloNewNet.pesW.copy()
        self.numK = self.hiloNewNet.numK.copy()
        self.textPesoW.setText("W: " + str(self.pesW.size))
        self.hiloTrainNet.error = np.zeros(0, dtype=float)
        self.accuracyNet()

    def finHiloTrainNet(self):
        self.textGo.setText("Go%: " + str(int((float(self.hiloTrainNet.iteracion[0]) /
                                               self.hiloTrainNet.iteracion[1]) * 100.0)))
        self.graphLine(self.plotTrain, self.hiloTrainNet.error, 0.1)
        if self.hiloTrainNet.iteracion[0] < self.hiloTrainNet.iteracion[1] and not self.abortar:
            self.hiloTrainNet.start()
        else:
            self.ejecucion = False
            self.textEstado.setText("...")
            self.pesW = self.hiloTrainNet.pesW.copy()
            self.accuracyNet()

    def finHiloAccuracyNet(self):
        self.ejecucion = False
        self.textEstado.setText("...")
        try:
            # limpiar las casillas
            for i in range(len(self.className)):
                self.classExacti[i].setText("")
                self.classSensi[i].setText("")
            # hacer los calculos para cada clase
            matrix = self.hiloAccuracyNet.matrix
            for i in range(np.shape(matrix)[0]):
                # hallar la exactitud en ingles precision
                num = float(matrix[i, i]) / max(1, matrix[:, i].sum())
                self.classExacti[i].setText(str(int(num * 100.0)))
                # hallar la sensibilidad
                num = float(matrix[i, i]) / max(1, matrix[i, :].sum())
                self.classSensi[i].setText(str(int(num * 100.0)))
            # agregar la accuracy general
            num = float(np.diagonal(matrix).sum()) / max(1, matrix.sum())
            self.textAccuracy.setText("Acc%: " + str(round(num * 100.0, 2)))
        except:
            pass

    def finHiloTestOpt(self):
        self.finGeneralTest(self.hiloTestOpt)

    def finHiloTestLow(self):
        self.finGeneralTest(self.hiloTestLow)

    def finGeneralTest(self, hilo):
        self.ejecucion = False
        self.textEstado.setText("...")
        # poner los resultados
        for i in range(len(self.classResult)):
            self.classResult[i].setText("")
        winner = np.argmax(hilo.prediction)
        for i in range(hilo.prediction.size):
            if i == winner:
                self.classResult[i].setText("(" + str(int(hilo.prediction[i] * 100.0)) + ")")
            else:
                self.classResult[i].setText(str(int(hilo.prediction[i] * 100.0)))

    def calculaInfoPatrones(self):
        # poner el numero de patrones para las clases en info
        for i in range(len(self.className)):
            if self.className[i].text() != "...":
                num = np.sum(self.patrones[:, -1] == i)
                num /= max(1, np.shape(self.patrones)[0])
                self.classNumber[i].setText(str(int(num * 100.0)))
        # agregar el total al titulo
        titulo = self.textTituPat.text().split(")")
        num = np.shape(self.patrones)[0]
        self.textTituPat.setText("(" + str(num) + ")" + titulo[1])

    def limpiarInfo(self, nombresTambien):
        self.textAccuracy.setText("Acc%: 0")
        for i in range(len(self.className)):
            if nombresTambien:
                self.className[i].setText("...")
                self.classNumber[i].setText("")
            self.classExacti[i].setText("")
            self.classSensi[i].setText("")
            self.classResult[i].setText("")

# funciones externas o globales

def sacarPitch(sound, fs):
    return 0.0

def meanTrozos(matrix, grupo):
    L = np.shape(matrix)
    output = np.zeros((0, L[1]), dtype=float)
    n = 0
    while n < L[0]:
        aux = np.mean(matrix[n: min(L[0], n + grupo), :], axis=0)
        output = np.append(output, np.atleast_2d(aux), axis=0)
        n += grupo
    return output.copy()

def ExecuteDMNN(entrada, pesW, numK, softmax):
    X = entrada.copy()
    while X.size < pesW.size / 2:
        X = np.hstack((X, entrada))
    W = pesW.copy().reshape(-1, 2)
    WH = W[:, 0] - X
    WL = X - W[:, 1]
    Wmki = np.minimum(WH, WL)
    Wmki = Wmki.reshape(-1, entrada.size)
    Smk = Wmki.min(axis=1)
    Zm = np.zeros(numK.size)
    n = 0
    for m in range(Zm.size):
        Zm[m] = Smk[n:(n + numK[m])].max()
        n += numK[m]
    if softmax:
        Zm = np.exp(Zm)
        Ym = Zm / min(Zm.sum(), 1000000.0)
        return Ym
    else:
        y = np.argmax(Zm)
        return y

def multiExecuteDMNN(param, tono, pesW, numK):
    L = np.shape(param)[0]
    entrada = param
    aux = np.ones((L, 1)) * tono
    entrada = np.concatenate((aux, entrada), axis=1)
    prediction = np.zeros(numK.size, dtype=float)
    for e in range(L):
        prediction += ExecuteDMNN(entrada[e, :], pesW, numK, True)
    prediction /= L
    return prediction

def ourMFCC(signal, fs, alpha=0.97, frame_size=20e-03, frame_overlap=10e-03,
            fft_points=320, fft_power=False, f_low=60, f_high=16e03,
            filter_order=14, MFCC_coef=13):
    samples = len(signal)
    # se hace un preenfasis, para borrar las frecuencias
    signal_alpha = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    # Encuentro el numero de muestra que se toma por frame y el numero de muestras de solapamiento
    frame_length, frame_step = int(round(frame_size * fs)), int(round(frame_overlap * fs))
    # total de frames que se generan parcialmente
    num_frames = int(np.ceil(float(abs(samples - frame_length) / frame_step)))
    # cada path va a tener el numero de muestras por solapamiento mas el tamaño de cada frame
    pad_signal_length = num_frames * frame_step + frame_length
    # Se añaden los datos extras debidos a los solapamientos
    zeros = np.zeros((pad_signal_length - samples))
    pad_signal = np.append(signal_alpha, zeros)
    # creamos los indices necesarios para cada frame
    index = np.tile(np.arange(0, frame_length), (num_frames, 1)) +\
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[index.astype(np.int32, copy=False)]
    # multiplico por ventana hamming los frames
    signal_frames = frames * np.hamming(frame_length)

    # hallo la magnitud de la transformada de fourier
    fft_frames = fftn(signal_frames)
    fft_frames_mag = np.abs(fft_frames)
    # Elijo los datos de la transformada de fourier
    fft_frames_mag = fft_frames_mag[:][:, 0: fft_points] / fft_points
    if fft_power:
        fft_frames_mag = (fft_frames_mag ** 2)

    # genero las ecuaciones para transformar los datos
    hz2mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    mel2hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)
    # se transforman los limites del filtro a unidades de mel
    mel_low = hz2mel(f_low)
    mel_high = hz2mel(f_high)
    # Se genera los puntos en escala de mel
    mel = np.linspace(mel_low, mel_high, filter_order + 2)
    # transforma la frecuencia del mel a herz
    mel_hz = mel2hz(mel)
    # Inicia la creacion de los coeficientes mel
    f_i = np.floor((fft_points + 1) * mel_hz) / fs
    # Reservo memoria para el filtro mel
    Hm = np.zeros((filter_order, int(np.floor(fft_points))))
    # Genero el filtro Mel
    for m in range(1, filter_order + 1):
        f_m_1 = int(f_i[m - 1])
        f_m = int(f_i[m])
        f_m_plus_1 = int(f_i[m + 1])
        for k in range(f_m_1, f_m):
            Hm[m - 1, k] = (k - f_i[m - 1]) / (f_i[m] - f_i[m - 1])
        for k in range(f_m, f_m_plus_1):
            Hm[m - 1, k] = (f_i[m + 1] - k) / (f_i[m + 1] - f_i[m])
    Hm = np.where(Hm == 0, np.finfo(float).eps, Hm)  # Numerical Stability

    frames_filtered = np.dot(fft_frames_mag, Hm.T)

    frames_filtered_log = 20 * np.log10(frames_filtered)

    miMFCC = dct(frames_filtered_log, type=2, axis=1, norm='ortho')
    miMFCC = miMFCC[:, 1: MFCC_coef + 1]

    return miMFCC

def estilo(esc):
    # aqui se editan los estilos de los widgets de la GUI
    txt = "font-size: $px; " \
          "QGroupBox { " \
          "background-color: rgba(255,255,255,90); " \
          "border: 1px solid gray; " \
          "border-radius: 15px; }; " \
          "icon-size: $$px $$px;"
    escala = esc / 1200.0
    txt = txt.replace("$$", str(int(32 * escala)))
    txt = txt.replace("$", str(int(16 * escala)))
    return txt

def tooltips(titulo):
    if titulo == "about":
        txt = "about the software"
    elif titulo == "extractOpt":
        txt = "extract audio features using optimal functions"
    elif titulo == "extractLow":
        txt = "extract audio features using our slow functions"
    elif titulo == "clase":
        txt = "name of class to add to patterns"
    elif titulo == "cutBand":
        txt = "split the audio, maintain the middle band"
    elif titulo == "cutOuter":
        txt = "split the audio, remove the middle band"
    elif titulo == "cutMin":
        txt = "minimum cut time in seconds, void zero"
    elif titulo == "cutMax":
        txt = "maximum cut time in seconds, void maximum"
    elif titulo == "record":
        txt = "start recording audio, wait 0.5 s to start"
    elif titulo == "recTime":
        txt = "record time in seconds, default 3 s"
    elif titulo == "compact":
        txt = "amount of data to compact with mean, default 10"
    elif titulo == "accuracy":
        txt = "find the performance metrics for patterns set"
    elif titulo == "netTestOpt":
        txt = "execute a test using the audio, with optimal extraction"
    elif titulo == "netTestLow":
        txt = "execute a test using the audio, with our slow extraction"
    elif titulo == "netClusters":
        txt = "number of clusters by class, default 10"
    elif titulo == "netBoxSize":
        txt = "size of hiper-boxes, default 10 %"
    elif titulo == "patCut":
        txt = "destroy 10 % of patterns randomly"
    elif titulo == "patClean":
        txt = "destroy all the patterns"
    elif titulo == "infoPercent":
        txt = "percentage of patterns belonging to each class"
    elif titulo == "infoSens":
        txt = "sensitivity, percentage of patterns correctly classified"
    elif titulo == "infoExac":
        txt = "precision, probability that the prediction is correct"
    elif titulo == "infoTest":
        txt = "probability of current prediction, maximum wins"
    elif titulo == "infoTitle":
        txt = "show the number of patterns, and problem title"
    elif titulo == "infoAcc":
        txt = "accuracy, relation between amount of true classified vs all"
    elif titulo == "infoW":
        txt = "number of synaptic weights of the net"
    elif titulo == "netNew":
        txt = "create a new DMNN (an ANN) dont need mutation parameter"
    elif titulo == "netMutar":
        txt = "mutation for genetic train, default 1 %"
    elif titulo == "netItera":
        txt = "number of iterations for initialization or train, default 100"
    elif titulo == "infoGo":
        txt = "show the percentage of train ok"
    elif titulo == "netTrain":
        txt = "execute the train of the net, dont need clusters or size parameters"
    elif titulo == "stop":
        txt = "stop the playing audio, or if training, abort the train"
    elif titulo == "play":
        txt = "play the audio"
    elif titulo == "estado":
        txt = "show the system internal state"
    else:
        txt = "?"
    return txt

def Kmedias(matrix, clusters, iteraciones):
    # crear las estructuras de datos y la inicializacion al azar
    L = np.shape(matrix)[0]
    aux = np.zeros((L, 1))
    puntos = np.concatenate((matrix, aux), axis=1)
    dMax = np.max(matrix, axis=0)
    dMin = np.min(matrix, axis=0)
    centros = np.random.random_sample((clusters, np.shape(matrix)[1]))
    centros = (dMax - dMin) * centros + dMin
    # comenzar el ciclo con sus dos partes
    i = 0
    while i < iteraciones:
        i += 1
        fin = True
        # asociar los puntos al centroide mas cercano
        for p in range(L):
            dist = np.sum(np.power(centros - puntos[p, :-1], 2), axis=1)
            puntos[p, -1] = np.argmin(dist)
        # mover los centroides a sus puntos asociados
        for c in range(clusters):
            viejo = centros[c, :].copy()
            t = 0.0
            encola = True
            for p in range(L):
                if puntos[p, -1] == c:
                    if encola:
                        encola = False
                        centros[c, :] *= 0
                    centros[c, :] += puntos[p, :-1]
                    t += 1.0
            centros[c, :] /= (t if t != 0 else 1.0)
            # verificar si hubo cambios
            if fin:
                if not (False in (viejo == centros[c, :])):
                    fin = False
        # frenar si no hubo cambios, termina antes de iteraciones
        if fin:
            break
    return centros

def inicializaDMNN(patrones, clusters, iteraciones, dimCajas):
    # calcular dimension de las hiper-cajas
    dMax = np.max(patrones[:, :-1], axis=0)
    dMin = np.min(patrones[:, :-1], axis=0)
    dim = (dMax - dMin) * 0.5 * (dimCajas / 100.0)
    # inicializar la red
    numK = np.ones(int(patrones[:, -1].max() + 1), dtype=int) * clusters
    pesW = np.array([])
    # ciclo para obtener los centroides
    for m in range(numK.size):
        cen = Kmedias(patrones[patrones[:, -1] == m, :-1], clusters, iteraciones)
        vH = (cen + dim).ravel()
        vL = (cen - dim).ravel()
        pesW = np.concatenate((pesW, np.dstack((vH, vL)).ravel()))
    return pesW, numK

# clases hilos

class HiloRecord(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.tiempo = 1.0
        self.Fs = 16000
        self.voz = np.zeros(0, dtype=float)

    def run(self):
        sd.stop(True)
        muestras = int((self.tiempo + 0.5) * self.Fs)
        record = sd.rec(muestras, samplerate=self.Fs, channels=1)
        sd.wait()
        inicial = int(0.5 * self.Fs)
        self.voz = record[inicial:, 0]
        self.voz = self.voz.astype(float)

class HiloTestOpt(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.voz = np.zeros(0, dtype=float)
        self.Fs = 16000
        self.pesW = np.array([0.0])
        self.numK = np.array([0])
        self.particion = 1
        self.prediction = np.zeros(1, dtype=float)

    def run(self):
        param = meanTrozos(mfcc(self.voz, self.Fs), self.particion)
        tono = sacarPitch(self.voz, self.Fs)
        self.prediction = multiExecuteDMNN(param, tono, self.pesW, self.numK)

class HiloTestLow(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.voz = np.zeros(0, dtype=float)
        self.Fs = 16000
        self.pesW = np.array([0.0])
        self.numK = np.array([0])
        self.particion = 1
        self.prediction = np.zeros(1, dtype=float)

    def run(self):
        param = meanTrozos(ourMFCC(self.voz, self.Fs), self.particion)
        tono = sacarPitch(self.voz, self.Fs)
        self.prediction = multiExecuteDMNN(param, tono, self.pesW, self.numK)

class HiloExtractOpt(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.voz = np.zeros(0, dtype=float)
        self.Fs = 16000
        self.name = ""
        self.particion = 1
        self.param = np.zeros((0, 13), dtype=float)
        self.tono = 0

    def run(self):
        self.param = meanTrozos(mfcc(self.voz, self.Fs), self.particion)
        self.tono = sacarPitch(self.voz, self.Fs)

class HiloExtractLow(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.voz = np.zeros(0, dtype=float)
        self.Fs = 16000
        self.name = ""
        self.particion = 1
        self.param = np.zeros((0, 13), dtype=float)
        self.tono = 0

    def run(self):
        self.param = meanTrozos(ourMFCC(self.voz, self.Fs), self.particion)
        self.tono = sacarPitch(self.voz, self.Fs)

class HiloNewNet(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.clusters = 1
        self.iteraciones = 100
        self.dimCajas = 10
        self.patrones = np.zeros((0, 15), dtype=float)
        self.pesW = np.array([0.0])
        self.numK = np.array([0])

    def run(self):
        self.pesW, self.numK = inicializaDMNN(self.patrones, self.clusters,
                                              self.iteraciones, self.dimCajas)

class HiloTrainNet(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.patrones = np.zeros((0, 15), dtype=float)
        self.pesW = np.array([0.0])
        self.numK = np.array([0])
        self.iteracion = [0, 100]
        self.muta = 1.0
        self.error = np.zeros(1, dtype=float)

    def run(self):
        error = self.error[-1]
        limit = min(self.iteracion[0] + 10, self.iteracion[1])
        while self.iteracion[0] < limit:
            self.iteracion[0] += 1
            hijo = self.pesW.copy() + (np.random.rand(self.pesW.size) * 2.0 - 1.0) * self.muta
            newerror = self.funError(hijo)
            if newerror <= error:
                if newerror == 0:
                    self.iteracion[0] = self.iteracion[1]
                self.pesW = hijo.copy()
                error = newerror
        self.error = np.append(self.error, error)

    def funError(self, pesW):
        L = np.shape(self.patrones)[0]
        res = 0.0
        for p in range(L):
            if self.patrones[p, -1] == ExecuteDMNN(self.patrones[p, :-1],
                                                   pesW, self.numK, False):
                res += 1.0
        return 1.0 - (res / L)

class HiloAccuracyNet(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.patrones = np.zeros((0, 15), dtype=float)
        self.pesW = np.array([0.0])
        self.numK = np.array([0])
        self.matrix = np.zeros((0, 0), dtype=int)

    def run(self):
        self.matrix = np.zeros((self.numK.size, self.numK.size), dtype=int)
        try:
            for p in range(np.shape(self.patrones)[0]):
                res = ExecuteDMNN(self.patrones[p, :-1], self.pesW, self.numK, False)
                self.matrix[int(self.patrones[p, -1]), res] += 1
        except:
            pass

# funcion para generar parametros de compilacion en linea de comandos
# (no usada en el software), (sin dependencias)
def compilador(img):
    com = "pyinstaller -y -F -w -i \"icono.ico\""
    for i in range(img):
        com += " --add-data \"img" + str(i) + ".png\";\".\""
    com += " SoundRecognitionDSP.py"
    f = open("compilar.txt", "w")
    f.write(com)
    f.close()
    print(com)

# instanciar el software
main()

"""
Tareas:
- hacer la funcion que devuelva el pitch (tono) en lugar de cero
- boton para normalizar audio, o normalizarlo automaticamente...
- error, desfase en chart respecto a cero, clic para hallar rangos de corte
"""
