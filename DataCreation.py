import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QTextEdit, QProgressBar, QDialog
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, QObject
from PyQt5.QtGui import QTextCursor
from Preprocessing import process_stl_files
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import atexit
import os
import numpy as np

class Stream(QObject):
    new_output = pyqtSignal(str)

    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream

    def write(self, text):
        self.new_output.emit(str(text))
        self.original_stream.write(str(text))

    def flush(self):
        self.original_stream.flush()

class ProcessThread(QThread):
    def __init__(self, stl_folder, output_folder):
        super().__init__()
        self.stl_folder = stl_folder
        self.output_folder = output_folder

    def run(self):
        for _ in range(1, 101):
            time.sleep(0.1)
        process_stl_files(self.stl_folder, self.output_folder)

class STLApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selectedSTLFolder = None
        self.selectedOutputFolder = None
        self.classifier = None
        self.model = None
        self.initUI()
        self.initOutputStream()
        
    def initUI(self):
        self.setWindowTitle('STL Processing Application')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        centralWidget.setLayout(layout)

        # Button to Select STL Folder
        self.selectSTLFolderBtn = QPushButton('Select STL Folder', self)
        self.selectSTLFolderBtn.clicked.connect(self.selectSTLFolder)
        layout.addWidget(self.selectSTLFolderBtn)

        self.selectOutputFolderBtn = QPushButton('Select Output Folder', self)
        self.selectOutputFolderBtn.clicked.connect(self.selectOutputFolder)
        layout.addWidget(self.selectOutputFolderBtn)

        self.infoText = QTextEdit(self)
        self.infoText.setReadOnly(True)
        layout.addWidget(self.infoText)

        self.processBtn = QPushButton('Process STL Files', self)
        self.processBtn.clicked.connect(self.onProcessButtonClicked)
        layout.addWidget(self.processBtn)

        
    def initOutputStream(self):
        self.log_file = open("application.log", "a")
        atexit.register(self.log_file.close)

        sys.stdout = Stream(self.log_file)
        sys.stderr = Stream(self.log_file)


    def selectSTLFolder(self):
        folder_name = QFileDialog.getExistingDirectory(self, "Select STL Folder")
        if folder_name:
            self.infoText.append(f"Selected STL Folder: {folder_name}")
            self.selectedSTLFolder = folder_name

    def selectOutputFolder(self):
        folder_name = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_name:
            self.infoText.append(f"Selected Output Folder: {folder_name}")
            self.selectedOutputFolder = folder_name

    def onProcessButtonClicked(self):
        if self.selectedSTLFolder and self.selectedOutputFolder:
            self.infoText.append(f"Data Creation And Agumentation Progress is started.....(Dont Stop)")
            self.process_thread = ProcessThread(self.selectedSTLFolder, self.selectedOutputFolder)
            self.process_thread.start()
        else:
            self.infoText.append("Please select an STL folder and output folder first.")


if __name__ == '__main__':  
    app = QApplication(sys.argv)
    ex = STLApp()
    ex.show()
    sys.exit(app.exec_())
