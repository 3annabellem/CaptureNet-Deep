import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QApplication, QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QDesktopWidget, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QDateTime, QTime
from PyQt5.QtGui import QFont, QPixmap
import os
import time
import threading
import random
import os
import h5py
import gc
import torch
import torch.nn as nn
import json
try:
    # Try relative import (when running as module)
    from .themes import choose_palette
except ImportError:
    # Try direct import (when running script directly)
    from themes import choose_palette
from PyQt5.QtWidgets import QProgressBar
from scipy.ndimage import label
import torch.nn.functional as F

class ClickablePlotWidget(pg.PlotWidget):
    clicked = pyqtSignal(object)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit(self)

import torch.nn.functional as F

class CaptureNetDeep(nn.Module):
    def __init__(self, dropout=0.3):
        super(CaptureNetDeep, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(2)
        x = self.fc(x) 
        x = self.sigmoid(x).squeeze(1)
        return x

class Config_100:
    CHUNK_SIZE = 2**11  # later -- 65,536 points
    
    # Training params
    BATCH_SIZE = 32 #128*2*2
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    DOWNSAMPLE = 100
    
    DF_PATH = 'all_'
    
    CLIP_MIN = -1
    CLIP_MAX = 160
    DATA_MEAN = 17.17 #np.mean(df.raw.apply(np.mean))
    DATA_STD = 24.02 #np.mean(df.raw.apply(np.std))

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class TranslocationUNet1D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(TranslocationUNet1D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ImportFast5(QMainWindow):
    def __init__(self, file_name=None):
        # Ensure fast5_data directory exists
        if not os.path.exists('fast5_data'):
            os.makedirs('fast5_data')
        super().__init__()
        self.palette = choose_palette()
        p = self.palette

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.setStyleSheet(f"QMainWindow {{ background-color: {p['bg']}; }}")
        self.main_widget.setStyleSheet(f"background-color: {p['bg']};")
        self.file_name = file_name

        self.main_layout = QHBoxLayout(self.main_widget)
        self.grid_layout = QGridLayout()
        self.main_layout.addLayout(self.grid_layout)
        self.side_layout = QVBoxLayout()
        self.main_layout.addLayout(self.side_layout)

        self.y_min = -20
        self.y_max = 200

        # Button style
        btn_css = f"""
        QPushButton {{
            background-color: {p['button_bg']};
            color:            {p['button_fg']};
            font:             bold 14px;
            border-radius:    10px;
            padding:          6px;
        }}
        QPushButton:hover {{
            background-color: #a0a0a0;
        }}
        """

        data_path = "fast5_data"
        files = [file for file in os.listdir(data_path) if file.endswith(f"_{self.file_name.split('/')[-1]}.bin")]
        self.num_bin_files = len(files) 

        # Widgets for changing Y axis range
        self.y_min_input = QLineEdit(self)
        self.y_min_input.setStyleSheet(f"background-color: {p['entry_bg']}; color: {p['entry_fg']};")
        self.y_max_input = QLineEdit(self)
        self.y_max_input.setStyleSheet(f"background-color: {p['entry_bg']}; color: {p['entry_fg']};")
        self.update_button = QPushButton("Update Y Range", self)
        self.update_button.setStyleSheet(btn_css)
        self.update_button.clicked.connect(self.update_y_range)
        self.update_button.setEnabled(False)  # Optionally make button initially disabled

        self.large_plot_res_input = QLineEdit(self)
        self.large_plot_res_input.setStyleSheet(f"background-color: {p['entry_bg']}; color: {p['entry_fg']};")
        self.large_plot_update_bttn = QPushButton("Update downsampling factor", self)
        self.large_plot_update_bttn.setStyleSheet(btn_css)
        self.large_plot_update_bttn.clicked.connect(self.update_large_plot_resolution)

        self.export_button = QPushButton("Export data", self)
        self.export_button.setStyleSheet(btn_css)
        self.export_button.clicked.connect(self.export_data)

        # Adding widgets to the side layout
        min_y_lbl = QLabel("Min Y:")
        min_y_lbl.setStyleSheet(f"color: {p['fg']};")
        max_y_lbl = QLabel("Max Y:")
        max_y_lbl.setStyleSheet(f"color: {p['fg']};")
        
        large_plot_res_label = QLabel("Downsampling factor of \nlarge plot (default: 12,000):")
        large_plot_res_label.setStyleSheet(f"color: {p['fg']};")

        self.side_layout.addWidget(min_y_lbl, 0, Qt.AlignTop)
        self.side_layout.addWidget(self.y_min_input, 0, Qt.AlignTop)
        self.side_layout.addWidget(max_y_lbl, 0, Qt.AlignTop)
        self.side_layout.addWidget(self.y_max_input, 0, Qt.AlignTop)
        self.side_layout.addWidget(self.update_button, 0, Qt.AlignTop)
        self.side_layout.addWidget(self.export_button, 0, Qt.AlignTop)
        self.side_layout.addWidget(large_plot_res_label, 0, Qt.AlignTop)
        self.side_layout.addWidget(self.large_plot_res_input, 0, Qt.AlignTop)
        self.side_layout.addWidget(self.large_plot_update_bttn, 0, Qt.AlignTop)

        self.loading_label = QLabel("Loading files")
        self.loading_label.setStyleSheet(f"color: {p['fg']};")
        self.side_layout.addWidget(self.loading_label, 0, Qt.AlignTop)

        self.loaded_channels = 0
        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue((int)(self.num_bin_files / 512 * 100))
        self.side_layout.addWidget(self.progressBar, 0, Qt.AlignTop)
        self.side_layout.addStretch()

        # Connect text changes to validation method
        self.y_min_input.textChanged.connect(self.validate_input)
        self.y_max_input.textChanged.connect(self.validate_input)

        self.channel_length = [0] * 512  
        self.plots = []
        self.plot_items = []
        self.colors = [p['bg']] * 512

        self.dead_channels_count = 0
        self.capture_channels_count = 0
        self.translocation_channels_count = 0
        self.channel_status = ['unknown'] * 512  # Keeps track of the status of each channel
        self.captures = [[] for _ in range(512)]
        self.translocations = [[] for _ in range(512)]

        self.x_values = list(range(1, 60))
        self.data_points = [[] for _ in range(512)]

        self.grid_layout.setHorizontalSpacing(2)
        self.grid_layout.setVerticalSpacing(2)

        self.timerLabel = QLabel(self)
        self.timerLabel.setStyleSheet(f"font-size: 20px; color: {p['fg']};")
        self.side_layout.addWidget(self.timerLabel, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.side_layout.addStretch()

        self.screen_size = QDesktopWidget().screenGeometry(-1)
        self.setGeometry(100, 100, self.screen_size.width(), self.screen_size.height())
        self.showMaximized()

        # Create a new top layout to hold status labels and the large plot
        top_layout = QVBoxLayout()

        font = QFont()
        font.setPointSize(15)

        # Setup for dead channels
        dead_layout = QHBoxLayout()
        dead_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        dead_image_label = QLabel()
        self.dead_text_label = QLabel("Dead channels: 0")
        self.dead_text_label.setStyleSheet("color: black;")
        self.dead_text_label.setFont(font)
        images_dir = os.path.join(os.path.dirname(__file__), 'images')
        if os.path.exists(images_dir):
            dead_image_label.setPixmap(QPixmap(os.path.join(images_dir, "Red.png")).scaled(50, 50))
        else: 
            self.dead_text_label.setStyleSheet(f"color: {p['red']};")
        dead_layout.addWidget(dead_image_label)
        dead_layout.addWidget(self.dead_text_label)

        # Setup for capture channels
        capture_layout = QHBoxLayout()
        capture_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        capture_image_label = QLabel()
        self.capture_text_label = QLabel("Channels with capture sections: 0")
        self.capture_text_label.setStyleSheet("color: black;")
        self.capture_text_label.setFont(font)
        if os.path.exists(images_dir):
            capture_image_label.setPixmap(QPixmap(os.path.join(images_dir, "Blue.png")).scaled(50, 50))
        else: 
            self.capture_text_label.setStyleSheet(f"color: {p['blue']};")
        capture_layout.addWidget(capture_image_label)
        capture_layout.addWidget(self.capture_text_label)


        self.capture_model = CaptureNetDeep()

        # Load the trained model weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use the integrated CaptureNet-Deep model
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best-model.ckpt')
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        filtered_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.replace('model.', '') in self.capture_model.state_dict()}
        self.capture_model.load_state_dict(filtered_state_dict)

        # Set model to evaluation mode
        self.capture_model.eval()

        # TODO: Add translocation model when available
        self.translocation_model = None  # TranslocationUNet1D()
        # translocation_model_path = 'unet1d_model_100ds_2048CS_9epoch.pth'

        # print("Loading translocation weights from:", translocation_model_path)
        # ckpt = torch.load(translocation_model_path, map_location="cpu")

        # show all the running_mean shapes in *this* file
        # for name, tensor in ckpt.items():
        #     if "running_mean" in name:
        #         print(f"{name}: {tuple(tensor.shape)}")

        # for name, mod in self.translocation_model.named_modules():
        #     if isinstance(mod, nn.BatchNorm1d):
        #         assert mod.num_features in (64,128,256,512)
        # print("All BN layers match the 64–512 channel ranges.")


        # self.translocation_model.load_state_dict(torch.load(translocation_model_path, map_location=self.device))

        # ckpt = torch.load(translocation_model_path, map_location="cpu")
        # for key, val in ckpt.items():
        #     if "running_mean" in key:
        #         print(key, val.shape)

        # for name, mod in self.translocation_model.named_modules():
        #     if isinstance(mod, nn.BatchNorm1d):
        #         print(name, mod.num_features)

        # Setup for translocation channels
        translocation_layout = QHBoxLayout()
        translocation_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        translocation_image_label = QLabel()
        # translocation_image_label.setPixmap(QPixmap(os.path.join(images_dir, "Green.png")).scaled(50, 50))
        self.translocation_text_label = QLabel("Channels with translocations: 0")
        self.translocation_text_label.setStyleSheet("color: black;")
        self.translocation_text_label.setFont(font)
        if os.path.exists(images_dir):
            translocation_image_label.setPixmap(QPixmap(os.path.join(images_dir, "Green.png")).scaled(50, 50))
        else: 
            self.translocation_text_label.setStyleSheet(f"color: {p['green']};")
        translocation_layout.addWidget(translocation_image_label)
        translocation_layout.addWidget(self.translocation_text_label)

        self.timerLabel = QLabel(self)
        self.timerLabel.setStyleSheet(f"font-size: 20px; color: {p['fg']};")
        self.timerLabel.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        top_layout.addWidget(self.timerLabel)

        # Add all to the top layout
        top_layout.addLayout(dead_layout)
        top_layout.addLayout(capture_layout)
        top_layout.addLayout(translocation_layout)

        # Setup the large plot widget
        self.large_plot_resolution = 500
        self.large_plot_widget = pg.PlotWidget()
        self.large_plot_widget.setFixedSize((int)(self.screen_size.width() / 3), (int)(self.screen_size.height() / 2))
        self.large_plot_widget.setYRange(self.y_min, self.y_max, padding=0)
        self.large_plot_widget.setXRange(0, self.large_plot_resolution, padding=0)
        self.large_plot_widget.getPlotItem().hideAxis('bottom')
        
        # Configure title display
        plot_item = self.large_plot_widget.getPlotItem()
        plot_item.titleLabel.setMaximumHeight(50)  # Ensure enough space for title
        plot_item.layout.setContentsMargins(10, 10, 10, 10)  # Add margins
        
        self.large_plot_item = pg.PlotDataItem(pen=pg.mkPen(color='k', width=2))
        self.large_plot_widget.addItem(self.large_plot_item)
        self.large_plot_points = [[] for _ in range(512)]
        self.large_plot_widget.setBackground(p['bg'])

        # Add the large plot widget to the top layout
        top_layout.addWidget(self.large_plot_widget)

        # Use top_layout as the main widget's layout
        self.main_layout.addLayout(top_layout)

        self.setWindowTitle('Protein Sequencing Dashboard')
        self.selected_channel = None
        self.clicked_plot = None

        self.large_plot_timer = QTimer(self)
        self.large_plot_timer.timeout.connect(self.update_large_plot)
        self.large_plot_timer.start(10)

        self.startTime = QDateTime.currentDateTime()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateTime)
        self.timer.start(1000)

        self.background_counts_timer = QTimer(self)
        self.background_counts_timer.timeout.connect(self.update_background_counts)
        self.background_counts_timer.start(10)

        for channel in range(512):
            plot_widget = ClickablePlotWidget(self)
            plot_widget.clicked.connect(self.plot_clicked)
            plot_widget.getPlotItem().hideAxis('bottom')
            plot_widget.getPlotItem().hideAxis('left')
            plot_widget.setYRange(self.y_min, self.y_max, padding=0)
            plot_widget.setXRange(-1, 60, padding=0)
            plot_widget.setFixedSize((int)(self.screen_size.width() / 32), (int)(self.screen_size.height() / 40))
            self.grid_layout.addWidget(plot_widget, channel // 16, channel % 16)
            plot_item = pg.PlotDataItem(pen=pg.mkPen(color='k', width=2))
            plot_widget.addItem(plot_item)
            self.plots.append(plot_widget)
            self.plot_items.append(plot_item)
            plot_widget.setBackground(p['bg'])

        self.num_channels_loaded = 0

        if not os.path.exists("fast5_data"):
            os.makedirs("fast5_data")

        print("Initialization complete. Timer set.")

        for thread in range(8):
            thread = threading.Thread(target=self.load_data, args=(self.file_name, thread)).start()

    def load_nanopore_data(self, fn, channels_range):
        sampling_rate = None
        channel_data = {}
        if type(channels_range) != dict:
            channels_range = {chan:(0, None) for chan in channels_range}
        for number, (start, end) in channels_range.items():
            number = int(number) + 1
            if number == 0:
                print("DANGER!!! WARNING!! Ranges should start at 1, not 0, because there is no 0 channel")
                print("Make sure tha the the last channel you want is included (e.g., if you want channels 1-512, but passed range(512), this will only return channels 1-511)")
                continue
            if number % 20 == 0:
                print(f"finished channels: {number-19} to {str(number)}")
            channel = '/Raw/Channel_' + str(number) + '/'
            signal = '/Raw/Channel_' + str(number) + '/Signal'
            file_load = h5py.File(fn, mode="r").get(channel)
            file_meta = file_load["Meta"]
            offset = file_meta.attrs["offset"]
            digitisation = file_meta.attrs["digitisation"]
            _range = file_meta.attrs["range"]
            if sampling_rate and file_meta.attrs["sample_rate"] != sampling_rate:
                print("DANGER: sampling rate is NOT THE SAME FOR ALL RUNS -- must adjust segmentation algo for this")
                print("Changed from ", sampling_rate, 'to', file_meta.attrs["sample_rate"])
            sampling_rate = file_meta.attrs["sample_rate"]

            signal = np.array(file_load["Signal"][start:end])
            
            # convert to pA
            current = (signal + offset) * (_range / digitisation)
            float16_array = current.astype(np.float16)
        
            # Write the array to a binary file
            with open(f"fast5_data/{number}_{fn.split('/')[-1]}.bin", 'wb') as f:
                float16_array.tofile(f)
            self.num_bin_files += 1
            #channel_data[number] = current

            del signal
            del current
            gc.collect()
            
        del fn
        del sampling_rate
        del channels_range
        del number
        del channel
        del offset
        del digitisation
        del _range
        del file_load
        del file_meta
        gc.collect()
        
        return None
 


    def export_data(self):
        # Create data structure for JSON
        p = self.palette
        export_data = {
            "data": {
                "dead_pores": 0,
                "channels_with_captures": 0,
                "channels_with_translocations": 0
            }
        }

        # Process each channel
        for channel in range(len(self.captures)):
            is_dead_pore = self.colors[channel] == p['red']
            captures = self.captures[channel]
            translocations = self.translocations[channel]

            capArr = []
            conArr = []

            for capture in captures:
                capArr.append([capture[0], capture[1]])
                conArr.append(round(capture[2], 4))

            # Add channel data
            export_data[f"{channel + 1}"] = {
                "captures": capArr,
                "capture confidences": conArr,
                "translocations": translocations,
                "dead_pore": is_dead_pore
            }

        # Update top-level counts
        export_data["data"]["dead_pores"] = self.dead_channels_count
        export_data["data"]["channels_with_captures"] = self.capture_channels_count
        export_data["data"]["channels_with_translocations"] = self.translocation_channels_count

        # Write to JSON file
        output_file_name = f"{self.file_name}_exported_data.json"
        with open(output_file_name, "w") as json_file:
            json.dump(export_data, json_file, indent=4)

        print(f"Data exported to {output_file_name}")

        # Ask the user if they want to delete the temporary bin data
        reply = QMessageBox.question(self, "Delete Temporary Files?", 
                                     "Do you want to delete the temporary bin data?", 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.delete_bin_files()

        pop_up = QMessageBox.question(self, "Data exported", 
                                     f"Data exported to {output_file_name}", 
                                     QMessageBox.Ok, QMessageBox.Ok)

    def delete_bin_files(self):
        bin_directory = "fast5_data"
        if os.path.exists(bin_directory):
            for file in os.listdir(bin_directory):
                if file.endswith(f"{self.file_name.split('/')[5]}.bin"):
                    os.remove(os.path.join(bin_directory, file))
                    print(f"Deleted: {file}")
            print("Temporary bin files deleted.")
        else:
            print("No bin files found to delete.")

    def validate_input(self):
        # Enable button only if both inputs are valid floats
        try:
            float(self.y_min_input.text())
            float(self.y_max_input.text())
            self.update_button.setEnabled(True)
        except ValueError:
            self.update_button.setEnabled(False)

    def update_y_range(self):
        try:
            self.y_min = float(self.y_min_input.text())
            self.y_max = float(self.y_max_input.text())
            self.update_axes()
            self.y_min_input.clear()
            self.y_max_input.clear()
            self.update_button.setEnabled(False)
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid float values for Y range.', QMessageBox.Ok)

    def update_large_plot_resolution(self):
        try:
            self.large_plot_resolution = (int) (self.channel_length[0] / int(self.large_plot_res_input.text()))
            print(f"Large plot resolution set to: {self.large_plot_resolution}")
            self.large_plot_widget.setXRange(0, self.large_plot_resolution, padding=0)
            self.large_plot_res_input.clear()
            self.plot_clicked(self.clicked_plot)
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for downsampling factor.', QMessageBox.Ok)

    def update_axes(self):
        for plot in self.plots:
            plot.setYRange(self.y_min, self.y_max, padding=0)
        self.large_plot_widget.setYRange(self.y_min, self.y_max, padding=0)

    def updateTime(self):
        elapsed = self.startTime.secsTo(QDateTime.currentDateTime())
        elapsed_time = QTime(0, 0).addSecs(elapsed)
        self.timerLabel.setText(elapsed_time.toString('hh:mm:ss'))
        self.timerLabel.adjustSize()
        self.timerLabel.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.progressBar.setValue((int)(self.num_bin_files / 512 * 100))
        if self.num_bin_files >= 512:
            self.progressBar.setValue((int)(self.num_channels_loaded / 512 * 100))
            self.loading_label.setText("Running analysis")
        if self.num_channels_loaded >= 512:
            self.progressBar.setVisible(False)
            self.loading_label.setVisible(False)


    def plot_clicked(self, plot_widget=ClickablePlotWidget):
        self.clicked_plot = plot_widget
        channel = 0
        for plot in self.plots:
            if plot is plot_widget:
                self.selected_channel = channel
                break
            channel += 1

        if not os.path.exists(f"fast5_data/{channel + 1}_{self.file_name.split('/')[-1]}.bin"):
            return
        with open(f"fast5_data/{channel + 1}_{self.file_name.split('/')[-1]}.bin", 'rb') as f:
            data = np.fromfile(f, dtype=np.float16)
        self.large_plot_points[channel] = data[::(int)(self.channel_length[channel]/self.large_plot_resolution)]

        # Build confidence display string with labels - keep it concise
        title_parts = [f"Ch {channel + 1}"]
        
        # Add capture confidences (limit to first 2 for space)
        if self.captures[channel]:
            capture_confidences = [f"{capture[2]*100:.1f}%" for capture in self.captures[channel][:2]]
            if len(self.captures[channel]) > 2:
                capture_confidences.append(f"+{len(self.captures[channel])-2} more")
            capture_text = ", ".join(capture_confidences)
            title_parts.append(f"Cap: {capture_text}")
        
        # Add translocation confidences (limit to first 2 for space)
        if self.translocations[channel]:
            translocation_confidences = [f"{trans[2]*100:.1f}%" for trans in self.translocations[channel][:2]]
            if len(self.translocations[channel]) > 2:
                translocation_confidences.append(f"+{len(self.translocations[channel])-2} more")
            translocation_text = ", ".join(translocation_confidences)
            title_parts.append(f"Trans: {translocation_text}")
        
        # Create final title with proper formatting
        if len(title_parts) > 1:
            title = " | ".join(title_parts)
            print(f"Selected channel: {self.selected_channel + 1} - {' | '.join(title_parts[1:])}")
        else:
            title = title_parts[0]
            print(f"Selected channel: {self.selected_channel + 1}")
        
        # Set title with proper formatting options
        plot_item = self.large_plot_widget.getPlotItem()
        plot_item.setTitle(title)
        
        # Ensure title is visible and properly sized
        title_item = plot_item.titleLabel
        title_item.setMaximumHeight(40)  # Ensure enough space for title

        self.large_plot_widget

    def read_channel(self, file, channels):
        start = time.time()
        if (not os.path.exists(f"fast5_data/{list(channels)[-1]}_{self.file_name.split('/')[-1]}.bin")) and (file != None):
            self.load_nanopore_data(file, channels)
        for channel in channels:
            #print(f"Loading channel {channel}...")
            if (channel != 512) and (file != None):
                with open(f"fast5_data/{channel + 1}_{self.file_name.split('/')[-1]}.bin", 'rb') as f:
                    data = np.fromfile(f, dtype=np.float16)
                plot_widget = self.plots[channel]

                averages = []
                self.channel_length[channel] = len(data)

                # Background processing of data
                while len(data) > 100000:
                    chunk = data[:100000].astype(np.float32)
                    data = data[100000:]
                    sum_chunk = 0
                    for num in chunk:
                        sum_chunk += num
                    average = sum_chunk / 100000
                    averages.append(average)

                self.update_plot(plot_widget, averages, channel)
                self.update_background(channel, plot_widget, averages)
                del data
                self.num_channels_loaded += 1
        
        print(f"Data loaded in {time.time() - start} seconds for {channels}")

    def update_plot(self, plot_widget, data, channel):
        # Update the plot with new data
        for point in data: 
            self.data_points[channel].append(point)

        self.plot_items[channel].setData(self.data_points[channel])
        plot_widget.setXRange(0, len(data), padding=0)
        plot_widget.update()

    def update_background(self, channel, plot_widget, data):
        p = self.palette
        with open(f"fast5_data/{channel + 1}_{self.file_name.split('/')[-1]}.bin", 'rb') as f:
            long_data = np.fromfile(f, dtype=np.float16)
        if all(x < 5 for x in data[10:]):
            # too little signal → dead pore
            self.colors[channel] = p['red']
        else:
            capture, capture_sections         = self.capture(long_data)
            transloc, transloc_sections       = self.translocation(long_data)

            # record both
            self.captures[channel]     = capture_sections
            self.translocations[channel] = transloc_sections

            # Count both capture and translocation separately
            if capture:
                self.capture_channels_count += 1
            if transloc:
                self.translocation_channels_count += 1

            # choose a display color
            if transloc and capture:
                # Use a mixed color or alternate - let's use green but we'll show both on graph
                self.colors[channel] = p['green']
            elif transloc:
                self.colors[channel] = p['green']
            elif capture:
                self.colors[channel] = p['blue']
            else:
                self.colors[channel] = p['bg']

        if (self.colors[channel] == p['bg']):
            plot_widget.setBackground(p['bg'])
        elif (self.colors[channel] == p['red']):
            plot_widget.setBackground(p['red'])
            self.dead_channels_count += 1
        elif (self.colors[channel] == p['blue']):
            plot_widget.setBackground(p['blue'])
        elif (self.colors[channel] == p['green']):
            plot_widget.setBackground(p['green'])
            
    def update_background_counts(self):
        self.dead_text_label.setText(f"Dead channels: {self.dead_channels_count}")
        self.capture_text_label.setText(f"Channels with capture sections: {self.capture_channels_count}")
        self.translocation_text_label.setText(f"Channels with translocations: {self.translocation_channels_count}")

    
    def predict_signal(self, model, signal_data, config):
        """
        Run inference on a single full-length signal using your chunked classifier model.
        Feeds the model a 2D tensor [batch, length], letting forward() add the channel dim.
        """
        model.eval()

        # 1) Clip & normalize
        sig = np.clip(signal_data, config.CLIP_MIN, config.CLIP_MAX)
        sig = (sig - config.DATA_MEAN) / config.DATA_STD
        L, C = len(sig), config.CHUNK_SIZE

        full_mask  = np.zeros(L, dtype=np.float32)
        count_mask = np.zeros(L, dtype=np.float32)
        step = C // 2

        with torch.no_grad():
            for start in range(0, L, step):
                end = min(start + C, L)
                chunk = sig[start:end].astype(np.float32)

                # If it's shorter than CHUNK_SIZE, pad to the right
                if chunk.shape[0] < C:
                    pad = C - chunk.shape[0]
                    chunk = np.pad(chunk, (0, pad), 'constant')

                # Convert to tensor of shape [length]
                t = torch.from_numpy(chunk).to(config.DEVICE)  # -> [C]

                # **** HERE: make it [1, C], NOT [1,1,C] ****
                t = t.unsqueeze(0).unsqueeze(0)                            # -> [1, C]

                # Now forward: model will do x.unsqueeze(1) -> [1,1,C]
                out = model(t)

                # --- segmentation: get per-sample probabilities ---
                probs = torch.sigmoid(out)        \
                               .squeeze()        \
                               .cpu()            \
                               .numpy()          # shape = (chunk_len,)
            
                # trim any padding, then accumulate
                probs = probs[: end - start]
                full_mask[start:end]  += probs
                count_mask[start:end] += 1

        # Average overlapping windows
        count_mask[count_mask == 0] = 1
        full_mask /= count_mask
        print(f"full_mask.shape: {full_mask.shape}, full_mask head: {full_mask[:10]}")
        return full_mask


    def mask_to_events(self, mask, threshold=0.5):
        """
        Convert a probability mask to a list of (start, end) event tuples.
        """
        # Apply threshold to get a binary mask
        binary_mask = (mask > threshold).astype(int)
        
        # Find contiguous regions of 1s
        labeled_mask, num_features = label(binary_mask)
        
        if num_features == 0:
            return []

        events = []
        for i in range(1, num_features + 1):
            indices = np.where(labeled_mask == i)[0]
            start_idx = indices.min()
            end_idx = indices.max() + 1 # end is exclusive
            events.append((start_idx, end_idx))
            
        return events

    def translocation(self, data):
        """
        Detect translocations in a full data sequence.

        Args:
            data (np.ndarray): 1D array of raw signal samples.

        Returns:
            has_translocation (bool): True if ≥1 translocation found.
            trans_sections (List[List[int, int, float]]):
                Each entry is [orig_start_idx, orig_end_idx, avg_confidence].
        """
        # 1. Downsample to match the model’s training resolution
        Config = Config_100
        downsample = Config.DOWNSAMPLE  # e.g. 100 :contentReference[oaicite:0]{index=0}
        signal = data[::downsample]

        # 2. Run the segmentation‐style inference to get per‐point probabilities
        prob_mask = self.predict_signal(self.translocation_model, signal, Config)  # :contentReference[oaicite:1]{index=1}

        # 3. Threshold the probability mask into contiguous events
        #    using a higher threshold to match capture detection sensitivity
        events = self.mask_to_events(prob_mask, threshold=0.8)         # Increased from 0.5 to 0.8
        min_dur = 10  # Minimum duration in downsampled samples (equivalent to 1000 original samples)
        events = [(s,e) for s,e in events if (e-s) >= min_dur]

        # 4. Build the output list of [start, end, avg_confidence]
        # Calculate the 5% threshold in original signal samples
        signal_length = len(data)
        threshold_5_percent = signal_length * 0.1
        
        trans_sections = []
        for start, end in events:
            # Scale indices back to original sampling
            orig_start = start * downsample
            orig_end = end * downsample
            
            # Discard translocations that start in the first 5% of the signal
            if orig_start < threshold_5_percent:
                print(f"Discarding translocation at {orig_start}-{orig_end} (within first 5% of signal)")
                continue
                
            # Compute mean probability over this region
            avg_conf = float(prob_mask[start:end].mean()) if end > start else 0.0
            trans_sections.append([orig_start, orig_end, avg_conf])

        # 5. Return a flag plus the list
        has_translocation = len(trans_sections) > 0
        return has_translocation, trans_sections


    def capture(self, data):
        window_size = 1000
        data = data[::100]
        num_points = len(data)
        num_windows = num_points // window_size
        labels = np.zeros(num_points)  # Initialize labels array
        confidenceArr = []
        
        # Process each window
        for w in range(num_windows):
            start_idx = w * window_size
            end_idx = (w + 1) * window_size
            window = data[start_idx:end_idx].reshape(-1, 1).flatten()
            
            # Convert to tensor and predict
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            likelihood = self.capture_model(window_tensor).item()  # Get the float output from the model
            labels[start_idx:end_idx] = 1 if likelihood >= 0.8 else 0
            confidenceArr.append(likelihood if likelihood >= 0.8 else 0)

        # Adjust isolated labels
        for w in range(1, num_windows-1):
            start_idx = w * window_size
            end_idx = (w + 1) * window_size
            if labels[start_idx] != labels[start_idx - window_size] and labels[start_idx] != labels[end_idx]:
                labels[start_idx:end_idx] = labels[start_idx - window_size]
                confidenceArr[w] = confidenceArr[w-1]

        # Convert labels to start and end indices for contiguous regions
        confidenceArr.append(0)
        capture_sections = []
        current_label = labels[0]
        section_start = 0
        confidence = 0
        count = 0
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                if current_label == 1:
                    # Scale back to original sampling rate (multiply by 100)
                    capture_sections.append([section_start * 100, (i - 1) * 100, confidence/count])
                current_label = labels[i]
                section_start = i
                confidence = 0
                count = 0
            confidence += confidenceArr[i//window_size]
            count += 1
        # Add the last section if it's a capture section
        if current_label == 1:
            # Scale back to original sampling rate (multiply by 100)
            capture_sections.append([section_start * 100, (len(labels) - 1) * 100, confidence/count])

        capture = False
        if len(capture_sections) > 0:
            capture = True

        return capture, capture_sections

    def update_large_plot(self):
        p = self.palette
        if self.selected_channel is None:
            return
        self.large_plot_widget.setYRange(self.y_min, self.y_max, padding=0)

        channel = self.selected_channel
        self.large_plot_item.setData(self.large_plot_points[channel])
        self.large_plot_widget.setXRange(0, len(self.large_plot_points[channel]), padding=0)
        plot_widget = (self.plots[channel])
        plot_widget.setYRange(self.y_min, self.y_max, padding=0)

        for item in self.large_plot_widget.items():
            if isinstance(item, pg.LinearRegionItem):
                self.large_plot_widget.removeItem(item)

        # recompute down‑sample factor
        raw_len = self.channel_length[channel]
        plot_len = len(self.large_plot_points[channel])
        factor   = raw_len / plot_len
        print(f"Channel {channel + 1}: raw_len={raw_len}, plot_len={plot_len}, factor={factor}")
        print(f"large_plot_resolution={self.large_plot_resolution}")
        
        # Calculate the actual pixel width of the plot widget
        plot_widget_width = self.large_plot_widget.width()
        pixels_per_data_unit = plot_widget_width / plot_len if plot_len > 0 else 1
        min_width_data_units = 5 / pixels_per_data_unit  # Convert 5 pixels to data units
        print(f"Plot widget width: {plot_widget_width}px, pixels per data unit: {pixels_per_data_unit:.4f}")
        print(f"5 pixels = {min_width_data_units:.4f} data units")

        # Draw captures
        for (samp_start, samp_end, _) in self.captures[channel]:
            x0 = samp_start / factor
            x1 = samp_end   / factor
            original_width = x1 - x0
            # Ensure minimum width of 5 pixels
            if (x1 - x0) < min_width_data_units:
                center = (x0 + x1) / 2
                x0 = center - min_width_data_units / 2
                x1 = center + min_width_data_units / 2
            print(f"Capture: original_width={original_width:.2f}, final_width={x1-x0:.2f}, x0={x0:.2f}, x1={x1:.2f}")
            cap_region = pg.LinearRegionItem(values=(x0, x1),
                                             movable=False,
                                             brush=pg.mkBrush(p['blue'], alpha=100))  # Semi-transparent
            cap_region.setZValue(10)
            self.large_plot_widget.addItem(cap_region)

        # Draw translocations with overlap merging
        translocation_regions = []
        
        # First pass: convert to pixel coordinates (no minimum width yet)
        for (samp_start, samp_end, confidence) in self.translocations[channel]:
            x0 = samp_start / factor
            x1 = samp_end   / factor
            translocation_regions.append((x0, x1, confidence))
        
        # Second pass: merge overlapping regions
        if translocation_regions:
            # Sort by start position
            translocation_regions.sort(key=lambda x: x[0])
            merged_regions = [translocation_regions[0]]
            
            for current in translocation_regions[1:]:
                last = merged_regions[-1]
                # Check if current region overlaps with the last merged region
                if current[0] <= last[1]:  # Overlap detected
                    # Merge regions: extend end and average confidence
                    new_end = max(last[1], current[1])
                    avg_confidence = (last[2] + current[2]) / 2
                    merged_regions[-1] = (last[0], new_end, avg_confidence)
                else:
                    # No overlap, add as new region
                    merged_regions.append(current)
            
            # Third pass: apply minimum width and draw (same as captures)
            for x0, x1, confidence in merged_regions:
                original_width = x1 - x0
                # Apply minimum width of 5 pixels (same logic as captures)
                if (x1 - x0) < min_width_data_units:
                    center = (x0 + x1) / 2
                    x0 = center - min_width_data_units / 2
                    x1 = center + min_width_data_units / 2
                
                print(f"Translocation: original_width={original_width:.2f}, final_width={x1-x0:.2f}, x0={x0:.2f}, x1={x1:.2f}")
                trans_region = pg.LinearRegionItem(values=(x0, x1),
                                                   movable=False,
                                                   brush=pg.mkBrush(p['green'], alpha=100))  # Semi-transparent
                trans_region.setZValue(11)
                self.large_plot_widget.addItem(trans_region)

        self.large_plot_item.setZValue(20)

    def load_data(self, file, thread):
        time.sleep(15)
        print(f"Thread {thread} starting.")
        self.read_channel(file, range(thread * 64, thread * 64 + 64))
        print(f"Thread {thread} finished.")   

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImportFast5()
    ex.show()
    sys.exit(app.exec_())