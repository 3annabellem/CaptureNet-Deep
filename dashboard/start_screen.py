import paramiko
from PyQt5.QtWidgets import (QDialog, QListWidget, QVBoxLayout, QPushButton, QMessageBox, QMainWindow,
                             QLabel, QLineEdit, QHBoxLayout, QWidget, QSpacerItem, QSizePolicy, QFileDialog, QApplication, QDesktopWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import sys
import os
from PyQt5.QtWidgets import QProgressBar
try:
    # Try relative import (when running as module)
    from .themes import choose_palette
    from .import_fast5 import ImportFast5
except ImportError:
    # Try direct import (when running script directly)
    from themes import choose_palette
    from import_fast5 import ImportFast5

class RemoteFileDialog(QDialog):
    def __init__(self, ssh_client, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select a .fast5 file")
        self.setModal(True)

        self.layout = QVBoxLayout()

        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(self.navigate_directory)
        self.layout.addWidget(self.file_list)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.select_file)
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)

        self.ssh_client = ssh_client
        self.current_path = "/disk2/pore_data"
        self.load_remote_files()

    def load_remote_files(self):
        try:
            sftp = self.ssh_client.open_sftp()
            files = sftp.listdir_attr(self.current_path)
            self.file_list.clear()
            
            # Add parent directory option
            if self.current_path != "/disk2/pore_data":
                self.file_list.addItem("..")
            
            for file in files:
                if file.st_mode & 0o40000:  # Check if it's a directory
                    self.file_list.addItem(f"[DIR] {file.filename}")
                elif file.filename.endswith(".fast5"):
                    self.file_list.addItem(file.filename)
            sftp.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to retrieve remote files:\n{str(e)}")
            self.reject()

    def navigate_directory(self, item):
        selected_item = item.text()
        try:
            sftp = self.ssh_client.open_sftp()
            
            if selected_item == "..":
                if self.current_path != "/":
                    self.current_path = os.path.dirname(self.current_path)  # Go up one directory
            elif selected_item.startswith("[DIR] "):
                dir_name = selected_item[6:]  # Remove "[DIR] " prefix
                new_path = f"{self.current_path}/{dir_name}"  # Construct new remote path
                
                # Verify if the new directory exists before navigating
                try:
                    sftp.listdir(new_path)  # Try listing the directory to confirm it's valid
                    self.current_path = new_path
                except FileNotFoundError:
                    QMessageBox.critical(self, "Error", f"Cannot access directory: {new_path}")
                    return
            
            sftp.close()
            self.load_remote_files()  # Refresh file list
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to retrieve remote files:\n{str(e)}")


    def select_file(self):
        selected_item = self.file_list.currentItem()
        if selected_item and not selected_item.text().startswith("[DIR] ") and selected_item.text() != "..":
            self.selected_file = os.path.join(self.current_path, selected_item.text())
            self.accept()
        else:
            QMessageBox.warning(self, "Invalid Selection", "Please select a valid .fast5 file.")

class StartupScreen(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.palette = choose_palette()
        self.initUI()


    def initUI(self):
        p = self.palette
        self.setWindowTitle('Protein Sequencing Dashboard')
        self.setGeometry(100, 100, 280, 170)
        self.center_window()
        self.showMaximized()

        # Apply background
        self.setStyleSheet(f"QMainWindow {{ background-color: {p['bg']}; }}")

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


        layout = QVBoxLayout()
        layout.addSpacerItem(QSpacerItem(20, 80, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Buttons
        for text, handler in [
            ('Import local fast5', self.open_import_fast5),
            ('Import remote fast5', self.open_remote_fast5),
        ]:
            btn = QPushButton(text)
            btn.setStyleSheet(btn_css)
            btn.clicked.connect(handler)
            layout.addWidget(btn, alignment=Qt.AlignCenter)

        layout.addStretch(1)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def center_window(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())



    def open_import_fast5(self):
        self._setup_path_screen(
            title_text="Enter the path to the fast5 file:",
            browse_callback=self.import_browse_file,
            go_callback=self.launch_import_fast_5
        )

    def launch_import_fast_5(self):
        self.close()
        fast5_path = self.textbox.text()
        self.dashboard = ImportFast5(fast5_path)
        self.dashboard.show()         

    def open_remote_fast5(self):
        p = self.palette
        btn_css = (
            f"QPushButton {{ background-color: {p['button_bg']}; "
            f"color: {p['button_fg']}; font: bold 14px; "
            f"border-radius: 10px; padding: 6px; }} "
            f"QPushButton:hover {{ background-color: #a0a0a0; }}"
        )

        layout = QVBoxLayout()
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Username
        self.username_label = QLabel("Enter your MISL-A username:")
        self.username_label.setStyleSheet(f"color: {p['fg']}")
        self.username_textbox = QLineEdit()
        self.username_textbox.setStyleSheet(f"background-color: {p['entry_bg']}; color: {p['entry_fg']}")

        # Password
        self.password_label = QLabel("Enter your password:")
        self.password_label.setStyleSheet(f"color: {p['fg']}")
        self.password_textbox = QLineEdit()
        self.password_textbox.setEchoMode(QLineEdit.Password)
        self.password_textbox.setStyleSheet(f"background-color: {p['entry_bg']}; color: {p['entry_fg']}")

        # Path entry
        self.path_label = QLabel("Enter the path to the fast5 file:")
        self.path_label.setStyleSheet(f"color: {p['fg']}")
        self.path_textbox = QLineEdit()
        self.path_textbox.setMinimumWidth(400)
        self.path_textbox.setStyleSheet(f"background-color: {p['entry_bg']}; color: {p['entry_fg']}")
        self.path_textbox.returnPressed.connect(self.launch_remote_fast_5)

        # Browse & Go buttons
        btn_browse = QPushButton('Browse...')
        btn_browse.setStyleSheet(btn_css)
        btn_browse.clicked.connect(self.remote_browse_file)

        btn_go = QPushButton('Go')
        btn_go.setStyleSheet(btn_css)
        btn_go.clicked.connect(self.launch_remote_fast_5)

        # Font setup
        font = QFont('Arial', 14)
        for w in [
            self.username_label, self.username_textbox,
            self.password_label, self.password_textbox,
            self.path_label, self.path_textbox,
            btn_browse, btn_go
        ]:
            w.setFont(font)

        # Layout assembly
        layout.addWidget(self.username_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.username_textbox, alignment=Qt.AlignCenter)
        layout.addWidget(self.password_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.password_textbox, alignment=Qt.AlignCenter)
        layout.addWidget(self.path_label, alignment=Qt.AlignCenter)
        hbox = QHBoxLayout()
        hbox.addWidget(self.path_textbox)
        hbox.addWidget(btn_browse)
        layout.addLayout(hbox)
        layout.addWidget(btn_go, alignment=Qt.AlignCenter)
        layout.addStretch(1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Save references for handlers
        self.textbox = self.path_textbox
        self.btn_go = btn_go

    def _setup_path_screen(self, title_text, browse_callback, go_callback):
        p = self.palette
        btn_css = (
            f"QPushButton {{ background-color: {p['button_bg']}; "
            f"color: {p['button_fg']}; font: bold 14px; "
            f"border-radius: 10px; padding: 6px; }} "
            f"QPushButton:hover {{ background-color: #a0a0a0; }}"
        )

        layout = QVBoxLayout()
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        label = QLabel(title_text)
        label.setStyleSheet(f"color: {p['fg']}")
        entry = QLineEdit()
        entry.setMinimumWidth(400)
        entry.setStyleSheet(f"background-color: {p['entry_bg']}; color: {p['entry_fg']}")
        entry.returnPressed.connect(go_callback)

        btn_browse = QPushButton('Browse...')
        btn_browse.setStyleSheet(btn_css)
        btn_browse.clicked.connect(browse_callback)

        btn_go = QPushButton('Go')
        btn_go.setStyleSheet(btn_css)
        btn_go.clicked.connect(go_callback)

        font = QFont('Arial', 14)
        for w in (label, entry, btn_browse, btn_go):
            w.setFont(font)

        hbox = QHBoxLayout()
        hbox.addWidget(entry)
        hbox.addWidget(btn_browse)

        layout.addWidget(label, alignment=Qt.AlignCenter)
        layout.addLayout(hbox)
        layout.addWidget(btn_go, alignment=Qt.AlignCenter)
        layout.addStretch(1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.textbox = entry
        self.btn_go = btn_go
    
    def launch_remote_fast_5(self):
        self.close()
        fast5_path = self.textbox.text()
        print(fast5_path)
        self.dashboard = ImportFast5(fast5_path)
        self.dashboard.show()    
        
    def import_browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select FAST5 File", "", "FAST5 Files (*.fast5)", options=options
        )
        if filename:
            self.textbox.setText(filename)

    def remote_browse_file(self):
        self.username = self.username_textbox.text().strip()
        self.password = self.password_textbox.text()

        if not self.username or not self.password:
            QMessageBox.warning(self, "Missing Credentials", "Please enter your username and password first.")
            return

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            print("Attempting to connect...")
            ssh.connect("misl-a.cs.washington.edu", username=self.username, password=self.password)
            print("Connected")
            dialog = RemoteFileDialog(ssh, self)
            print("Opened remote file dialog")
            result = dialog.exec_()  # Run the dialog in the existing event loop
            print("Got result")
            if result == QDialog.Accepted:
                print("Result accepted")
                selected_file = dialog.selected_file
                print(f"Got a selected file: {selected_file}")
                
                # Ensure remote file path uses Unix-style slashes
                selected_file = selected_file.replace("\\", "/")
                print(f"Normalized remote file path: {selected_file}")

                # Ensure the 'fast5' directory exists locally
                #if not os.path.exists("fast5"):
                #    os.makedirs("fast5")

                local_path = os.path.basename(selected_file)#os.path.join("fast5", os.path.basename(selected_file))
                print(f"Downloading {selected_file} to {local_path}...")

                # Create a loading dialog with a progress bar
                ######## FIX THIS ########
                loading_dialog = QDialog(self)
                loading_dialog.setWindowTitle("Downloading File")
                loading_dialog.setModal(True)
                loading_layout = QVBoxLayout()
                loading_label = QLabel("Downloading file...")
                loading_bar = QProgressBar()
                loading_bar.setRange(0, 0)  # Indeterminate progress
                loading_layout.addWidget(loading_label)
                loading_layout.addWidget(loading_bar)
                loading_dialog.setLayout(loading_layout)
                loading_dialog.show()

                # Perform the file download in a separate thread

                def download_file():
                    try:
                        sftp = ssh.open_sftp()
                        sftp.get(selected_file, local_path)  # Ensure full remote path is used
                        sftp.close()
                        ssh.close()
                        self.textbox.setText(local_path)
                        print("Download complete and path set")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to download file: {e}")
                    finally:
                        loading_dialog.accept()  # Close the loading dialog

                from threading import Thread
                download_thread = Thread(target=download_file)
                download_thread.start()
        except paramiko.AuthenticationException:
            QMessageBox.critical(self, "Authentication Failed", "Invalid username or password.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to download file: {e}")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = StartupScreen(app)
    ex.show()
    sys.exit(app.exec_())
