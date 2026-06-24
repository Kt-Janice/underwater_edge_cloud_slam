import sys
import collections
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QPolygon, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class StateMachineVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(60)
        self.current_value = 80

    def update_value(self, value):
        self.current_value = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        bar_height = 30
        y_offset = 10

        # Define zone widths based on thresholds (0-10: Lost, 11-20: Warning, 21-100: Normal)
        lost_width = int(width * 0.3)
        warning_width = int(width * 0.3)
        normal_width = width - lost_width - warning_width

        # Draw Lost Zone
        painter.setBrush(QColor('#ffebee'))
        painter.setPen(QColor('#f44336'))
        painter.drawRect(0, y_offset, lost_width, bar_height)
        painter.setPen(QColor('#b71c1c'))
        painter.drawText(0, y_offset, lost_width, bar_height, Qt.AlignCenter, "Lost")

        # Draw Warning Zone
        painter.setBrush(QColor('#fff3e0'))
        painter.setPen(QColor('#ff9800'))
        painter.drawRect(lost_width, y_offset, warning_width, bar_height)
        painter.setPen(QColor('#e65100'))
        painter.drawText(lost_width, y_offset, warning_width, bar_height, Qt.AlignCenter, "Warning")

        # Draw Normal Zone
        painter.setBrush(QColor('#e8f5e9'))
        painter.setPen(QColor('#4caf50'))
        painter.drawRect(lost_width + warning_width, y_offset, normal_width, bar_height)
        painter.setPen(QColor('#1b5e20'))
        painter.drawText(lost_width + warning_width, y_offset, normal_width, bar_height, Qt.AlignCenter, "Normal Tracking")

        # Draw Pointer
        pointer_x = int((self.current_value / 100.0) * width)
        
        painter.setBrush(QColor('#1976d2'))
        painter.setPen(Qt.NoPen)
        poly = QPolygon([
            pointer_x, y_offset + bar_height,
            pointer_x - 8, y_offset + bar_height + 10,
            pointer_x + 8, y_offset + bar_height + 10
        ])
        painter.drawPolygon(poly)

class SimulatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge-Cloud Collaborative Tracking State Machine Simulator")
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet("background-color: #f5f6fa; font-family: Arial;")

        self.current_points = 80
        self.history_data = collections.deque([80]*100, maxlen=100)
        self.x_data = list(range(100))

        self.init_ui()
        self.update_state_logic()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_chart)
        self.timer.start(100)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # 1. Header Module
        header_layout = QHBoxLayout()
        
        title_layout = QVBoxLayout()
        title_label = QLabel("Edge-Cloud Collaborative Tracking State Machine Simulator")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50;")
        subtitle_label = QLabel("Adjust the slider to observe state machine failure handling and auto-reconnection logic.")
        subtitle_label.setStyleSheet("font-size: 13px; color: #7f8c8d;")
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(30)
        
        self.val_points = self.create_stat_vbox("Current Points", str(self.current_points), stats_layout)
        self.val_state = self.create_stat_vbox("System State", "NORMAL", stats_layout)
        self.val_cmd = self.create_stat_vbox("Active Command", "Pose Stabilization", stats_layout)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        header_layout.addLayout(stats_layout)
        
        main_layout.addLayout(header_layout)

        # 2. Communication Command Flow Module (Isolated to prevent occlusion)
        self.comm_frame = QFrame()
        self.comm_frame.setStyleSheet("background-color: white; border-radius: 8px; border: 1px solid #dcdde1;")
        comm_layout = QVBoxLayout(self.comm_frame)
        
        comm_title = QLabel("Edge Communication Command Flow")
        comm_title.setStyleSheet("font-size: 16px; font-weight: bold; border: none;")
        comm_layout.addWidget(comm_title)

        self.comm_line1 = QLabel("")
        self.comm_line2 = QLabel("")
        self.comm_line3 = QLabel("")
        for lbl in [self.comm_line1, self.comm_line2, self.comm_line3]:
            lbl.setStyleSheet("font-size: 14px; border: none; margin-top: 5px;")
            comm_layout.addWidget(lbl)

        main_layout.addWidget(self.comm_frame)

        # 3. State Machine Tracking Module
        self.state_visualizer = StateMachineVisualizer()
        main_layout.addWidget(self.state_visualizer)

        # 4. Real-Time Monitoring Module
        chart_frame = QFrame()
        chart_frame.setStyleSheet("background-color: white; border-radius: 8px; border: 1px solid #dcdde1;")
        chart_layout = QVBoxLayout(chart_frame)
        
        chart_title = QLabel("Real-time Points Monitoring")
        chart_title.setStyleSheet("font-size: 16px; font-weight: bold; border: none; padding: 10px 10px 0px 10px;")
        chart_layout.addWidget(chart_title)

        self.figure = Figure(figsize=(8, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.15)
        chart_layout.addWidget(self.canvas)
        
        main_layout.addWidget(chart_frame)

        # 5. Control Module
        control_layout = QHBoxLayout()
        
        ctrl_label = QLabel("Current Optimized Points:")
        ctrl_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.val_display = QLabel(str(self.current_points))
        self.val_display.setStyleSheet("font-size: 14px; border: 1px solid #bdc3c7; padding: 5px 15px; background: white; border-radius: 4px;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(self.current_points)
        self.slider.valueChanged.connect(self.on_slider_change)
        
        reset_btn = QPushButton("Reset Simulation")
        reset_btn.setStyleSheet("padding: 8px 20px; font-size: 14px; background-color: #ecf0f1; border: 1px solid #bdc3c7; border-radius: 4px;")
        reset_btn.clicked.connect(self.reset_simulation)

        control_layout.addWidget(ctrl_label)
        control_layout.addWidget(self.val_display)
        control_layout.addWidget(self.slider)
        control_layout.addWidget(reset_btn)

        main_layout.addLayout(control_layout)

    def create_stat_vbox(self, title, init_val, parent_layout):
        vbox = QVBoxLayout()
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-size: 12px; color: #7f8c8d; font-weight: bold;")
        lbl_title.setAlignment(Qt.AlignCenter)
        
        lbl_val = QLabel(init_val)
        lbl_val.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        lbl_val.setAlignment(Qt.AlignCenter)
        
        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_val)
        parent_layout.addLayout(vbox)
        return lbl_val

    def on_slider_change(self, value):
        self.current_points = value
        self.val_display.setText(str(value))
        self.val_points.setText(str(value))
        self.state_visualizer.update_value(value)
        self.update_state_logic()

    def update_state_logic(self):
        if self.current_points > 60:
            self.val_state.setText("NORMAL")
            self.val_state.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")
            self.val_cmd.setText("Pose Stabilization Transmission")
            self.comm_line1.setText("▶ Normal Publish: Pose Topic")
            self.comm_line1.setStyleSheet("color: #27ae60; font-weight: bold; border: none;")
            self.comm_line2.setText("▶ Heartbeat: Edge Node Healthy")
            self.comm_line2.setStyleSheet("color: #27ae60; font-weight: bold; border: none;")
            self.comm_line3.setText("")
        elif self.current_points > 30:
            self.val_state.setText("WARNING")
            self.val_state.setStyleSheet("font-size: 16px; font-weight: bold; color: #f39c12;")
            self.val_cmd.setText("Collaborative Intervention")
            self.comm_line1.setText("▶ Enable: Local Image Buffer")
            self.comm_line1.setStyleSheet("color: #f39c12; font-weight: bold; border: none;")
            self.comm_line2.setText("▶ Intercept: Invalid Pose Publish")
            self.comm_line2.setStyleSheet("color: #c0392b; font-weight: bold; border: none;")
            self.comm_line3.setText("Command: Prepare Relocalization Signal")
            self.comm_line3.setStyleSheet("color: #7f8c8d; border: none;")
        else:
            self.val_state.setText("LOST")
            self.val_state.setStyleSheet("font-size: 16px; font-weight: bold; color: #c0392b;")
            self.val_cmd.setText("Collaborative Intervention")
            self.comm_line1.setText("▶ Send: Action Goal Signal")
            self.comm_line1.setStyleSheet("color: #c0392b; font-weight: bold; border: none;")
            self.comm_line2.setText("▶ Start: Topic Image Stream")
            self.comm_line2.setStyleSheet("color: #2980b9; font-weight: bold; border: none;")
            self.comm_line3.setText("▶ Reset: Frontend State Machine")
            self.comm_line3.setStyleSheet("color: #c0392b; font-weight: bold; border: none;")

    def update_chart(self):
        self.history_data.append(self.current_points)
        self.ax.clear()
        
        y_data = list(self.history_data)
        self.ax.plot(self.x_data, y_data, color='#00acc1', linewidth=2)
        self.ax.fill_between(self.x_data, y_data, color='#00acc1', alpha=0.2)
        
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 100)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_xticks([])
        
        self.canvas.draw()

    def reset_simulation(self):
        self.slider.setValue(80)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimulatorWindow()
    window.show()
    sys.exit(app.exec_())
