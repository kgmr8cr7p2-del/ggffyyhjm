"""GUI и боевой цикл Combat WalkBot с полной настройкой параметров."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple
import math  # 🔥 Для округления
from queue import Queue  # 🔥 Для очереди
import os  # 🔥 Для проверки файлов
import json  # 🔥 Для сохранения/загрузки конфига
import random  # 🔥 Для рандомизации
import multiprocessing as mp  # 🔥 Для multiprocessing

import cv2
import numpy as np
import torch
from ultralytics import YOLO  # 🔥 Добавлено для YOLOv8
from PyQt6.QtCore import QObject, Qt, pyqtSignal, QRect, QTimer, QCoreApplication, QThread
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from logic import CTRL_KEY, W_KEY, InputController
from recorder import RouteRecorder
from walk import RouteNavigator, SpawnDetector

# 🔥 Импорт pynput для горячих клавиш
from pynput.keyboard import Key, Listener

# 🔥 Для оптимизации захвата экрана
import mss


@dataclass
class AimConfig:
    # Детекция
    conf_threshold: float = 0.35  # Минимальная уверенность YOLO для цели (0.1-0.95). Ниже - больше целей, выше - точнее
    target_class: str = "person"  # Класс цели: 'person' - люди, 'all' - все
    nearest_target: bool = True  # Выбирать ближайшую цель (True) или с max conf (False)

    # Прицеливание
    mouse_mode: str = "relative"  # Режим мыши: 'relative' - относительный, 'absolute' - абсолютный
    combat_fov: int = 380  # Размер FOV для детекции (180-900 px). Авто-округление до кратного 32
    aim_gain_x: float = 4.2  # Чувствительность X (0.5-5.0). Выше - быстрее
    aim_gain_y: float = 3.9  # Чувствительность Y (0.5-5.0). Выше - быстрее
    aim_max_step_px: float = 85.0  # Макс шаг движения (1-150 px). Выше - агрессивнее
    center_radius_px: int = 14  # Радиус центра для стрельбы (5-30 px). Больше - чаще стреляет
    aim_head_offset_percent: float = 0.25  # 🔥 Изменено: Процент от высоты бокса для хеда (0.0-1.0). 0.25 - 25% от верха
    mouse_multiplier: float = 3.5  # Множитель мыши для sens (1.0-5.0)
    pid_kp: float = 4.2  # PID KP (0.1-10.0). Пропорциональный: выше - быстрее snap
    pid_ki: float = 28.0  # PID KI (0-50). Интегральный: выше - лучше на движении
    pid_kd: float = 0.15  # PID KD (0-1.0). Дифференциальный: выше - меньше овершута
    prediction_frames: int = 2  # Предсказание кадров (0-5). Для движущихся целей
    deadzone_px: float = 1.2  # Мертвая зона (0-10 px)
    deadzone_hysteresis_px: float = 0.8  # Гистерезис для deadzone (0-10 px)
    max_speed_px_per_sec: float = 2600.0  # Лимит скорости наведения (px/sec)
    max_accel_px_per_sec2: float = 14000.0  # Лимит ускорения наведения (px/sec^2)
    use_bezier: bool = False  # Кривые Безье для human-like движения
    bezier_intensity: float = 10.0  # Интенсивность кривизны Безье (0-50 px). Выше - сильнее изгиб
    bezier_steps: int = 4  # Количество шагов Безье (2-10). Больше - плавнее

    # Стрельба
    auto_shoot: bool = False  # Авто-стрельба при наведении (по умолчанию выключена)
    shoot_cooldown_sec: float = 0.08  # Задержка shots (0.03-1.0 сек). Меньше - спам
    shoot_click_delay_sec: float = 0.010  # Длительность клика (0.005-0.200 сек)
    burst_shots: int = 3  # Выстрелов в burst (1-10)
    recoil_comp_px: float = 8.0  # Компенсация отдачи (0-20 px). Движение вниз

    # Навигация
    nav_mouse_gain: float = 0.55  # Чувствительность поворота (0.05-1.50). Выше - быстрее
    nav_pause_when_enemy: bool = True  # Пауза при враге

    # Режим проверки
    desktop_test_mode: bool = False  # Тест на столе (F8)
    desktop_test_autoclick: bool = False  # Стрельба в тесте
    desktop_force_absolute: bool = True  # Absolute в тесте

    # Общее
    cycle_sleep_sec: float = 0.005  # Задержка цикла (0.001-0.100 сек). Меньше - выше FPS
    model_type: str = "pytorch"  # Тип модели: 'pytorch' или 'tensorrt' (если CUDA)
    use_fp16: bool = True  # Использовать FP16 для TensorRT
    use_int8: bool = True  # 🔥 Новый: Использовать INT8 для TensorRT

    # 🔥 Новые параметры
    random_timing_variance: float = 0.02  # Диапазон рандомизации таймингов (± сек, 0.0-0.1)
    target_switch_delay_sec: float = 0.15  # Задержка перед захватом новой цели (0.0-0.5 сек)


class BotState(QObject):
    frame_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str)
    routes_signal = pyqtSignal(list)
    log_signal = pyqtSignal(str)
    fov_changed_signal = pyqtSignal(int)
    target_lock_signal = pyqtSignal(list)
    recorder_status_signal = pyqtSignal(str)  # Новый сигнал для статуса записи
    performance_signal = pyqtSignal(float, float, float, float, float, float)  # FPS, Latency, T_capture, T_infer, T_post, T_input

    def __init__(self) -> None:
        super().__init__()
        self.running = False
        self.manual_route: Optional[Path] = None
        self.config = AimConfig()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bezier_t(t: float, p0: float, p1: float, p2: float) -> float:
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


class CombatWalkBotWindow(QWidget):
    def __init__(self, state: BotState, routes_dir: Path):
        super().__init__()
        self.state = state
        self.routes_dir = routes_dir
        self.config_file = Path(__file__).resolve().parent / "config.json"
        self.init_ui()
        self.load_config()  # 🔥 Загрузка конфига при инициализации

    def init_ui(self) -> None:
        self.setWindowTitle("Combat WalkBot")
        self.setFixedSize(1220, 800)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        root = QHBoxLayout()
        left = QVBoxLayout()

        self.status_lbl = QLabel("Статус: Остановлен")
        self.status_lbl.setStyleSheet("font-weight: bold; font-size: 16px;")
        left.addWidget(self.status_lbl)

        self.route_list = QListWidget()
        self.route_list.setMinimumHeight(160)
        left.addWidget(self.route_list)

        btn_row1 = QHBoxLayout()
        self.btn_refresh = QPushButton("Обновить список")
        self.btn_delete = QPushButton("Удалить выбранный")
        btn_row1.addWidget(self.btn_refresh)
        btn_row1.addWidget(self.btn_delete)
        left.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.btn_load = QPushButton("Загрузить вручную")
        self.btn_start = QPushButton("Старт / Стоп (F9)")
        btn_row2.addWidget(self.btn_load)
        btn_row2.addWidget(self.btn_start)
        left.addLayout(btn_row2)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(self._make_settings_widget())
        left.addWidget(settings_scroll)

        self.info_lbl = QLabel(
            "Горячие клавиши:\n"
            "F9 — запуск/остановка бота\n"
            "F10 — старт/стоп записи маршрута\n"
            "F8 — включить/выключить режим проверки на рабочем столе\n"
            "Маршруты: /routes, Спавны: /spawns"
        )
        left.addWidget(self.info_lbl)

        right = QVBoxLayout()
        self.preview = QLabel()
        self.preview.setFixedSize(560, 480)
        self.preview.setStyleSheet("background: #111;")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.preview)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Здесь будут логи работы бота...")
        self.log_view.setMinimumHeight(240)
        right.addWidget(self.log_view)

        root.addLayout(left, 1)
        root.addLayout(right, 1)
        self.setLayout(root)

        self.btn_refresh.clicked.connect(self.refresh_routes)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_load.clicked.connect(self.load_selected)
        self.btn_start.clicked.connect(self.toggle_running)

        self.state.frame_signal.connect(self.update_preview)
        self.state.status_signal.connect(self.update_status)
        self.state.routes_signal.connect(self._set_routes)
        self.state.log_signal.connect(self.append_log)

        self.refresh_routes()

    def _make_settings_widget(self) -> QWidget:
        cfg = self.state.config
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Группа Детекция
        detection_box = QGroupBox("Детекция (Распознавание целей)")
        detection_form = QFormLayout()
        detection_form.setSpacing(5)

        label_conf = QLabel("Порог уверенности YOLO")
        label_conf.setToolTip("Это значение определяет, насколько модель YOLO должна быть уверена в обнаружении цели, чтобы считать её реальной. Для новичков: Если установить ниже (например, 0.3), бот будет видеть больше целей, но может реагировать на ложные объекты (стены, предметы). Если выше (0.6), он будет точнее, но может пропускать врагов. Рекомендация: Начните с 0.4 и уменьшайте, если бот 'слепой', или увеличивайте, если много ложных срабатываний.")
        self.sb_conf = QDoubleSpinBox()
        self.sb_conf.setRange(0.10, 0.95)
        self.sb_conf.setSingleStep(0.01)
        self.sb_conf.setDecimals(2)
        self.sb_conf.setValue(cfg.conf_threshold)
        self.sb_conf.valueChanged.connect(lambda v: self._update_config("conf_threshold", float(v)))
        self.sb_conf.setToolTip(label_conf.toolTip())
        detection_form.addRow(label_conf, self.sb_conf)

        label_target_class = QLabel("Класс цели")
        label_target_class.setToolTip("Выберите, что бот должен распознавать как цель. 'person' - только людей (врагов в игре), это стандарт для CS2. 'all' - все объекты (машины, животные и т.д.), но это может вызвать ложные срабатывания. Для новичков: Оставьте 'person', если игра про людей; 'all' полезно, если в игре есть другие цели, но тестируйте, чтобы избежать ошибок.")
        self.combo_target_class = QComboBox()
        self.combo_target_class.addItems(["person", "all"])
        self.combo_target_class.setCurrentText(cfg.target_class)
        self.combo_target_class.currentTextChanged.connect(lambda v: self._update_config("target_class", v))
        self.combo_target_class.setToolTip(label_target_class.toolTip())
        detection_form.addRow(label_target_class, self.combo_target_class)

        self.cb_nearest = QCheckBox("Выбирать ближайшую цель к центру")
        self.cb_nearest.setChecked(cfg.nearest_target)
        self.cb_nearest.toggled.connect(lambda v: self._update_config("nearest_target", bool(v)))
        self.cb_nearest.setToolTip("Если включено, бот будет прицеливаться на врага, который ближе к центру экрана (удобно в ближнем бою). Если выключено, выберет цель с наибольшей уверенностью (может быть дальше). Для новичков: Включите для динамичных матчей, выключите, если хотите фокус на 'главных' целях.")
        detection_form.addRow(self.cb_nearest)

        detection_box.setLayout(detection_form)
        layout.addWidget(detection_box)

        # Группа Прицеливание
        aim_box = QGroupBox("Прицеливание (Наведение на цель)")
        aim_form = QFormLayout()
        aim_form.setSpacing(5)

        label_mouse_mode = QLabel("Режим перемещения мышки")
        label_mouse_mode.setToolTip("Определяет, как бот двигает мышь. 'relative' - плавно, как в игре (стандарт для CS2). 'absolute' - напрямую на позицию (для тестов на столе). Для новичков: Используйте 'relative' в игре; 'absolute' только для отладки, иначе мышь может 'прыгать'.")
        self.combo_mouse_mode = QComboBox()
        self.combo_mouse_mode.addItems(["relative", "absolute"])
        self.combo_mouse_mode.setCurrentText(cfg.mouse_mode)
        self.combo_mouse_mode.currentTextChanged.connect(lambda v: self._update_config("mouse_mode", v))
        self.combo_mouse_mode.setToolTip(label_mouse_mode.toolTip())
        aim_form.addRow(label_mouse_mode, self.combo_mouse_mode)

        label_fov = QLabel("Боевой FOV")
        label_fov.setToolTip("Размер квадрата на экране, где бот ищет цели (в пикселях). Меньше (300-400) - быстрее работает, но меньше видит. Больше (500+) - шире обзор, но медленнее. Для новичков: Начните с 400; уменьшайте для скорости на слабом ПК, увеличивайте для большего поля зрения.")
        self.sb_fov = QSpinBox()
        self.sb_fov.setRange(180, 900)
        self.sb_fov.setValue(cfg.combat_fov)
        self.sb_fov.valueChanged.connect(self._update_fov)
        self.sb_fov.setToolTip(label_fov.toolTip())
        aim_form.addRow(label_fov, self.sb_fov)

        label_aim_x = QLabel("Чувствительность наведения X")
        label_aim_x.setToolTip("Скорость движения мыши по горизонтали. Выше (4+) - быстрее поворот, но может перелетать цель. Ниже (2-3) - плавнее, точнее. Для новичков: Установите 3.5-4 для агрессивного стиля; тестируйте в desktop_mode, чтобы не было 'дерганья'.")
        self.sb_aim_x = QDoubleSpinBox()
        self.sb_aim_x.setRange(0.5, 5.0)
        self.sb_aim_x.setSingleStep(0.1)
        self.sb_aim_x.setValue(cfg.aim_gain_x)
        self.sb_aim_x.valueChanged.connect(lambda v: self._update_config("aim_gain_x", float(v)))
        self.sb_aim_x.setToolTip(label_aim_x.toolTip())
        aim_form.addRow(label_aim_x, self.sb_aim_x)

        label_aim_y = QLabel("Чувствительность наведения Y")
        label_aim_y.setToolTip("Скорость движения мыши по вертикали. Аналогично X, но для вверх/вниз. Для новичков: Сделайте чуть меньше X (например, 3.5 если X=4), так как голова выше, и вертикаль критична для хедшотов.")
        self.sb_aim_y = QDoubleSpinBox()
        self.sb_aim_y.setRange(0.5, 5.0)
        self.sb_aim_y.setSingleStep(0.1)
        self.sb_aim_y.setValue(cfg.aim_gain_y)
        self.sb_aim_y.valueChanged.connect(lambda v: self._update_config("aim_gain_y", float(v)))
        self.sb_aim_y.setToolTip(label_aim_y.toolTip())
        aim_form.addRow(label_aim_y, self.sb_aim_y)

        label_aim_max_step = QLabel("Макс. шаг наведения (px)")
        label_aim_max_step.setToolTip("Максимум, на сколько пикселей мышь двигается за раз. Выше (80+) - быстро захватывает дальних врагов. Ниже (40) - точнее на близких. Для новичков: 60-80 для баланса; если бот 'промахивается', уменьшите.")
        self.sb_aim_max_step = QDoubleSpinBox()
        self.sb_aim_max_step.setRange(1.0, 150.0)
        self.sb_aim_max_step.setSingleStep(1.0)
        self.sb_aim_max_step.setValue(cfg.aim_max_step_px)
        self.sb_aim_max_step.valueChanged.connect(lambda v: self._update_config("aim_max_step_px", float(v)))
        self.sb_aim_max_step.setToolTip(label_aim_max_step.toolTip())
        aim_form.addRow(label_aim_max_step, self.sb_aim_max_step)

        label_center = QLabel("Радиус точного попадания (px)")
        label_center.setToolTip("Зона вокруг центра прицела, где цель считается 'навденной' для стрельбы. Больше (15+) - чаще стреляет, но менее точно. Меньше (8) - ждет идеала для хедшотов. Для новичков: 10-12; увеличьте, если бот не стреляет timely.")
        self.sb_center = QSpinBox()
        self.sb_center.setRange(5, 30)
        self.sb_center.setValue(cfg.center_radius_px)
        self.sb_center.valueChanged.connect(lambda v: self._update_config("center_radius_px", int(v)))
        self.sb_center.setToolTip(label_center.toolTip())
        aim_form.addRow(label_center, self.sb_center)

        label_head_offset = QLabel("Сдвиг для headshot (%)")  # 🔥 Изменено на процент
        label_head_offset.setToolTip("Процент от высоты бокса для сдвига точки прицела вверх (0.0-1.0). 0.25 - цель на 25% от верха бокса (голова).")
        self.sb_head_offset = QDoubleSpinBox()
        self.sb_head_offset.setRange(0.0, 1.0)
        self.sb_head_offset.setSingleStep(0.01)
        self.sb_head_offset.setValue(cfg.aim_head_offset_percent)
        self.sb_head_offset.valueChanged.connect(lambda v: self._update_config("aim_head_offset_percent", float(v)))
        self.sb_head_offset.setToolTip(label_head_offset.toolTip())
        aim_form.addRow(label_head_offset, self.sb_head_offset)

        label_mouse_mult = QLabel("Множитель мыши")
        label_mouse_mult.setToolTip("Умножает перемещение мыши для sensitivity в игре. Выше (3+) - агрессивнее. Для новичков: Соответствуйте вашей sens в CS2 (1.5-2.5 sens = 3-4 mult); тестируйте, чтобы не было лагов.")
        self.sb_mouse_mult = QDoubleSpinBox()
        self.sb_mouse_mult.setRange(1.0, 5.0)
        self.sb_mouse_mult.setSingleStep(0.1)
        self.sb_mouse_mult.setValue(cfg.mouse_multiplier)
        self.sb_mouse_mult.valueChanged.connect(lambda v: self._update_config("mouse_multiplier", float(v)))
        self.sb_mouse_mult.setToolTip(label_mouse_mult.toolTip())
        aim_form.addRow(label_mouse_mult, self.sb_mouse_mult)

        label_pid_kp = QLabel("PID KP")
        label_pid_kp.setToolTip("Пропорциональный коэффициент PID: реагирует на ошибку. Выше (4+) - быстрее snap. Для новичков: 3.5-4.5; если овершут, уменьшите KD сначала.")
        self.sb_pid_kp = QDoubleSpinBox()
        self.sb_pid_kp.setRange(0.1, 10.0)
        self.sb_pid_kp.setSingleStep(0.1)
        self.sb_pid_kp.setValue(cfg.pid_kp)
        self.sb_pid_kp.valueChanged.connect(lambda v: self._update_config("pid_kp", float(v)))
        self.sb_pid_kp.setToolTip(label_pid_kp.toolTip())
        aim_form.addRow(label_pid_kp, self.sb_pid_kp)

        label_pid_ki = QLabel("PID KI")
        label_pid_ki.setToolTip("Интегральный: накапливает ошибку для точности. Выше (25+) - лучше на движущихся. Для новичков: 20-30; если 'колеблется', уменьшите.")
        self.sb_pid_ki = QDoubleSpinBox()
        self.sb_pid_ki.setRange(0.0, 50.0)
        self.sb_pid_ki.setSingleStep(0.5)
        self.sb_pid_ki.setValue(cfg.pid_ki)
        self.sb_pid_ki.valueChanged.connect(lambda v: self._update_config("pid_ki", float(v)))
        self.sb_pid_ki.setToolTip(label_pid_ki.toolTip())
        aim_form.addRow(label_pid_ki, self.sb_pid_ki)

        label_pid_kd = QLabel("PID KD")
        label_pid_kd.setToolTip("Дифференциальный: гасит овершут. Выше (0.1-0.2) - smoother. Для новичков: 0.1; увеличьте, если бот 'перелетает' цель.")
        self.sb_pid_kd = QDoubleSpinBox()
        self.sb_pid_kd.setRange(0.0, 1.0)
        self.sb_pid_kd.setSingleStep(0.01)
        self.sb_pid_kd.setValue(cfg.pid_kd)
        self.sb_pid_kd.valueChanged.connect(lambda v: self._update_config("pid_kd", float(v)))
        self.sb_pid_kd.setToolTip(label_pid_kd.toolTip())
        aim_form.addRow(label_pid_kd, self.sb_pid_kd)

        label_prediction = QLabel("Предсказание кадров")
        label_prediction.setToolTip("Сколько кадров вперед предсказывать позицию цели на основе скорости. 0 - без предсказания. Для новичков: 1-3 для движущихся врагов; тестируйте, чтобы не 'промахивался'.")
        self.sb_prediction = QSpinBox()
        self.sb_prediction.setRange(0, 5)
        self.sb_prediction.setValue(cfg.prediction_frames)
        self.sb_prediction.valueChanged.connect(lambda v: self._update_config("prediction_frames", int(v)))
        self.sb_prediction.setToolTip(label_prediction.toolTip())
        aim_form.addRow(label_prediction, self.sb_prediction)

        label_deadzone = QLabel("Deadzone (px)")
        label_deadzone.setToolTip("Мертвая зона ошибки: внутри нее движение не подается.")
        self.sb_deadzone = QDoubleSpinBox()
        self.sb_deadzone.setRange(0.0, 10.0)
        self.sb_deadzone.setSingleStep(0.1)
        self.sb_deadzone.setValue(cfg.deadzone_px)
        self.sb_deadzone.valueChanged.connect(lambda v: self._update_config("deadzone_px", float(v)))
        aim_form.addRow(label_deadzone, self.sb_deadzone)

        label_deadzone_hyst = QLabel("Deadzone hysteresis (px)")
        label_deadzone_hyst.setToolTip("Доп. порог выхода из deadzone для предотвращения дрожания около нуля.")
        self.sb_deadzone_hyst = QDoubleSpinBox()
        self.sb_deadzone_hyst.setRange(0.0, 10.0)
        self.sb_deadzone_hyst.setSingleStep(0.1)
        self.sb_deadzone_hyst.setValue(cfg.deadzone_hysteresis_px)
        self.sb_deadzone_hyst.valueChanged.connect(lambda v: self._update_config("deadzone_hysteresis_px", float(v)))
        aim_form.addRow(label_deadzone_hyst, self.sb_deadzone_hyst)

        label_max_speed = QLabel("Макс скорость (px/sec)")
        self.sb_max_speed = QDoubleSpinBox()
        self.sb_max_speed.setRange(100.0, 30000.0)
        self.sb_max_speed.setSingleStep(100.0)
        self.sb_max_speed.setValue(cfg.max_speed_px_per_sec)
        self.sb_max_speed.valueChanged.connect(lambda v: self._update_config("max_speed_px_per_sec", float(v)))
        aim_form.addRow(label_max_speed, self.sb_max_speed)

        label_max_accel = QLabel("Макс ускорение (px/sec^2)")
        self.sb_max_accel = QDoubleSpinBox()
        self.sb_max_accel.setRange(100.0, 60000.0)
        self.sb_max_accel.setSingleStep(250.0)
        self.sb_max_accel.setValue(cfg.max_accel_px_per_sec2)
        self.sb_max_accel.valueChanged.connect(lambda v: self._update_config("max_accel_px_per_sec2", float(v)))
        aim_form.addRow(label_max_accel, self.sb_max_accel)

        self.cb_bezier = QCheckBox("Использовать кривые Безье для наведения")
        self.cb_bezier.setChecked(cfg.use_bezier)
        self.cb_bezier.toggled.connect(lambda v: self._update_config("use_bezier", bool(v)))
        self.cb_bezier.setToolTip("Если включено, траектория наведения будет curved (как человеческая), а не прямой. Для новичков: Включите для анти-детекта; выключите, если нужно быстро.")
        aim_form.addRow(self.cb_bezier)

        label_bezier_intensity = QLabel("Интенсивность кривизны Безье")
        label_bezier_intensity.setToolTip("Смещение контрольной точки для кривизны (0-50 px). Выше - сильнее изгиб (human-like), 0 - прямая линия.")
        self.sb_bezier_intensity = QDoubleSpinBox()
        self.sb_bezier_intensity.setRange(0.0, 50.0)
        self.sb_bezier_intensity.setSingleStep(1.0)
        self.sb_bezier_intensity.setValue(cfg.bezier_intensity)
        self.sb_bezier_intensity.valueChanged.connect(lambda v: self._update_config("bezier_intensity", float(v)))
        self.sb_bezier_intensity.setToolTip(label_bezier_intensity.toolTip())
        aim_form.addRow(label_bezier_intensity, self.sb_bezier_intensity)

        label_bezier_steps = QLabel("Шаги Безье")
        label_bezier_steps.setToolTip("Количество шагов движения по кривой (2-10). Больше - плавнее, но медленнее.")
        self.sb_bezier_steps = QSpinBox()
        self.sb_bezier_steps.setRange(2, 10)
        self.sb_bezier_steps.setValue(cfg.bezier_steps)
        self.sb_bezier_steps.valueChanged.connect(lambda v: self._update_config("bezier_steps", int(v)))
        self.sb_bezier_steps.setToolTip(label_bezier_steps.toolTip())
        aim_form.addRow(label_bezier_steps, self.sb_bezier_steps)

        aim_box.setLayout(aim_form)
        layout.addWidget(aim_box)

        # Группа Стрельба
        shoot_box = QGroupBox("Стрельба (Автоматическая стрельба)")
        shoot_form = QFormLayout()
        shoot_form.setSpacing(5)

        self.cb_autoshoot = QCheckBox("Авто-стрельба при точном наведении")
        self.cb_autoshoot.setChecked(cfg.auto_shoot)
        self.cb_autoshoot.toggled.connect(lambda v: self._update_config("auto_shoot", bool(v)))
        self.cb_autoshoot.setToolTip("Бот стреляет сам, когда цель в центре. Для новичков: Включите для бота; выключите, если хотите manual shoot, но это сделает бота пассивным.")
        shoot_form.addRow(self.cb_autoshoot)

        label_cd = QLabel("Задержка между выстрелами (сек)")
        label_cd.setToolTip("Время ожидания между shots. Меньше (0.05-0.1) - быстрый огонь, как auto. Больше (0.2) - single shots с контролем. Для новичков: 0.1 для пистолетов, 0.15 для rifles; тестируйте с recoil comp.")
        self.sb_cd = QDoubleSpinBox()
        self.sb_cd.setRange(0.03, 1.00)
        self.sb_cd.setSingleStep(0.01)
        self.sb_cd.setDecimals(2)
        self.sb_cd.setValue(cfg.shoot_cooldown_sec)
        self.sb_cd.valueChanged.connect(lambda v: self._update_config("shoot_cooldown_sec", float(v)))
        self.sb_cd.setToolTip(label_cd.toolTip())
        shoot_form.addRow(label_cd, self.sb_cd)

        label_click_hold = QLabel("Длительность нажатия ЛКМ (сек)")
        label_click_hold.setToolTip("Как долго держать кнопку мыши для выстрела. Коротко (0.01) - клик. Длиннее (0.05) - для burst. Для новичков: 0.01 для single, увеличьте с burst_shots.")
        self.sb_click_hold = QDoubleSpinBox()
        self.sb_click_hold.setRange(0.005, 0.200)
        self.sb_click_hold.setSingleStep(0.005)
        self.sb_click_hold.setDecimals(3)
        self.sb_click_hold.setValue(cfg.shoot_click_delay_sec)
        self.sb_click_hold.valueChanged.connect(lambda v: self._update_config("shoot_click_delay_sec", float(v)))
        self.sb_click_hold.setToolTip(label_click_hold.toolTip())
        shoot_form.addRow(label_click_hold, self.sb_click_hold)

        label_burst_shots = QLabel("Выстрелы в burst")
        label_burst_shots.setToolTip("Сколько shots за раз. 1 - single. 2-3 - burst. Для новичков: 1 для пистолетов, 3 для rifles; комбинируйте с cooldown.")
        self.sb_burst_shots = QSpinBox()
        self.sb_burst_shots.setRange(1, 10)
        self.sb_burst_shots.setValue(cfg.burst_shots)
        self.sb_burst_shots.valueChanged.connect(lambda v: self._update_config("burst_shots", int(v)))
        self.sb_burst_shots.setToolTip(label_burst_shots.toolTip())
        shoot_form.addRow(label_burst_shots, self.sb_burst_shots)

        label_recoil_comp = QLabel("Компенсация отдачи (px)")
        label_recoil_comp.setToolTip("Движение вниз после shot для контроля recoil. Выше (5-10) для rifles. Для новичков: 0 для пистолетов, 6-8 для AK; тестируйте в игре.")
        self.sb_recoil_comp = QDoubleSpinBox()
        self.sb_recoil_comp.setRange(0.0, 20.0)
        self.sb_recoil_comp.setSingleStep(0.5)
        self.sb_recoil_comp.setValue(cfg.recoil_comp_px)
        self.sb_recoil_comp.valueChanged.connect(lambda v: self._update_config("recoil_comp_px", float(v)))
        self.sb_recoil_comp.setToolTip(label_recoil_comp.toolTip())
        shoot_form.addRow(label_recoil_comp, self.sb_recoil_comp)

        shoot_box.setLayout(shoot_form)
        layout.addWidget(shoot_box)

        # Группа Навигация
        nav_box = QGroupBox("Навигация (Движение по маршруту)")
        nav_form = QFormLayout()
        nav_form.setSpacing(5)

        label_nav_gain = QLabel("Чувствительность поворота в навигации")
        label_nav_gain.setToolTip("Скорость поворота камеры при следовании маршруту. Выше (0.5+) - быстрее доходит. Ниже (0.2) - плавнее. Для новичков: 0.4; если бот 'крутит слишком резко', уменьшите.")
        self.sb_nav_gain = QDoubleSpinBox()
        self.sb_nav_gain.setRange(0.05, 1.50)
        self.sb_nav_gain.setSingleStep(0.01)
        self.sb_nav_gain.setValue(cfg.nav_mouse_gain)
        self.sb_nav_gain.valueChanged.connect(lambda v: self._update_config("nav_mouse_gain", float(v)))
        self.sb_nav_gain.setToolTip(label_nav_gain.toolTip())
        nav_form.addRow(label_nav_gain, self.sb_nav_gain)

        self.cb_pause_nav = QCheckBox("Ставить навигацию на паузу при контакте")
        self.cb_pause_nav.setChecked(cfg.nav_pause_when_enemy)
        self.cb_pause_nav.toggled.connect(lambda v: self._update_config("nav_pause_when_enemy", bool(v)))
        self.cb_pause_nav.setToolTip("Бот останавливается, когда видит врага, чтобы стрелять. Для новичков: Включите для deathmatch; выключите, если хотите бежать через врагов.")
        nav_form.addRow(self.cb_pause_nav)

        nav_box.setLayout(nav_form)
        layout.addWidget(nav_box)

        # Группа Тестовый режим
        test_box = QGroupBox("Тестовый режим (Отладка на столе)")
        test_form = QFormLayout()
        test_form.setSpacing(5)

        self.cb_desktop_test = QCheckBox("Режим проверки на рабочем столе")
        self.cb_desktop_test.setChecked(cfg.desktop_test_mode)
        self.cb_desktop_test.toggled.connect(lambda v: self._update_config("desktop_test_mode", bool(v)))
        self.cb_desktop_test.setToolTip("Тест без игры: бот прицеливается на фото/видео на столе. Для новичков: Включите (F8) для настройки, выключите в игре.")
        test_form.addRow(self.cb_desktop_test)

        self.cb_desktop_autoclick = QCheckBox("В desktop-режиме тоже стрелять")
        self.cb_desktop_autoclick.setChecked(cfg.desktop_test_autoclick)
        self.cb_desktop_autoclick.toggled.connect(lambda v: self._update_config("desktop_test_autoclick", bool(v)))
        self.cb_desktop_autoclick.setToolTip("В тесте бот кликает (симулирует стрельбу). Для новичков: Включите для проверки auto_shoot.")
        test_form.addRow(self.cb_desktop_autoclick)

        self.cb_desktop_abs = QCheckBox("Desktop-тест: принудительно absolute")
        self.cb_desktop_abs.setChecked(cfg.desktop_force_absolute)
        self.cb_desktop_abs.toggled.connect(lambda v: self._update_config("desktop_force_absolute", bool(v)))
        self.cb_desktop_abs.setToolTip("В тесте использовать прямое позиционирование. Для новичков: Включите, если relative не работает на столе.")
        test_form.addRow(self.cb_desktop_abs)

        test_box.setLayout(test_form)
        layout.addWidget(test_box)

        # Группа Общее
        general_box = QGroupBox("Общие настройки")
        general_form = QFormLayout()
        general_form.setSpacing(5)

        label_cycle_sleep = QLabel("Задержка цикла (сек)")
        label_cycle_sleep.setToolTip("Пауза между фреймами бота. Меньше (0.005) - выше FPS, быстрее реакция, но нагрузка на ПК. Больше (0.01) - стабильнее. Для новичков: 0.007; уменьшите на мощном ПК.")
        self.sb_cycle_sleep = QDoubleSpinBox()
        self.sb_cycle_sleep.setRange(0.001, 0.100)
        self.sb_cycle_sleep.setSingleStep(0.001)
        self.sb_cycle_sleep.setDecimals(3)
        self.sb_cycle_sleep.setValue(cfg.cycle_sleep_sec)
        self.sb_cycle_sleep.valueChanged.connect(lambda v: self._update_config("cycle_sleep_sec", float(v)))
        self.sb_cycle_sleep.setToolTip(label_cycle_sleep.toolTip())
        general_form.addRow(label_cycle_sleep, self.sb_cycle_sleep)

        label_model_type = QLabel("Тип модели")
        label_model_type.setToolTip("Выберите формат модели: 'pytorch' - стандартный, 'tensorrt' - ускоренный для NVIDIA GPU (если CUDA доступна).")
        self.combo_model_type = QComboBox()
        self.combo_model_type.addItems(["pytorch", "tensorrt"])
        self.combo_model_type.setCurrentText(cfg.model_type)
        self.combo_model_type.currentTextChanged.connect(lambda v: self._update_config("model_type", v))
        self.combo_model_type.setToolTip(label_model_type.toolTip())
        general_form.addRow(label_model_type, self.combo_model_type)

        self.cb_fp16 = QCheckBox("Использовать FP16 для TensorRT")
        self.cb_fp16.setChecked(cfg.use_fp16)
        self.cb_fp16.toggled.connect(lambda v: self._update_config("use_fp16", bool(v)))
        self.cb_fp16.setToolTip("Включить FP16 (half precision) для ускорения на GPU.")
        general_form.addRow(self.cb_fp16)

        self.cb_int8 = QCheckBox("Использовать INT8 для TensorRT")  # 🔥 Новый чекбокс
        self.cb_int8.setChecked(cfg.use_int8)
        self.cb_int8.toggled.connect(lambda v: self._update_config("use_int8", bool(v)))
        self.cb_int8.setToolTip("Включить INT8 для максимального ускорения (с небольшим снижением точности).")
        general_form.addRow(self.cb_int8)

        # 🔥 Новые параметры в группе Общее
        label_random_variance = QLabel("Рандомизация таймингов (± сек)")
        label_random_variance.setToolTip("Добавляет случайность к задержкам (shoot_cooldown, shoot_click_delay, cycle_sleep). 0 - без рандома. Для human-like: 0.02.")
        self.sb_random_variance = QDoubleSpinBox()
        self.sb_random_variance.setRange(0.0, 0.1)
        self.sb_random_variance.setSingleStep(0.005)
        self.sb_random_variance.setDecimals(3)
        self.sb_random_variance.setValue(cfg.random_timing_variance)
        self.sb_random_variance.valueChanged.connect(lambda v: self._update_config("random_timing_variance", float(v)))
        self.sb_random_variance.setToolTip(label_random_variance.toolTip())
        general_form.addRow(label_random_variance, self.sb_random_variance)

        label_switch_delay = QLabel("Задержка переключения целей (сек)")
        label_switch_delay.setToolTip("Пауза перед захватом новой цели после потери предыдущей (100-200 мс). Для анти-детекта: 0.15.")
        self.sb_switch_delay = QDoubleSpinBox()
        self.sb_switch_delay.setRange(0.0, 0.5)
        self.sb_switch_delay.setSingleStep(0.01)
        self.sb_switch_delay.setDecimals(2)
        self.sb_switch_delay.setValue(cfg.target_switch_delay_sec)
        self.sb_switch_delay.valueChanged.connect(lambda v: self._update_config("target_switch_delay_sec", float(v)))
        self.sb_switch_delay.setToolTip(label_switch_delay.toolTip())
        general_form.addRow(label_switch_delay, self.sb_switch_delay)

        general_box.setLayout(general_form)
        layout.addWidget(general_box)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _update_config(self, key: str, value) -> None:
        setattr(self.state.config, key, value)
        self.save_config()  # 🔥 Сохранение при каждом изменении

    def _update_fov(self, v: int) -> None:
        nearest = round(v / 32) * 32
        self.sb_fov.blockSignals(True)
        self.sb_fov.setValue(nearest)
        self.sb_fov.blockSignals(False)
        self._update_config("combat_fov", nearest)
        self.state.fov_changed_signal.emit(nearest)

    def _set_routes(self, routes: list[str]) -> None:
        self.route_list.clear()
        self.route_list.addItems(routes)

    def refresh_routes(self) -> None:
        self._set_routes([p.name for p in sorted(self.routes_dir.glob("*.txt"))])

    def delete_selected(self) -> None:
        item = self.route_list.currentItem()
        if item is None:
            return
        route_path = self.routes_dir / item.text()
        spawn_path = self.routes_dir.parent / "spawns" / f"{route_path.stem}.png"
        if route_path.exists():
            route_path.unlink()
        if spawn_path.exists():
            spawn_path.unlink()
        self.refresh_routes()

    def load_selected(self) -> None:
        item = self.route_list.currentItem()
        if item is None:
            QMessageBox.warning(self, "Внимание", "Выберите маршрут в списке.")
            return
        self.state.manual_route = self.routes_dir / item.text()
        self.update_status(f"Статус: Маршрут выбран вручную: {item.text()}")

    def toggle_running(self) -> None:
        self.state.running = not self.state.running
        self.update_status("Статус: Запущен" if self.state.running else "Статус: Остановлен")

    def update_status(self, text: str) -> None:
        self.status_lbl.setText(text)

    def append_log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_view.append(f"[{ts}] {text}")
        cursor = self.log_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_view.setTextCursor(cursor)

    def update_preview(self, frame: np.ndarray) -> None:
        h, w, _ = frame.shape
        q = QImage(frame.data, w, h, w * 3, QImage.Format.Format_RGB888).rgbSwapped()
        self.preview.setPixmap(QPixmap.fromImage(q).scaled(560, 480))

    def save_config(self) -> None:
        """Сохраняет конфигурацию в JSON."""
        config_dict = asdict(self.state.config)
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)

    def load_config(self) -> None:
        """Загружает конфигурацию из JSON и применяет к GUI."""
        if not self.config_file.exists():
            return
        with open(self.config_file, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            setattr(self.state.config, key, value)

        # По запросу: на старте стрельба выключена, только наведение
        self.state.config.auto_shoot = False
        self.state.config.desktop_test_autoclick = False

        # Применяем к виджетам
        self.sb_conf.setValue(self.state.config.conf_threshold)
        self.combo_target_class.setCurrentText(self.state.config.target_class)
        self.cb_nearest.setChecked(self.state.config.nearest_target)
        self.combo_mouse_mode.setCurrentText(self.state.config.mouse_mode)
        self.sb_fov.setValue(self.state.config.combat_fov)
        self.sb_aim_x.setValue(self.state.config.aim_gain_x)
        self.sb_aim_y.setValue(self.state.config.aim_gain_y)
        self.sb_aim_max_step.setValue(self.state.config.aim_max_step_px)
        self.sb_center.setValue(self.state.config.center_radius_px)
        self.sb_head_offset.setValue(self.state.config.aim_head_offset_percent)  # 🔥 Изменено
        self.sb_mouse_mult.setValue(self.state.config.mouse_multiplier)
        self.sb_pid_kp.setValue(self.state.config.pid_kp)
        self.sb_pid_ki.setValue(self.state.config.pid_ki)
        self.sb_pid_kd.setValue(self.state.config.pid_kd)
        self.sb_prediction.setValue(self.state.config.prediction_frames)
        self.sb_deadzone.setValue(self.state.config.deadzone_px)
        self.sb_deadzone_hyst.setValue(self.state.config.deadzone_hysteresis_px)
        self.sb_max_speed.setValue(self.state.config.max_speed_px_per_sec)
        self.sb_max_accel.setValue(self.state.config.max_accel_px_per_sec2)
        self.cb_bezier.setChecked(self.state.config.use_bezier)
        self.sb_bezier_intensity.setValue(self.state.config.bezier_intensity)
        self.sb_bezier_steps.setValue(self.state.config.bezier_steps)
        self.cb_autoshoot.setChecked(self.state.config.auto_shoot)
        self.sb_cd.setValue(self.state.config.shoot_cooldown_sec)
        self.sb_click_hold.setValue(self.state.config.shoot_click_delay_sec)
        self.sb_burst_shots.setValue(self.state.config.burst_shots)
        self.sb_recoil_comp.setValue(self.state.config.recoil_comp_px)
        self.sb_nav_gain.setValue(self.state.config.nav_mouse_gain)
        self.cb_pause_nav.setChecked(self.state.config.nav_pause_when_enemy)
        self.cb_desktop_test.setChecked(self.state.config.desktop_test_mode)
        self.cb_desktop_autoclick.setChecked(self.state.config.desktop_test_autoclick)
        self.cb_desktop_abs.setChecked(self.state.config.desktop_force_absolute)
        self.sb_cycle_sleep.setValue(self.state.config.cycle_sleep_sec)
        self.combo_model_type.setCurrentText(self.state.config.model_type)
        self.cb_fp16.setChecked(self.state.config.use_fp16)
        self.cb_int8.setChecked(self.state.config.use_int8)  # 🔥 Новый
        # 🔥 Новые
        self.sb_random_variance.setValue(self.state.config.random_timing_variance)
        self.sb_switch_delay.setValue(self.state.config.target_switch_delay_sec)


# Оверлеи (StatusOverlay, RecorderOverlay, FovOverlay, TargetLockOverlay) — без изменений (копируй из предыдущего)
class StatusOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        self.lbl = QLabel("BOT: ВЫКЛ")
        self.lbl.setStyleSheet(
            "color: white; background: rgba(0,0,0,140); border: 1px solid rgba(255,255,255,80);"
            "font-weight: bold; padding: 6px 10px; border-radius: 6px;"
        )
        layout.addWidget(self.lbl)
        self.setLayout(layout)
        self.move(20, 20)

    def update_status(self, text: str) -> None:
        on = "Запущен" in text or "ВКЛ" in text
        self.lbl.setText("BOT: ВКЛ" if on else "BOT: ВЫКЛ")
        self.lbl.setStyleSheet(
            f"color: {'#8CFF8C' if on else '#FF8C8C'}; background: rgba(0,0,0,140);"
            "border: 1px solid rgba(255,255,255,80); font-weight: bold;"
            "padding: 6px 10px; border-radius: 6px;"
        )
        self.update()  # 🔥 Форсируем обновление оверлея


class RecorderOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        self.lbl = QLabel("REC: ВЫКЛ")
        self.lbl.setStyleSheet(
            "color: white; background: rgba(0,0,0,140); border: 1px solid rgba(255,255,255,80);"
            "font-weight: bold; padding: 6px 10px; border-radius: 6px;"
        )
        layout.addWidget(self.lbl)
        self.setLayout(layout)
        self.move(150, 20)  # Размещаем рядом со StatusOverlay (горизонтально)

    def update_status(self, text: str) -> None:
        on = "ВКЛ" in text
        self.lbl.setText("REC: ВКЛ" if on else "REC: ВЫКЛ")
        self.lbl.setStyleSheet(
            f"color: {'#FF8C8C' if on else '#8CFF8C'}; background: rgba(0,0,0,140);"  # Красный для записи ВКЛ, зеленый для ВЫКЛ
            "border: 1px solid rgba(255,255,255,80); font-weight: bold;"
            "padding: 6px 10px; border-radius: 6px;"
        )
        self.update()  # 🔥 Форсируем обновление оверлея


class FovOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        
        self.fov_radius = 210
        self.is_active = False

    def set_radius(self, fov_px: int):
        self.fov_radius = fov_px // 2
        self.update()

    def set_active(self, active: bool):
        self.is_active = active
        self.update()

    def paintEvent(self, event):
        if not self.is_active:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        pen = QPen(QColor(0, 255, 0, 120)) 
        pen.setWidth(2)
        painter.setPen(pen)
        
        center = self.rect().center()
        painter.drawEllipse(center, self.fov_radius, self.fov_radius)


class TargetLockOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        
        self.boxes: List[Tuple[int, int, int, int]] = []
        self.is_active = False
        
        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self.update)
        self.pulse_timer.start(60)
        
        self.pulse_phase = 0.0

    def set_boxes(self, boxes: List[Tuple[int, int, int, int]]):
        self.boxes = boxes
        self.update()

    def set_active(self, active: bool):
        self.is_active = active
        self.update()

    def paintEvent(self, event):
        if not self.is_active or not self.boxes:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        self.pulse_phase += 0.12
        
        for x1, y1, x2, y2 in self.boxes:
            rect = QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            
            pulse_width = 3 + int(1.5 * abs(np.sin(self.pulse_phase)))
            pen = QPen(QColor(0, 255, 0, 220))
            pen.setWidth(pulse_width)
            painter.setPen(pen)
            painter.drawRect(rect)
            
            inner_pen = QPen(QColor(0, 255, 0, 90))
            inner_pen.setWidth(6)
            painter.setPen(inner_pen)
            painter.drawRect(rect.adjusted(8, 8, -8, -8))
            
            font = QFont("Courier New", 12, QFont.Weight.Bold)
            painter.setFont(font)
            lock_text = "TARGET LOCK"
            fm = painter.fontMetrics()
            text_width = fm.horizontalAdvance(lock_text)
            text_x = rect.center().x() - text_width // 2
            text_y = rect.top() - 14
            
            painter.setPen(QColor(0, 0, 0, 160))
            painter.drawText(int(text_x + 1.5), int(text_y + 1.5), lock_text)
            
            alpha = 200 + int(55 * abs(np.sin(self.pulse_phase * 1.8)))
            painter.setPen(QColor(0, 255, 0, alpha))
            painter.drawText(int(text_x), int(text_y), lock_text)
            
            corner_size = 12
            corner_pen = QPen(QColor(0, 255, 0, 140))
            corner_pen.setWidth(2)
            painter.setPen(corner_pen)
            painter.drawLine(int(x1), int(y1), int(x1 + corner_size), int(y1))
            painter.drawLine(int(x1), int(y1), int(x1), int(y1 + corner_size))
            painter.drawLine(int(x2), int(y1), int(x2 - corner_size), int(y1))
            painter.drawLine(int(x2), int(y1), int(x2), int(y1 + corner_size))
            painter.drawLine(int(x1), int(y2), int(x1 + corner_size), int(y2))
            painter.drawLine(int(x1), int(y2), int(x1), int(y2 - corner_size))
            painter.drawLine(int(x2), int(y2), int(x2 - corner_size), int(y2))
            painter.drawLine(int(x2), int(y2), int(x2), int(y2 - corner_size))


# 🔥 Новый оверлей для FPS и Latency
class PerformanceOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        self.lbl = QLabel("FPS: 0 | Latency: 0 ms")
        self.lbl.setStyleSheet(
            "color: white; background: rgba(0,0,0,140); border: 1px solid rgba(255,255,255,80);"
            "font-weight: bold; padding: 6px 10px; border-radius: 6px;"
        )
        layout.addWidget(self.lbl)
        self.setLayout(layout)
        self.move(300, 20)  # Размещаем рядом с другими оверлеями

    def update_performance(self, fps: float, latency: float, t_capture: float, t_infer: float, t_post: float, t_input: float) -> None:
        self.lbl.setText(f"FPS: {fps:.1f} | Lat (общая задержка): {latency:.0f}ms | Cap (захват экрана): {t_capture:.0f}ms | Inf (инференс модели): {t_infer:.0f}ms | Post (постобработка): {t_post:.0f}ms | Inp (ввод команд): {t_input:.0f}ms")
        self.update()  # Форсируем обновление


class Kalman2D:
    """Расширенный Калман-фильтр состояния [x, y, vx, vy, ax, ay] для учета ускорения."""  # 🔥 Расширено на ускорение

    def __init__(self, process_var: float = 25.0, measurement_var: float = 36.0):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.x = np.zeros((6, 1), dtype=np.float64)  # [x, y, vx, vy, ax, ay]
        self.P = np.eye(6, dtype=np.float64) * 500.0
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float64)
        self.R = np.eye(2, dtype=np.float64) * measurement_var
        self.initialized = False

    def reset(self, px: Optional[float] = None, py: Optional[float] = None) -> None:
        self.x[:] = 0.0
        self.P = np.eye(6, dtype=np.float64) * 500.0
        self.initialized = px is not None and py is not None
        if self.initialized:
            self.x[0, 0] = float(px)
            self.x[1, 0] = float(py)

    def update(self, px: float, py: float, dt: float) -> tuple[float, float, float, float, float, float]:
        dt = max(0.001, min(dt, 0.20))
        dt2 = dt * dt / 2.0
        F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float64)
        q = self.process_var
        G = np.array([[dt2], [dt2], [dt], [dt], [1], [1]], dtype=np.float64)
        Q = (G @ G.T) * q

        if not self.initialized:
            self.reset(px, py)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        z = np.array([[float(px)], [float(py)]], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0]), float(self.x[4, 0]), float(self.x[5, 0])


# 🔥 Новый PID класс
class PID:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = 250.0
        self.clear()

    def clear(self):
        self.prev_err = 0.0
        self.integral = 0.0

    def step(self, err: float, dt: float) -> float:
        self.integral += err * dt
        self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (err - self.prev_err) / dt if dt > 0.0001 else 0.0
        output = self.kp * err + self.ki * self.integral + self.kd * derivative
        self.prev_err = err
        return output


# 🔥 Функция для YOLO процесса
def yolo_detection_process(input_q: mp.Queue, output_q: mp.Queue, device: str, model_type: str, conf_threshold: float, use_fp16: bool, use_int8: bool):
    model_path = "yolov10n.pt"  # 🔥 Изменено на YOLOv10n
    if device == "cuda" and model_type == "tensorrt":
        engine_path = "yolov10n.engine"
        if not os.path.exists(engine_path):
            print("Экспорт модели в TensorRT .engine...")
            model = YOLO(model_path)
            model.export(format="engine", device=0, half=use_fp16, int8=use_int8, dynamic=True)
            print("Экспорт завершен!")
        model_path = engine_path
    model = YOLO(model_path, task='detect')  # Load without .to() initially
    if model_path.endswith('.pt'):  # Only apply .to(device) for PyTorch models
        model = model.to(device)
    print(f"Модель в процессе загружена на {device}")

    while True:
        data = input_q.get()
        if data is None:
            break
        frame_id, frame_ts, frame_bgr, imgsz = data
        t_infer_start = time.time()
        results = list(model.predict(frame_bgr, imgsz=imgsz, conf=conf_threshold, verbose=False, half=use_fp16))[0]  # 🔥 Fix: list() для generator
        t_infer = (time.time() - t_infer_start) * 1000
        output_q.put((frame_id, frame_ts, results, t_infer))

    print("YOLO процесс завершен")


class CombatWalkBot(QThread):  # 🔥 Изменено на QThread
    MINIMAP_REGION = {"left": 40, "top": 40, "width": 220, "height": 220}

    def __init__(self, state: BotState, base_dir: Path):
        super().__init__()
        self.state = state
        self.base_dir = base_dir

        self.routes_dir = self.base_dir / "routes"
        self.spawns_dir = self.base_dir / "spawns"
        self.routes_dir.mkdir(parents=True, exist_ok=True)
        self.spawns_dir.mkdir(parents=True, exist_ok=True)

        self.arrow_template = self.base_dir / "arrow.png"
        if not self.arrow_template.exists():
            raise FileNotFoundError("Нужен шаблон arrow.png в корне проекта.")
        self.arrow_template_img = cv2.imread(str(self.arrow_template), cv2.IMREAD_COLOR)
        if self.arrow_template_img is None:
            raise FileNotFoundError("Не удалось загрузить шаблон arrow.png")

        self.input = InputController()
        self.spawn_detector = SpawnDetector(self.spawns_dir, self.routes_dir, self.MINIMAP_REGION, threshold=0.80)
        self.recorder = RouteRecorder(
            routes_dir=self.routes_dir,
            spawns_dir=self.spawns_dir,
            arrow_template_path=self.arrow_template,
            minimap_region=self.MINIMAP_REGION,
            sample_distance_px=15.0,
        )

        self.current_navigator: Optional[RouteNavigator] = None
        self.route_name_for_record = "route_1"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_shot_time = 0.0
        self._last_target_log = 0.0
        self._last_perf_log = 0.0
        self._desktop_prev = self.state.config.desktop_test_mode

        # 🔥 Новый PID с config
        cfg = self.state.config
        self.pid_x = PID(cfg.pid_kp, cfg.pid_ki, cfg.pid_kd)
        self.pid_y = PID(cfg.pid_kp, cfg.pid_ki, cfg.pid_kd)
        self.last_pid_time = time.time()
        self.prev_has_target = False
        self.prev_deadzone_active = False
        self.frame_seq = 0
        self.last_detection_ts = time.time()
        self.kalman = Kalman2D(process_var=20.0, measurement_var=30.0)  # 🔥 Расширенный Kalman

        # 🔥 Input queue
        self.input_queue = Queue()
        self.input_worker = threading.Thread(target=self._input_worker, daemon=True)
        self.input_worker.start()

        # 🔥 Для предсказания
        self.last_tx = 0.0
        self.last_ty = 0.0
        self.tx_vel = 0.0
        self.ty_vel = 0.0

        # 🔥 Плавность и скорость движения мыши
        self.prev_move_x = 0.0
        self.prev_move_y = 0.0
        self.prev_vel_x = 0.0  # 🔥 Новый для velocity
        self.prev_vel_y = 0.0
        self.filtered_tx: Optional[float] = None
        self.filtered_ty: Optional[float] = None

        # 🔥 Stop event
        self.stop_event = threading.Event()

        # 🔥 Для задержки переключения целей
        self.next_allowed_target_time = 0.0
        self.last_target_bbox: Optional[tuple[float, float, float, float]] = None

        # 🔥 Для FPS и latency
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0.0
        self.latency = 0.0

        # 🔥 Для замеров
        self.t_capture_list = []
        self.t_infer_list = []
        self.t_post_list = []
        self.t_input_list = []
        self.latency_list = []

        # 🔥 Логирование траектории наведения в файл для последующего анализа
        self.debug_dir = self.base_dir / "logs"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.aim_debug_file = self.debug_dir / f"aim_debug_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._last_debug_dump = 0.0

        # 🔥 Multiprocessing для YOLO
        self.input_yolo_q = mp.Queue(maxsize=1)
        self.output_yolo_q = mp.Queue(maxsize=1)
        self.yolo_proc = mp.Process(target=yolo_detection_process, args=(self.input_yolo_q, self.output_yolo_q, self.device, cfg.model_type, cfg.conf_threshold, cfg.use_fp16, cfg.use_int8))
        self.yolo_proc.start()

        # Добавь: Если no_target долго, reset PID
        self.pid_x.clear()
        self.pid_y.clear()
        self._last_error_dist = 0  # Для трекинга изменений

        # 🔥 MSS для захвата экрана
        self.cam = mss.mss()

    def _get_screen_size(self) -> tuple[int, int]:
        # MSS: monitors[0] содержит виртуальный экран (all monitors).
        try:
            mon = self.cam.monitors[0]
            return int(mon["width"]), int(mon["height"])
        except Exception:
            pass

        # fallback для окружений без доступа к монитору.
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.geometry()
            if geo.width() > 0 and geo.height() > 0:
                return int(geo.width()), int(geo.height())

        return 1920, 1080

    def _input_worker(self) -> None:
        while True:
            command = self.input_queue.get()
            cmd_type = command[0]
            args = command[1:]
            if cmd_type == 'move':
                self.input.move_mouse(*args)
            elif cmd_type == 'move_absolute':
                self.input.move_mouse_absolute(*args)
            elif cmd_type == 'shoot':
                self.input.shoot(*args)
            elif cmd_type == 'key_down':
                self.input.key_down(*args)
            elif cmd_type == 'key_up':
                self.input.key_up(*args)
            elif cmd_type == 'release_all':
                self.input.release_all()

    def log(self, text: str) -> None:
        self.state.log_signal.emit(text)

    def _write_aim_debug(self, payload: dict) -> None:
        payload = dict(payload)
        payload.setdefault("ts", time.time())
        try:
            with open(self.aim_debug_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            self.state.log_signal.emit(f"⚠️ Ошибка записи debug-лога: {e}")

    def _load_route(self, route_path: Path) -> None:
        if not route_path.exists():
            return
        nav = RouteNavigator.from_txt(route_path)
        nav.mouse_gain = self.state.config.nav_mouse_gain
        self.current_navigator = nav
        self.state.status_signal.emit(f"Статус: Загружен маршрут {route_path.name}")
        self.log(f"Маршрут загружен: {route_path.name}")

    def _detect_arrow_position(self, minimap_bgr: np.ndarray) -> Optional[tuple[float, float]]:
        tpl = self.arrow_template_img
        res = cv2.matchTemplate(minimap_bgr, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < 0.55:
            return None
        h, w = tpl.shape[:2]
        return float(max_loc[0] + w // 2), float(max_loc[1] + h // 2)

    def _choose_target(self, boxes, center: int, cfg: AimConfig):
        if len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu()
        conf = boxes.conf.cpu()
        cls = boxes.cls.cpu()

        # Фильтр по классу
        if cfg.target_class == "person":
            mask = cls == 0
        else:
            mask = torch.ones_like(cls, dtype=torch.bool)

        # Фильтр по conf
        mask = mask & (conf >= cfg.conf_threshold)

        if not mask.any():
            return None

        xyxy = xyxy[mask]
        conf = conf[mask]

        # Вычисляем центры
        tx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        ty = (xyxy[:, 1] + xyxy[:, 3]) / 2

        # Дистанции
        dists = ((tx - center) ** 2 + (ty - center) ** 2) ** 0.5

        if self.last_target_bbox is not None:
            lx1, ly1, lx2, ly2 = self.last_target_bbox
            inter_x1 = torch.maximum(xyxy[:, 0], torch.tensor(lx1))
            inter_y1 = torch.maximum(xyxy[:, 1], torch.tensor(ly1))
            inter_x2 = torch.minimum(xyxy[:, 2], torch.tensor(lx2))
            inter_y2 = torch.minimum(xyxy[:, 3], torch.tensor(ly2))
            inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
            inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
            inter_area = inter_w * inter_h
            prev_area = max((lx2 - lx1) * (ly2 - ly1), 1.0)
            current_area = torch.clamp((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]), min=1e-6)
            union = torch.tensor(prev_area) + current_area - inter_area
            iou = inter_area / torch.clamp(union, min=1e-6)
        else:
            iou = torch.zeros_like(conf)

        if cfg.nearest_target:
            score = 0.62 * (1.0 / (dists + 1.0)) + 0.23 * conf + 0.15 * iou
        else:
            score = 0.7 * conf + 0.2 * (1.0 / (dists + 1.0)) + 0.1 * iou

        idx = score.argmax()

        x1, y1, x2, y2 = xyxy[idx].tolist()
        return dists[idx], conf[idx], x1, y1, x2, y2, tx[idx], ty[idx]

    def _combat_step(self, frame_bgr: np.ndarray, center: int, reg: dict, desktop_mode: bool = False) -> tuple[bool, np.ndarray]:
        cfg = self.state.config
        # 🔥 Update PID if changed
        self.pid_x.kp = cfg.pid_kp
        self.pid_x.ki = cfg.pid_ki
        self.pid_x.kd = cfg.pid_kd
        self.pid_y.kp = cfg.pid_kp
        self.pid_y.ki = cfg.pid_ki
        self.pid_y.kd = cfg.pid_kd
        self.input.mouse_multiplier = cfg.mouse_multiplier

        # 🔥 Округление imgsz до ближайшего кратного 32, но для tensorrt фиксируем 416
        imgsz = 416 if cfg.model_type == "tensorrt" else round(cfg.combat_fov / 32) * 32

        # 🔥 Отправляем фрейм в YOLO процесс и измеряем latency
        predict_start = time.time()
        self.frame_seq += 1
        frame_id = self.frame_seq
        frame_ts = time.time()
        while self.input_yolo_q.qsize() > 0:
            try:
                self.input_yolo_q.get_nowait()
            except Exception:
                break
        self.input_yolo_q.put((frame_id, frame_ts, frame_bgr, imgsz))
        out_frame_id, out_frame_ts, results, t_infer = self.output_yolo_q.get()
        while out_frame_id < frame_id and not self.output_yolo_q.empty():
            try:
                out_frame_id, out_frame_ts, results, t_infer = self.output_yolo_q.get_nowait()
            except Exception:
                break
        self.last_detection_ts = out_frame_ts
        t_queue = (time.time() - predict_start) * 1000  # Время на очередь + infer

        self.t_infer_list.append(t_infer)
        self.latency_list.append(t_queue)

        boxes = results.boxes  # 🔥 Fix: results — Results, not list[Results]

        now = time.time()

        # 🔥 Target switching delay: Если задержка активна, игнорируем цели
        if now < self.next_allowed_target_time:
            self.state.target_lock_signal.emit([])
            self.prev_has_target = False
            self.last_target_bbox = None
            self.prev_move_x = 0.0
            self.prev_move_y = 0.0
            self.filtered_tx = None
            self.filtered_ty = None
            self.tx_vel = 0.0
            self.ty_vel = 0.0
            self.prev_deadzone_active = False
            self.kalman.reset()
            if now - self._last_debug_dump > 0.20:
                self._write_aim_debug({
                    "event": "lock_delay",
                    "next_allowed_target_time": self.next_allowed_target_time,
                    "queue_size": self.input_queue.qsize(),
                    "t_infer_ms": round(t_infer, 3),
                    "t_queue_ms": round(t_queue, 3),
                    "out_frame_id": out_frame_id,
                    "expected_frame_id": frame_id,
                })
                self._last_debug_dump = now
            return False, frame_bgr

        t_post_start = time.time()
        target = self._choose_target(boxes, center, cfg)
        if target is None:
            if self.prev_has_target:
                # 🔥 Установим задержку при потере цели
                self.next_allowed_target_time = now + cfg.target_switch_delay_sec
                self.log(f"Цель потеряна, задержка переключения: {cfg.target_switch_delay_sec:.2f} сек")
            self.state.target_lock_signal.emit([])
            self.prev_has_target = False
            self.last_target_bbox = None
            self.prev_move_x = 0.0
            self.prev_move_y = 0.0
            self.filtered_tx = None
            self.filtered_ty = None
            self.tx_vel = 0.0
            self.ty_vel = 0.0
            self.prev_deadzone_active = False
            self.kalman.reset()
            self.pid_x.clear()
            self.pid_y.clear()
            if now - self._last_debug_dump > 0.20:
                self._write_aim_debug({
                    "event": "no_target",
                    "queue_size": self.input_queue.qsize(),
                    "t_infer_ms": round(t_infer, 3),
                    "t_queue_ms": round(t_queue, 3),
                    "conf_threshold": cfg.conf_threshold,
                    "out_frame_id": out_frame_id,
                    "expected_frame_id": frame_id,
                })
                self._last_debug_dump = now
            return False, frame_bgr

        # После target choice, if target not None, before PID.
        if t_infer > 100:  # Если inference >100 ms, пропусти движение (чтобы не лагало)
            self.log("Пропуск движения: высокая задержка inference")
            return True, frame_bgr

        _, conf, x1, y1, x2, y2, tx, ty = target
        height = y2 - y1

        # 🔥 Dynamic head offset: адаптация под дистанцию/размер бокса в FOV.
        # Маленький бокс (дальняя цель) => чуть выше в голову; крупный (близко) => мягче.
        relative_box_height = clamp(height / max(1.0, float(cfg.combat_fov)), 0.08, 0.70)
        adaptive_head_percent = cfg.aim_head_offset_percent + (0.35 - relative_box_height) * 0.18
        adaptive_head_percent = clamp(adaptive_head_percent, 0.12, 0.45)
        ty -= height * adaptive_head_percent

        # 🔥 Kalman filter [x,y,vx,vy,ax,ay] + предсказание при пропусках детекции
        raw_tx = float(tx)
        raw_ty = float(ty)
        dt_det = max(0.001, min(time.time() - self.last_detection_ts, 0.20))
        kx, ky, kvx, kvy, kax, kay = self.kalman.update(raw_tx, raw_ty, dt_det)

        prediction_time = max(0.0, float(cfg.prediction_frames)) * dt_det
        tx = kx + kvx * prediction_time + 0.5 * kax * (prediction_time ** 2)
        ty = ky + kvy * prediction_time + 0.5 * kay * (prediction_time ** 2)
        self.tx_vel = kvx
        self.ty_vel = kvy
        self.last_tx = tx
        self.last_ty = ty

        # 🔥 Рисуем бокс (предполагаем, что fov_frame - это frame_bgr)
        cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        cv2.line(frame_bgr, (center, center), (int(tx), int(ty)), (0, 255, 0), 2)

        screen_left = reg["left"]
        screen_top = reg["top"]
        abs_x1 = int(screen_left + x1)
        abs_y1 = int(screen_top + y1)
        abs_x2 = int(screen_left + x2)
        abs_y2 = int(screen_top + y2)
        self.state.target_lock_signal.emit([(abs_x1, abs_y1, abs_x2, abs_y2)])
        self.last_target_bbox = (x1, y1, x2, y2)

        if now - self._last_target_log > 1.5:
            self.log(f"Цель обнаружена: conf={conf:.2f}, режим={cfg.mouse_mode}")
            self._last_target_log = now

        # 🔥 Новый PID с dt
        dt = now - self.last_pid_time
        if dt > 0.1: dt = 0.1  # Cap
        self.last_pid_time = now

        # Clear PID если новая цель (после потери)
        if not self.prev_has_target:
            self.pid_x.clear()
            self.pid_y.clear()
            self.prev_has_target = True

        error_x = (tx - center)
        error_y = (ty - center)

        error_dist = (error_x * error_x + error_y * error_y) ** 0.5
        deadzone_enter = max(0.0, cfg.deadzone_px)
        deadzone_exit = deadzone_enter + max(0.0, cfg.deadzone_hysteresis_px)
        if self.prev_deadzone_active:
            in_deadzone = error_dist <= deadzone_exit
        else:
            in_deadzone = error_dist <= deadzone_enter
        self.prev_deadzone_active = in_deadzone

        # 🔥 Динамический PID: kp выше для дальних, kd выше для близких
        base_kp = cfg.pid_kp
        base_kd = cfg.pid_kd
        base_ki = cfg.pid_ki
        error_norm = clamp(error_dist / max(1.0, cfg.combat_fov * 0.5), 0.0, 1.5)
        kp_scale = 1.0 + 1.8 * error_norm  # Чем дальше, тем агрессивнее рывок
        kd_scale = 1.0 + 1.6 * (1.0 - min(error_norm, 1.0))  # Чем ближе, тем больше демпфирование
        ki_scale = 0.55 + 0.65 * error_norm  # На близкой цели меньше интеграла, чтобы не "трясло"
        self.pid_x.kp = base_kp * kp_scale
        self.pid_x.kd = base_kd * kd_scale
        self.pid_x.ki = base_ki * ki_scale
        self.pid_y.kp = base_kp * kp_scale
        self.pid_y.kd = base_kd * kd_scale
        self.pid_y.ki = base_ki * ki_scale

        if in_deadzone:
            dx = 0.0
            dy = 0.0
        else:
            dx = self.pid_x.step(error_x, dt)
            dy = self.pid_y.step(error_y, dt)

        # 🔥 Dynamic PID gains
        kp_mult = 1.0 + (error_dist / 100.0) * 0.5  # Boost kp на big error
        dx *= kp_mult * cfg.aim_gain_x
        dy *= kp_mult * cfg.aim_gain_y

        # Динамический лимит шага: быстрое захватывание + ускоренная доводка без "рывков"
        if error_dist < 50:  # Близко — медленно
            dynamic_step = cfg.aim_max_step_px * 0.4  # 10px
        elif error_dist < 150:  # Средне
            dynamic_step = cfg.aim_max_step_px * 0.8  # 20px
        else:  # Далеко — быстрее snap
            dynamic_step = cfg.aim_max_step_px  # 25px

        dx = clamp(dx, -dynamic_step, dynamic_step)
        dy = clamp(dy, -dynamic_step, dynamic_step)

        # 🔥 Velocity smoothing (human-like accel)
        target_vel_x = dx / dt
        target_vel_y = dy / dt
        accel_limit = cfg.max_accel_px_per_sec2 * dt
        self.prev_vel_x = self.prev_vel_x + clamp(target_vel_x - self.prev_vel_x, -accel_limit, accel_limit)
        self.prev_vel_y = self.prev_vel_y + clamp(target_vel_y - self.prev_vel_y, -accel_limit, accel_limit)
        move_x = self.prev_vel_x * dt
        move_y = self.prev_vel_y * dt

        # Exp average (alpha=0.3-0.6)
        alpha = 0.4 if error_dist > 50 else 0.2
        move_x = alpha * move_x + (1 - alpha) * self.prev_move_x
        move_y = alpha * move_y + (1 - alpha) * self.prev_move_y

        self.prev_move_x, self.prev_move_y = move_x, move_y
        self.prev_vel_x, self.prev_vel_y = self.prev_vel_x, self.prev_vel_y  # Save

        t_input_start = time.time()
        queue_size_before = self.input_queue.qsize()
        added_commands = 0

        if not desktop_mode:
            self.input_queue.put(('key_up', W_KEY))
            added_commands += 1
            self.input_queue.put(('key_down', CTRL_KEY))
            added_commands += 1
            if cfg.use_bezier:
                # Кривые Безье: разбиваем на cfg.bezier_steps шагов
                prev_bx = 0.0
                prev_by = 0.0
                for t in np.linspace(0, 1, cfg.bezier_steps + 1)[1:]:
                    bx = bezier_t(t, 0, dx / 2, dx)
                    by = bezier_t(t, 0, dy / 2 + (cfg.bezier_intensity if dy > 0 else -cfg.bezier_intensity), dy)
                    self.input_queue.put(('move', bx - prev_bx, by - prev_by))
                    added_commands += 1
                    prev_bx = bx
                    prev_by = by
            else:
                target_move_x = dx * cfg.aim_gain_x
                target_move_y = dy * cfg.aim_gain_y

                # Плавная динамика мыши: ограничение ускорения + адаптивное сглаживание
                dt_cmd = max(dt, 0.001)
                accel_limit = max(1.0, cfg.max_accel_px_per_sec2 * dt_cmd)
                move_x = self.prev_move_x + clamp(target_move_x - self.prev_move_x, -accel_limit, accel_limit)
                move_y = self.prev_move_y + clamp(target_move_y - self.prev_move_y, -accel_limit, accel_limit)

                blend = 0.35 if error_dist > cfg.center_radius_px * 5 else 0.18
                move_x = (1.0 - blend) * move_x + blend * target_move_x
                move_y = (1.0 - blend) * move_y + blend * target_move_y

                # Анти-дрожание около центра
                if abs(error_x) < 0.9:
                    move_x = 0.0
                if abs(error_y) < 0.9:
                    move_y = 0.0

                max_speed_step = max(1.0, cfg.max_speed_px_per_sec * dt_cmd)
                max_out = min(dynamic_step * max(cfg.aim_gain_x, cfg.aim_gain_y), max_speed_step)
                move_x = clamp(move_x, -max_out, max_out)
                move_y = clamp(move_y, -max_out, max_out)

                self.input_queue.put(('move', move_x, move_y))
                self.prev_move_x = move_x
                self.prev_move_y = move_y
                added_commands += 1
            if now - self._last_target_log > 0.25:
                self.log(f"🎯 PID AIM: dx={dx:.1f} dy={dy:.1f} err=({error_x:.0f},{error_y:.0f}) dt={dt*1000:.0f}ms")
        else:
            cur_x, cur_y = self.input.get_cursor_pos()
            target_x = int(reg["left"] + tx)
            target_y = int(reg["top"] + ty)
            if cfg.desktop_force_absolute:
                self.input_queue.put(('move_absolute', target_x, target_y))  # Absolute move
                added_commands += 1
            else:
                raw_dx = (target_x - cur_x)
                raw_dy = (target_y - cur_y)
                dx = clamp(raw_dx, -cfg.aim_max_step_px, cfg.aim_max_step_px)
                dy = clamp(raw_dy, -cfg.aim_max_step_px, cfg.aim_max_step_px)
                self.input_queue.put(('move', dx, dy))  # Relative fallback
                added_commands += 1

        in_center = abs(tx - center) <= cfg.center_radius_px and abs(ty - center) <= cfg.center_radius_px
        can_shoot = (cfg.desktop_test_autoclick if desktop_mode else cfg.auto_shoot)
        if can_shoot and in_center:
            # 🔥 Рандомизация cooldown
            randomized_cooldown = cfg.shoot_cooldown_sec + random.uniform(-cfg.random_timing_variance, cfg.random_timing_variance)
            randomized_cooldown = max(0.03, randomized_cooldown)  # Min bound
            if now - self.last_shot_time >= randomized_cooldown:
                for _ in range(cfg.burst_shots):  # 🔥 Burst
                    # 🔥 Рандомизация click delay
                    randomized_click_delay = cfg.shoot_click_delay_sec + random.uniform(-cfg.random_timing_variance, cfg.random_timing_variance)
                    randomized_click_delay = max(0.005, randomized_click_delay)  # Min bound
                    self.input_queue.put(('shoot', randomized_click_delay))
                    added_commands += 1
                self.input_queue.put(('move', 0, cfg.recoil_comp_px))  # 🔥 Recoil comp
                added_commands += 1
                self.last_shot_time = now
                self.log("Сделан выстрел по цели")

        queue_size_after = self.input_queue.qsize()
        t_input = (time.time() - t_input_start) * 1000
        executed = added_commands - (queue_size_after - queue_size_before)  # Если очередь не изменилась, executed = added

        t_post = (time.time() - t_post_start) * 1000

        self.t_post_list.append(t_post)
        self.t_input_list.append(t_input)

        if now - self._last_perf_log > 0.35:
            self.log(f"T_infer: {t_infer:.1f}ms, T_post: {t_post:.1f}ms, T_input: {t_input:.1f}ms (queue: {queue_size_before}->{queue_size_after}, executed: {executed})")
            self._last_perf_log = now

        if now - self._last_debug_dump > 0.08:
            self._write_aim_debug({
                "event": "tracking",
                "target_conf": round(float(conf), 4),
                "error_x": round(float(error_x), 3),
                "error_y": round(float(error_y), 3),
                "error_dist": round(float(error_dist), 3),
                "dx": round(float(dx), 3),
                "dy": round(float(dy), 3),
                "move_x": round(float(self.prev_move_x), 3),
                "move_y": round(float(self.prev_move_y), 3),
                "dynamic_step": round(float(dynamic_step), 3),
                "in_center": bool(in_center),
                "queue_before": int(queue_size_before),
                "queue_after": int(queue_size_after),
                "t_infer_ms": round(t_infer, 3),
                "t_post_ms": round(t_post, 3),
                "t_input_ms": round(t_input, 3),
                "t_queue_ms": round(t_queue, 3),
                "out_frame_id": out_frame_id,
                "expected_frame_id": frame_id,
            })
            self._last_debug_dump = now

        return True, frame_bgr

    def _navigation_step(self) -> None:
        mm_region = {"left": self.MINIMAP_REGION["left"], "top": self.MINIMAP_REGION["top"], "width": self.MINIMAP_REGION["width"], "height": self.MINIMAP_REGION["height"]}
        try:
            mm = np.array(self.cam.grab(mm_region))
        except Exception:
            return
        minimap = cv2.cvtColor(mm, cv2.COLOR_BGRA2BGR) if mm.shape[2] == 4 else mm

        pos = self._detect_arrow_position(minimap)
        if pos is None:
            return

        if self.current_navigator is None or not self.current_navigator.has_points():
            auto_route = self.spawn_detector.auto_detect_spawn()
            if auto_route is not None:
                self.log(f"Автовыбор маршрута по спавну: {auto_route.name}")
                self._load_route(auto_route)
            return

        self.current_navigator.mouse_gain = self.state.config.nav_mouse_gain

        x, y = pos
        self.current_navigator.advance_if_reached(x, y)
        turn = self.current_navigator.compute_camera_turn(x, y)
        self.input_queue.put(('move', turn, 0))

        self.input_queue.put(('key_up', CTRL_KEY))
        self.input_queue.put(('key_down', W_KEY))

    def run(self) -> None:  # 🔥 Теперь это метод QThread
        sw, sh = self._get_screen_size()
        self.log("Поток бота запущен")
        self.log(f"Debug-лог наведения: {self.aim_debug_file}")

        while not self.stop_event.is_set():
            cfg = self.state.config
            if cfg.desktop_test_mode != self._desktop_prev:
                self.log(f"Режим проверки на рабочем столе: {'ВКЛ' if cfg.desktop_test_mode else 'ВЫКЛ'}")
                self._desktop_prev = cfg.desktop_test_mode

            combat_fov = int(cfg.combat_fov)
            reg = {
                "left": int(sw // 2 - combat_fov // 2),
                "top": int(sh // 2 - combat_fov // 2),
                "width": combat_fov,
                "height": combat_fov,
            }
            center = combat_fov // 2

            self.recorder.update()

            if self.state.manual_route is not None:
                self._load_route(self.state.manual_route)
                self.state.manual_route = None

            t_capture_start = time.time()
            region_tuple = {"left": reg["left"], "top": reg["top"], "width": reg["width"], "height": reg["height"]}
            try:
                frame = np.array(self.cam.grab(region_tuple))
            except Exception:
                continue
            fov_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame
            cv2.circle(fov_frame, (center, center), 4, (255, 255, 255), -1)
            cv2.circle(fov_frame, (center, center), max(8, center - 2), (255, 180, 0), 1)
            t_capture = (time.time() - t_capture_start) * 1000

            self.t_capture_list.append(t_capture)

            if t_capture > 20:  # 🔥 Adaptive sleep
                time.sleep(0.005)

            has_target = False
            if self.state.running:
                while self.input_yolo_q.qsize() > 1:
                    try:
                        self.input_yolo_q.get_nowait()
                    except Exception:
                        break
                has_target, fov_frame = self._combat_step(fov_frame, center, reg, desktop_mode=cfg.desktop_test_mode)
                if not has_target or not cfg.nav_pause_when_enemy:
                    self._navigation_step()
            else:
                self.input_queue.put(('release_all',))
                self.state.target_lock_signal.emit([])

            self.state.frame_signal.emit(fov_frame)

            # 🔥 Расчет FPS
            self.frame_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                t_capture_avg = sum(self.t_capture_list) / len(self.t_capture_list) if self.t_capture_list else 0
                t_infer_avg = sum(self.t_infer_list) / len(self.t_infer_list) if self.t_infer_list else 0
                t_post_avg = sum(self.t_post_list) / len(self.t_post_list) if self.t_post_list else 0
                t_input_avg = sum(self.t_input_list) / len(self.t_input_list) if self.t_input_list else 0
                latency_avg = sum(self.latency_list) / len(self.latency_list) if self.latency_list else 0
                self.state.performance_signal.emit(self.fps, latency_avg, t_capture_avg, t_infer_avg, t_post_avg, t_input_avg)
                self.frame_count = 0
                self.fps_start_time = time.time()
                self.t_capture_list = []
                self.t_infer_list = []
                self.t_post_list = []
                self.t_input_list = []
                self.latency_list = []

            # 🔥 Рандомизация cycle sleep
            randomized_sleep = cfg.cycle_sleep_sec + random.uniform(-cfg.random_timing_variance, cfg.random_timing_variance)
            randomized_sleep = max(0.001, randomized_sleep)  # Min bound
            time.sleep(randomized_sleep)  # 🔥 Configurable sleep

    def stop(self):
        self.stop_event.set()
        self.input_yolo_q.put(None)
        self.yolo_proc.join()


def toggle_bot(bot, state):
    state.running = not state.running
    state.status_signal.emit("Статус: Запущен" if state.running else "Статус: Остановлен")
    bot.log("Бот запущен" if state.running else "Бот остановлен")
    if not state.running:
        bot.input_queue.put(('release_all',))


def toggle_desktop(bot, state):
    state.config.desktop_test_mode = not state.config.desktop_test_mode
    bot.log(f"Режим проверки на рабочем столе: {'ВКЛ' if state.config.desktop_test_mode else 'ВЫКЛ'}")


def toggle_recorder(bot, state):
    bot.recorder.toggle(bot.route_name_for_record)
    status = "запись ВКЛ" if bot.recorder.is_recording else "запись ВЫКЛ"
    state.status_signal.emit(f"Статус: {status}")
    state.recorder_status_signal.emit("ВКЛ" if bot.recorder.is_recording else "ВЫКЛ")  # Эмитим сигнал для оверлея
    bot.log(f"Рекордер: {status}")
    state.routes_signal.emit([p.name for p in sorted(bot.routes_dir.glob('*.txt'))])


def main() -> None:
    app = QApplication([])
    base = Path(__file__).resolve().parent

    state = BotState()
    window = CombatWalkBotWindow(state=state, routes_dir=base / "routes")
    window.show()

    overlay = StatusOverlay()
    overlay.show()
    state.status_signal.connect(overlay.update_status)

    recorder_overlay = RecorderOverlay()  # Новый оверлей для записи
    recorder_overlay.show()
    state.recorder_status_signal.connect(recorder_overlay.update_status)

    fov_overlay = FovOverlay()
    fov_overlay.set_radius(state.config.combat_fov)
    fov_overlay.show()
    state.fov_changed_signal.connect(fov_overlay.set_radius)
    state.status_signal.connect(lambda s: fov_overlay.set_active("Запущен" in s))

    target_lock_overlay = TargetLockOverlay()
    target_lock_overlay.show()
    state.target_lock_signal.connect(target_lock_overlay.set_boxes)
    state.status_signal.connect(lambda s: target_lock_overlay.set_active("Запущен" in s or "ВКЛ" in s))

    # 🔥 Новый оверлей для FPS/Latency
    performance_overlay = PerformanceOverlay()
    performance_overlay.show()
    state.performance_signal.connect(performance_overlay.update_performance)

    bot = CombatWalkBot(state=state, base_dir=base)
    bot.start()  # 🔥 Запуск QThread

    # 🔥 Горячие клавиши с pynput
    def on_press(key):
        try:
            if key == Key.f9:
                toggle_bot(bot, state)
            elif key == Key.f8:
                toggle_desktop(bot, state)
            elif key == Key.f10:
                toggle_recorder(bot, state)
        except AttributeError:
            pass

    listener = Listener(on_press=on_press)
    listener.start()

    app.exec()

    listener.stop()

    # 🔥 Stop bot
    bot.stop()
    bot.wait()


if __name__ == "__main__":
    main()
