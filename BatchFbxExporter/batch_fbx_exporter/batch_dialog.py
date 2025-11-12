# -*- coding: utf-8 -*-
"""
Batch Export Dialog with Index Range and Attribute Mapping
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

__author__ = "Maek"
__date__ = "2025-11-11"

import os
import tempfile
from functools import partial
from PySide2 import QtWidgets, QtCore, QtGui


class BatchExportDialog(object):

    title = "Batch FBX Export Settings"
    space = "%-15s"

    edit_config = {
        "POSITION": space % "Vertex Position",
        "TANGENT": space % "Vertex Tangent",
        "NORMAL": space % "Vertex Normal",
        "BINORMAL": space % "Vertex BiNormal",
        "COLOR": space % "Vertex Color",
        "UV": space % "UV",
        "UV2": space % "UV2",
        "UV3": space % "UV3",
        "UV4": space % "UV4",
        "UV5": space % "UV5",
    }

    button_dict = {}
    mapper = {}
    start_index = 0
    end_index = 1000
    output_folder = ""

    def __init__(self, mqt):
        self.mqt = mqt
        name = "RenderDoc_%s.ini" % self.__class__.__name__
        path = os.path.join(tempfile.gettempdir(), name)
        self.settings = QtCore.QSettings(path, QtCore.QSettings.IniFormat)
        if not os.path.exists(path):
            self.template_select(0)

    def template_select(self, index):
        text = self.combo.itemText(index) if hasattr(self, "combo") else "unity"
        config = {}
        if text == "unity":
            config = {
                "POSITION": "POSITION",
                "TANGENT": "TANGENT",
                "BINORMAL": "",
                "NORMAL": "NORMAL",
                "COLOR": "COLOR",
                "UV": "TEXCOORD0",
                "UV2": "TEXCOORD1",
                "UV3": "TEXCOORD2",
                "UV4": "TEXCOORD3",
                "UV5": "TEXCOORD4",
            }
        elif text == "unreal":
            config = {
                "POSITION": "ATTRIBUTE0",
                "TANGENT": "ATTRIBUTE1",
                "BINORMAL": "",
                "NORMAL": "ATTRIBUTE2",
                "COLOR": "ATTRIBUTE13",
                "UV": "ATTRIBUTE5",
                "UV2": "ATTRIBUTE6",
                "UV3": "ATTRIBUTE7",
                "UV4": "ATTRIBUTE8",
                "UV5": "ATTRIBUTE9",
            }

        self.settings.setValue("Engine", text)
        for name, input_widget in self.button_dict.items():
            value = config.get(name, "")
            self.settings.setValue(name, value)
            self.mqt.SetWidgetText(input_widget.edit, value)

    def browse_folder(self, context, widget, text):
        """Browse for output folder"""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select Output Folder", self.output_folder
        )
        if folder:
            self.output_folder = folder
            self.mqt.SetWidgetText(self.folder_edit, folder)
            self.settings.setValue("OutputFolder", folder)

    def init_ui(self):
        self.widget = self.mqt.CreateToplevelWidget(self.title, None)

        # ==================== Index Range Section ====================
        index_group = self.mqt.CreateGroupBox(True)
        self.mqt.SetWidgetText(index_group, "Export Range")
        
        # Index range in one row (more compact)
        index_container = self.mqt.CreateHorizontalContainer()
        
        # Start Index
        start_label = self.mqt.CreateLabel()
        self.mqt.SetWidgetText(start_label, "Start:")
        self.start_edit = self.mqt.CreateTextBox(True, self.on_start_change)  # ✅ True = 单行
        start_value = self.settings.value("StartIndex", "0")
        self.mqt.SetWidgetText(self.start_edit, start_value)
        
        # Convert to QLineEdit for better control
        if hasattr(self.start_edit, 'Widget'):
            start_qt_edit = self.start_edit.Widget()
        else:
            start_qt_edit = self.start_edit
        
        if isinstance(start_qt_edit, QtWidgets.QLineEdit):
            start_qt_edit.setMaxLength(6)
            start_qt_edit.setFixedWidth(80)
            start_qt_edit.setFixedHeight(24)  # 设置固定高度
            start_qt_edit.setAlignment(QtCore.Qt.AlignRight)
            # Only allow digits
            start_qt_edit.setValidator(QtGui.QIntValidator(0, 999999))
            start_qt_edit.setPlaceholderText("0")
        
        # Separator
        separator_label = self.mqt.CreateLabel()
        self.mqt.SetWidgetText(separator_label, "  to  ")
        
        # End Index
        end_label = self.mqt.CreateLabel()
        self.mqt.SetWidgetText(end_label, "End:")
        self.end_edit = self.mqt.CreateTextBox(True, self.on_end_change)  # ✅ True = 单行
        end_value = self.settings.value("EndIndex", "1000")
        self.mqt.SetWidgetText(self.end_edit, end_value)
        
        # Convert to QLineEdit for better control
        if hasattr(self.end_edit, 'Widget'):
            end_qt_edit = self.end_edit.Widget()
        else:
            end_qt_edit = self.end_edit
            
        if isinstance(end_qt_edit, QtWidgets.QLineEdit):
            end_qt_edit.setMaxLength(6)
            end_qt_edit.setFixedWidth(80)
            end_qt_edit.setFixedHeight(24)  # 设置固定高度
            end_qt_edit.setAlignment(QtCore.Qt.AlignRight)
            # Only allow digits
            end_qt_edit.setValidator(QtGui.QIntValidator(0, 999999))
            end_qt_edit.setPlaceholderText("999999")
        
        # Add all to one row
        self.mqt.AddWidget(index_container, start_label)
        self.mqt.AddWidget(index_container, self.start_edit)
        self.mqt.AddWidget(index_container, separator_label)
        self.mqt.AddWidget(index_container, end_label)
        self.mqt.AddWidget(index_container, self.end_edit)
        
        # Add spacer to push everything to the left
        spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        index_container.layout().addItem(spacer)
        
        self.mqt.AddWidget(index_group, index_container)
        self.mqt.AddWidget(self.widget, index_group)

        # ==================== Output Folder Section ====================
        folder_group = self.mqt.CreateGroupBox(True)
        self.mqt.SetWidgetText(folder_group, "Output Settings")
        
        folder_container = self.mqt.CreateHorizontalContainer()
        folder_label = self.mqt.CreateLabel()
        self.mqt.SetWidgetText(folder_label, "Output Folder:")
        self.folder_edit = self.mqt.CreateTextBox(True, None)  # ✅ True = 单行
        self.output_folder = self.settings.value("OutputFolder", "F:/BatchExport")
        self.mqt.SetWidgetText(self.folder_edit, self.output_folder)
        
        # 设置folder_edit的固定高度
        if hasattr(self.folder_edit, 'Widget'):
            folder_qt_edit = self.folder_edit.Widget()
        else:
            folder_qt_edit = self.folder_edit
        
        if isinstance(folder_qt_edit, QtWidgets.QLineEdit):
            folder_qt_edit.setFixedHeight(24)  # 设置固定高度
        
        browse_button = self.mqt.CreateButton(self.browse_folder)
        self.mqt.SetWidgetText(browse_button, "Browse...")
        
        # 设置browse按钮的固定高度
        if hasattr(browse_button, 'Widget'):
            browse_qt_button = browse_button.Widget()
        else:
            browse_qt_button = browse_button
        
        if isinstance(browse_qt_button, QtWidgets.QPushButton):
            browse_qt_button.setFixedHeight(24)  # 设置固定高度
        
        self.mqt.AddWidget(folder_container, folder_label)
        self.mqt.AddWidget(folder_container, self.folder_edit)
        self.mqt.AddWidget(folder_container, browse_button)
        self.mqt.AddWidget(folder_group, folder_container)
        self.mqt.AddWidget(self.widget, folder_group)

        # ==================== Attribute Mapping Section ====================
        attr_group = self.mqt.CreateGroupBox(True)
        self.mqt.SetWidgetText(attr_group, "Attribute Mapping")
        
        # Template selection
        template_container = self.mqt.CreateHorizontalContainer()
        template_label = self.mqt.CreateLabel()
        
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["unity", "unreal"])
        self.combo.setCurrentText(self.settings.value("Engine", "unity"))
        self.combo.currentIndexChanged.connect(self.template_select)
        self.combo.setFixedHeight(24)  # 设置固定高度
        self.combo.setFixedWidth(120)  # 设置固定宽度

        self.mqt.SetWidgetText(template_label, "Template:")
        self.mqt.AddWidget(template_container, template_label)
        self.mqt.AddWidget(template_container, self.combo)
        
        # Add spacer for template
        template_spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        template_container.layout().addItem(template_spacer)
        
        self.mqt.AddWidget(attr_group, template_container)

        # Attribute inputs
        self.button_dict = {}
        for name, label in self.edit_config.items():
            w = self.input_widget(label, name)
            self.button_dict[name] = w
            self.mqt.AddWidget(attr_group, w)
            # Load settings
            text = self.settings.value(name, "")
            if text and text != name:
                self.mqt.SetWidgetText(w.edit, text)
        
        self.mqt.AddWidget(self.widget, attr_group)

        # ==================== Action Buttons ====================
        button_container = self.mqt.CreateHorizontalContainer()
        ok_button = self.mqt.CreateButton(self.accept)
        self.mqt.SetWidgetText(ok_button, "Export")
        callback = lambda *args: self.mqt.CloseCurrentDialog(False)
        cancel_button = self.mqt.CreateButton(callback)
        self.mqt.SetWidgetText(cancel_button, "Cancel")
        
        # 设置按钮固定高度
        if hasattr(ok_button, 'Widget'):
            ok_qt = ok_button.Widget()
        else:
            ok_qt = ok_button
        if isinstance(ok_qt, QtWidgets.QPushButton):
            ok_qt.setFixedHeight(28)
            ok_qt.setMinimumWidth(80)
        
        if hasattr(cancel_button, 'Widget'):
            cancel_qt = cancel_button.Widget()
        else:
            cancel_qt = cancel_button
        if isinstance(cancel_qt, QtWidgets.QPushButton):
            cancel_qt.setFixedHeight(28)
            cancel_qt.setMinimumWidth(80)
        
        self.mqt.AddWidget(button_container, cancel_button)
        self.mqt.AddWidget(button_container, ok_button)
        self.mqt.AddWidget(self.widget, button_container)

        return self.widget

    def on_start_change(self, c, w, text):
        """Handle start index change"""
        try:
            self.start_index = int(text)
            self.settings.setValue("StartIndex", text)
        except ValueError:
            pass

    def on_end_change(self, c, w, text):
        """Handle end index change"""
        try:
            self.end_index = int(text)
            self.settings.setValue("EndIndex", text)
        except ValueError:
            pass

    def accept(self, context, widget, text):
        """Accept and collect all settings"""
        self.mapper = {}
        for name, WIDGET in self.button_dict.items():
            text = self.mqt.GetWidgetText(WIDGET.edit)
            self.mapper[name] = text
        
        self.mapper['ENGINE'] = self.combo.currentText()
        
        # Get index range
        try:
            self.start_index = int(self.mqt.GetWidgetText(self.start_edit))
            self.end_index = int(self.mqt.GetWidgetText(self.end_edit))
        except ValueError:
            self.start_index = 0
            self.end_index = 1000
        
        # Get output folder
        self.output_folder = self.mqt.GetWidgetText(self.folder_edit)
        
        self.mqt.CloseCurrentDialog(True)

    def textChange(self, key, c, w, text):
        """Save attribute mapping changes"""
        self.settings.setValue(key, text)

    def input_widget(self, text, edit_text="", type=""):
        """Create input widget for attribute mapping"""
        container = self.mqt.CreateHorizontalContainer()
        label = self.mqt.CreateLabel()
        edit = self.mqt.CreateTextBox(True, partial(self.textChange, edit_text))

        self.mqt.SetWidgetText(label, text)
        self.mqt.SetWidgetText(edit, edit_text)
        
        # 设置edit的固定高度
        if hasattr(edit, 'Widget'):
            edit_qt = edit.Widget()
        else:
            edit_qt = edit
        
        if isinstance(edit_qt, QtWidgets.QLineEdit):
            edit_qt.setFixedHeight(24)  # 设置固定高度
        
        self.mqt.AddWidget(container, label)
        self.mqt.AddWidget(container, edit)

        container.label = label
        container.edit = edit
        return container

