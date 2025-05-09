import PySide6.QtDesigner as QtDesigner

from wvw.kernel_view import KernelView


if __name__ == "__main__":
    QtDesigner.QPyDesignerCustomWidgetCollection.registerCustomWidget(
        KernelView,
        module="wvw.kernel_view",
        tool_tip="A neural network kernel visualization widget.",
        xml="""
        <ui language="c++">
            <widget class="KernelView" name="kernel_view">
                <property name="geometry">
                    <rect>
                        <x>0</x>
                        <y>0</y>
                        <width>100</width>
                        <height>100</height>
                    </rect>
                </property>
            </widget>
        </ui>""",
    )
