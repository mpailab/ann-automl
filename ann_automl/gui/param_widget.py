from typing import Any, Callable, Dict, Optional
from bokeh.models import Select, Slider, MultiChoice, CheckboxGroup, \
                         DatePicker, TextInput, FuncTickFormatter
from datetime import date
import numpy as np
import traceback

from ..core.nnfuncs import param_values

Callback = Callable[[Any, Any, Any], None]

class ParamWidget(object):

    _widgets = {}

    def __init__(self, name: str, desc: Dict[str, Any]):

        assert 'gui' in desc and 'default' in desc and 'title' in desc and \
               'group' in desc['gui'] and 'widget' in desc['gui'] and \
                name not in self._widgets

        title = f"{desc['title']}"
        if desc['gui']['widget'] != 'Checkbox':
            title += ':'
        value = desc['default']
        kwargs = {
            'name': name,
            'tags': [desc['gui']['group']],
            'css_classes': ['ann-automl-align'],
            'min_height': 50,
            'sizing_mode': 'stretch_width',
            'margin': (5, 35, 5, 20)
        }

        prefix = name.split('_')[0]

        def default_getter():
            return self._obj.value

        def default_setter(value):
            self._obj.value = value

        def get_values():
            return self._obj.options

        def set_values(values):
            self._obj.options = values

        self._attr = 'value'
        self._title = title
        self._value = value
        self._default_value = value
        self._clear_value = None
        self._getter = default_getter
        self._setter = default_setter
        self._values_getter = None
        self._values_setter = None
        self._conditions = {}
        self._dependencies = []

        if desc['gui']['widget'] == 'Select':
            self._obj = Select(title=title, value=value,
                               options=[x for x in desc['values']], **kwargs)
            self._values_getter = get_values
            self._values_setter = set_values

        elif desc['gui']['widget'] == 'MultiChoice':
            self._obj = MultiChoice(title=title, value=value,
                                    options=[x for x in desc['values']], **kwargs)
            self._values_getter = get_values
            self._values_setter = set_values

        elif desc['gui']['widget'] == 'Slider':

            str_values, cur_index = param_values(
                return_str=True, **{**desc, 'default': value})
            self._values, _ = param_values(**desc)

            try:
                formatter = FuncTickFormatter(
                    code=f"const labels = {str_values};\nreturn labels[tick];")
            except Exception:
                traceback.print_exc()
                raise

            kwargs['min_height'] = 40
            self._obj = Slider(title=title, value=cur_index,
                               start=0, end=len(self._values)-1,
                               step=1, format=formatter, **kwargs)

            def slicer_getter():
                return self._values[max(0,min(self._obj.value,len(self._values)-1))]

            def slicer_setter(value):
                self._obj.value = np.argmin(np.abs(np.array(self._values) - value))

            self._getter = slicer_getter
            self._setter = slicer_setter

        elif desc['gui']['widget'] == 'Checkbox':
            kwargs['min_height'] = 20
            self._obj = CheckboxGroup(labels=[title],
                                      active=[0] if value else [], **kwargs)

            def checkbox_getter() -> bool:
                return len(self._obj.active) > 0

            def checkbox_setter(value: bool):
                self._obj.active = [0] if value else []

            self._attr = 'active'
            self._getter = checkbox_getter
            self._setter = checkbox_setter

        elif desc['gui']['widget'] == 'Text':
            self._obj = TextInput(title=title, value=value, **kwargs)
            self._clear_value = ""

        elif desc['gui']['widget'] == 'Date':
            if value is None:
                self._value = date.today()
                self._default_value = self._value
                self._obj = DatePicker(title=title, value=date.today(), **kwargs)
            else:
                self._obj = DatePicker(title=title, value=value, **kwargs)

            def date_setter(value):
                try:
                    self._obj.value = value
                except ValueError:
                    self._obj.value = date.today()

            self._setter = date_setter
            self._clear_value = date.today()

        else:
            raise ValueError(f'Unsupported widget type {desc["gui"]["widget"]}')

        self._widgets[name] = self

        if 'cond' in desc:
            for p, vs in desc['cond']:
                widget = self._widgets[p]
                self._conditions[widget] = vs
                widget.add_dependence(self)

        def default_callback(attr, old, new):
            for widget in self._dependencies:
                if widget.active():
                    widget.activate()
                else:
                    widget.hide()

        self._callbacks = [default_callback]

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return self.name != other.name

    def add_dependence(self, widget):
        self._dependencies.append(widget)

    @property
    def name(self):
        return self._obj.name

    @property
    def group(self):
        return self._obj.tags[0]

    @property
    def title(self):
        return self._title

    @property
    def value(self):
        return self._getter()

    @value.setter
    def value(self, value):
        self._setter(value)

    @property
    def values(self):
        assert self._values_getter is not None
        return self._values_getter()

    @values.setter
    def values(self, values):
        assert self._values_setter is not None
        self._values_setter(values)

    def active(self):
        return all (w.value in vs for w,vs in self._conditions)

    def activate(self):
        self._obj.visible = True

    def hide(self):
        self._obj.visible = False

    def enable(self):
        self._obj.disabled = False

    def disable(self):
        self._obj.disabled = True

    def on_change(self, *callbacks: Callback):
        self._callbacks.extend(callbacks)
        self._obj.on_change(self._attr, *self._callbacks)

    def clear(self):
        assert self._clear_value is not None
        self.value = self._clear_value

    def reset(self):
        self.value = self._value

    def default(self):
        self.value = self._default_value

    def set_default(self, value):
        self._default_value = value
        self._value = value
        self.value = value

    def update(self):
        self._value = self.value

    @property
    def interface(self):
        return self._obj