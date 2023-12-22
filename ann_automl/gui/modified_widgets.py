import bokeh

def Box(*args, **kwargs):
    return bokeh.models.Column(*args, **kwargs, spacing=10, height=700, height_policy='fixed',
                               css_classes=['ann-automl-shadow-border', 'ann-automl-scroll'],
                               margin=(10, 10, 10, 10))

def AnswerBox(*args, **kwargs):
    return bokeh.models.Div(*args, **kwargs, #min_width=800, height_policy = "max", width_policy = 'max',
                            sizing_mode='stretch_height',
                            css_classes=['answer-shadowbox'],
                            disabled=True)

def RequestBox(*args, **kwargs):
    return bokeh.models.Div(*args, **kwargs, #min_width=800, height_policy = "max", width_policy = 'max',
                            sizing_mode='stretch_height',
                            css_classes=['request-shadowbox'],
                            disabled=True)

def Button(label, on_click_func, *args, js=False, **kwargs):
    assert 'label' not in kwargs
    if 'button_type' not in kwargs:
        kwargs['button_type'] = 'primary'
    if 'width' not in kwargs:
        kwargs['width'] = 8*len(label) + 50
    button = bokeh.models.Button(*args, **kwargs, label=label)
    if js:
        button.js_on_click(on_click_func)
    else:
        button.on_click(on_click_func)
    return button


def Delimiter(*args, **kwargs, ):
    return bokeh.models.Spacer(*args, **kwargs, height=3, background="#b8b8b8",
                               margin=(10, 30, 10, 15))


def Table(source, columns, *args, **kwargs):
    return bokeh.models.DataTable(*args, **kwargs, source=source, columns=columns,
                                    index_position=None, autosize_mode='fit_columns',
                                    sizing_mode='stretch_both',
                                    css_classes=['ann-automl-shadow-border', 'ann-automl-scroll'],
                                    margin=(10, 10, 10, 10))


def Toggle(label, on_click_func, *args, **kwargs):
    button = bokeh.models.Toggle(*args, **kwargs, label=label,
                                    button_type='primary', width=8*len(label) + 50)
    button.on_click(on_click_func)
    return button