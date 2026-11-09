from ui.controls import controls, tooltip_label


class Target:
    def __init__(self):
        self.markup = []
        self.arguments = None
    def html(self, value):
        self.markup.append(value)
    def number_input(self, label, *args, **kwargs):
        self.arguments = kwargs
        return kwargs.get("value", 1)


def test_one_custom_tooltip_and_no_browser_title_or_native_help():
    target = Target()
    assert controls.number_input("Example", value=3, help='Explanation <tag> "quote"', _container=target) == 3
    assert len(target.markup) == 1
    assert target.markup[0].count('role="tooltip"') == 1
    assert ' title=' not in target.markup[0]
    assert "help" not in target.arguments
    assert '&lt;tag&gt;' in target.markup[0]


def test_existing_engine_label_does_not_gain_second_tooltip():
    target = Target()
    tooltip_label("Threads", "Description", target)
    controls.number_input("Threads", value=4, help="Description", label_visibility="collapsed", _container=target)
    assert len(target.markup) == 1 and "help" not in target.arguments
